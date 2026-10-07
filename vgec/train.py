"""Evaluate the unadapted model, train response-only QLoRA, and retain the best epoch."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def balanced_limit(records, limit, seed):
    if limit is None:
        return records
    import random
    groups = {label: [] for label in range(5)}
    for record in records:
        groups[record["label"]].append(record)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    selected = []
    while len(selected) < limit and any(groups.values()):
        for group in groups.values():
            if group and len(selected) < limit:
                selected.append(group.pop())
    return selected


def selection_key(metrics):
    """Macro-F1 first, error-only GLEU+ second; ties retain the earlier epoch."""
    return (metrics["classification"]["macro_f1"],
            metrics["corrections"]["error_only"]["gleu_plus"] or 0.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.seed is not None:
        config["seed"] = args.seed
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")
    if int(os.environ.get("WORLD_SIZE", 1)) != 1:
        raise ValueError("Run independent split experiments on one GPU, not torchrun/DDP")
    if not 1 <= config["epochs"] <= 3:
        raise ValueError("The first experiment supports 1–3 epochs")
    accumulation, remainder = divmod(config["effective_batch_size"], config["batch_size"])
    if remainder or accumulation < 1:
        raise ValueError("Effective batch must be a positive multiple of microbatch")

    import torch
    from transformers import AutoConfig, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from .data import load_records, split_records
    from .loading import load_adapter, load_base
    from .metrics import experiment_report
    from .modeling import ResponseCollator, SYSTEM_PROMPT, encode_record, generate_predictions

    set_seed(config["seed"])
    records = load_records(args.data)
    train, validation = split_records(records, config["validation_fraction"], config["seed"])
    train = balanced_limit(train, config["train_limit"], config["seed"])
    validation = balanced_limit(validation, config["validation_limit"], config["seed"] + 1)
    if not train or not validation:
        raise ValueError("Training and validation must both be nonempty")
    resolved = AutoConfig.from_pretrained(config["model_name"], revision=config["revision"])
    revision = resolved._commit_hash or config["revision"]
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    encoded = [encode_record(tokenizer, r, config["max_length"]) for r in train]
    validation_lengths = [len(encode_record(tokenizer, r, config["max_length"])["input_ids"]) for r in validation]
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "train.json", train)
    write_json(output / "validation.json", validation)
    write_json(output / "prompt.json", {"system": SYSTEM_PROMPT})
    majority = max(range(5), key=lambda label: sum(r["label"] == label for r in train))
    identity = [{"id": r["id"], "label": majority, "target": r["source"], "parse_error": None} for r in validation]
    write_json(output / "identity_metrics.json", experiment_report(validation, identity))
    model, dtype = load_base(config["model_name"], revision, args.device,
                             config["load_in_4bit"], config["compute_dtype"])
    print("Evaluating unadapted model", flush=True)
    baseline_predictions = generate_predictions(model, tokenizer, validation, config["max_new_tokens"])
    baseline_metrics = experiment_report(validation, baseline_predictions)
    write_json(output / "baseline_predictions.json", baseline_predictions)
    write_json(output / "baseline_metrics.json", baseline_metrics)
    print(json.dumps({"baseline_macro_f1": baseline_metrics["classification"]["macro_f1"]}), flush=True)
    if config["load_in_4bit"]:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                    gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=config["lora_rank"],
        lora_alpha=config["lora_alpha"], lora_dropout=config["lora_dropout"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none", revision=revision))
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    loading = {"load_in_4bit": config["load_in_4bit"], "compute_dtype": config["compute_dtype"]}
    epoch_history = []
    best = {"key": None, "epoch": None}

    class EpochEvaluation(TrainerCallback):
        def on_epoch_end(self, args, state, control, model=None, **kwargs):
            # Never evaluate or select from nonfinite updates.
            if not all(torch.isfinite(p).all().item() for p in model.parameters() if p.requires_grad):
                raise RuntimeError("Nonfinite adapter parameters")
            if any(not math.isfinite(row["grad_norm"]) for row in state.log_history if "grad_norm" in row):
                raise RuntimeError("Nonfinite gradients; training updates are not trustworthy")
            predictions = generate_predictions(model, tokenizer, validation, config["max_new_tokens"])
            metrics = experiment_report(validation, predictions)
            epoch_dir = output / f"epoch-{len(epoch_history) + 1}"
            epoch_dir.mkdir()
            model.save_pretrained(epoch_dir / "adapter")
            tokenizer.save_pretrained(epoch_dir / "adapter")
            write_json(epoch_dir / "adapter" / "loading.json", loading)
            write_json(epoch_dir / "predictions.json", predictions)
            write_json(epoch_dir / "metrics.json", metrics)
            epoch_history.append({"epoch": state.epoch, "optimizer_steps": state.global_step, "metrics": metrics})
            write_json(output / "epoch_history.json", epoch_history)
            key = selection_key(metrics)
            if best["key"] is None or key > best["key"]:
                best.update(key=key, epoch=state.epoch)
                model.save_pretrained(output / "adapter")
                tokenizer.save_pretrained(output / "adapter")
                write_json(output / "adapter" / "loading.json", loading)
            print(json.dumps({"epoch": state.epoch, "macro_f1": key[0], "error_gleu_plus": key[1], "best_epoch": best["epoch"]}), flush=True)
            model.train()
            return control

    training_args = TrainingArguments(
        output_dir=str(output), use_cpu=args.device == "cpu",
        num_train_epochs=config["epochs"], max_steps=config["max_steps"],
        per_device_train_batch_size=config["batch_size"], gradient_accumulation_steps=accumulation,
        learning_rate=config["learning_rate"], warmup_ratio=0.1, weight_decay=0.01,
        fp16=args.device == "cuda" and dtype == torch.float16,
        bf16=args.device == "cuda" and dtype == torch.bfloat16,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch", logging_steps=1, save_strategy="no", eval_strategy="no", report_to="none",
        label_names=["labels"], seed=config["seed"], data_seed=config["seed"], dataloader_num_workers=0)
    trainer = Trainer(model=model, args=training_args, train_dataset=encoded,
                      data_collator=ResponseCollator(tokenizer.pad_token_id), callbacks=[EpochEvaluation()])
    if training_args.fp16:
        trainer.accelerator.scaler = torch.amp.GradScaler("cuda", init_scale=4096.0)
    result = trainer.train()
    if not math.isfinite(result.training_loss):
        raise RuntimeError(f"Nonfinite training loss: {result.training_loss}")
    write_json(output / "trainer_history.json", trainer.state.log_history)
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainer.model = None
    trainer.model_wrapped = None
    trainer.optimizer = None
    trainer.accelerator.free_memory()
    del model, trainer
    import gc
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()
    reloaded, tokenizer = load_adapter(output / "adapter", args.device)
    predictions = generate_predictions(reloaded, tokenizer, validation, config["max_new_tokens"])
    metrics = experiment_report(validation, predictions)
    write_json(output / "predictions.json", predictions)
    write_json(output / "metrics.json", metrics)
    manifest = {
        "config": config, "resolved_model_revision": revision,
        "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        "train_ids": [r["id"] for r in train], "validation_ids": [r["id"] for r in validation],
        "train_max_tokens": max(len(r["input_ids"]) for r in encoded),
        "validation_max_tokens": max(validation_lengths), "optimizer_steps": result.global_step,
        "training_loss": result.training_loss, "world_size": 1,
        "effective_batch_size": config["effective_batch_size"], "compute_dtype": str(dtype),
        "trainable_parameters": trainable_parameters, "best_epoch": best["epoch"],
        "selection_rule": "highest Macro-F1, then error-only GLEU+; exact ties keep earlier epoch",
        "versions": {name: importlib.metadata.version(name) for name in ["torch", "transformers", "peft", "accelerate", "bitsandbytes"]},
        "devices": [torch.cuda.get_device_name(0)] if args.device == "cuda" else ["cpu"],
        "peak_allocated_vram_bytes_rank0": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
        "saved_adapter_reloaded_for_evaluation": True}
    write_json(output / "run_manifest.json", manifest)
    print(json.dumps({"output_dir": str(output), "best_epoch": best["epoch"], "macro_f1": metrics["classification"]["macro_f1"]}), flush=True)


if __name__ == "__main__":
    main()
