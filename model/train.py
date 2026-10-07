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


def selection_key(metrics, metric_paths=None):
    """Maximize dotted metric paths lexicographically; null values count as zero."""
    if metric_paths is None:
        metric_paths = ("classification.macro_f1", "corrections.error_only.gleu_plus")
    values = []
    for path in metric_paths:
        value = metrics
        for field in path.split("."):
            value = value[field]
        if value is None:
            value = 0.0
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Selection metric {path!r} must be a finite number or null")
        values.append(value)
    return tuple(values)


def main():
    from .config import PROJECT_ROOT, load_config, resolve_output

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data")
    parser.add_argument("--output-dir")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=["cuda", "cpu"])
    args = parser.parse_args()
    config = load_config(args.config)
    seed = args.seed if args.seed is not None else config["run"]["seeds"][0]
    if seed < 0:
        raise ValueError("Seed must be nonnegative")
    device = args.device or config["run"]["device"]
    from tools.kaggle_run import discover_data
    data_path = discover_data(args.data or config["data"]["path"], PROJECT_ROOT,
                              Path(config["data"]["input_root"]), config["data"]["filename"])
    output = resolve_output(config, args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")
    if int(os.environ.get("WORLD_SIZE", 1)) != 1:
        raise ValueError("Run independent split experiments on one GPU, not torchrun/DDP")
    config["run"].update(seeds=[seed], device=device, output_root=str(output.parent))
    config["data"]["path"] = str(data_path)
    training = config["training"]
    settings = config["model"]
    accumulation = training["effective_batch_size"] // training["args"]["per_device_train_batch_size"]
    inference = {"prompt": config["prompt"], "generation": config["generation"]}
    selection_metrics = config["evaluation"]["selection_metrics"]
    adapter_path = output / "adapter"

    import torch
    from transformers import AutoConfig, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from .data import load_training_split
    from .loading import load_adapter, load_base
    from .metrics import experiment_report
    from .modeling import ResponseCollator, encode_record, generate_predictions

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if settings["load_in_4bit"] and device != "cuda":
        raise ValueError("4-bit QLoRA requires CUDA; disable quantization for CPU diagnostics")
    dtype = ((torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
             if device == "cuda" else torch.float32)
    if settings["compute_dtype"] != "auto":
        dtype = getattr(torch, settings["compute_dtype"])
    training_args = TrainingArguments(
        **training["args"], output_dir=str(output), use_cpu=device == "cpu",
        gradient_accumulation_steps=accumulation,
        fp16=device == "cuda" and dtype == torch.float16,
        bf16=device == "cuda" and dtype == torch.bfloat16,
        save_strategy="no", eval_strategy="no", report_to="none",
        label_names=["labels"], seed=seed, data_seed=seed)
    # Validate PEFT settings before fetching the model or creating artifacts.
    lora_config = LoraConfig(**config["lora"], task_type=TaskType.CAUSAL_LM,
                             revision=settings["revision"])
    set_seed(seed)
    train, validation, data_provenance = load_training_split(data_path, config["data"]["validation_fraction"], seed)
    train = balanced_limit(train, config["data"]["train_limit"], seed)
    validation = balanced_limit(validation, config["data"]["validation_limit"], seed + 1)
    if not train or not validation:
        raise ValueError("Training and validation must both be nonempty")
    majority = max(range(5), key=lambda label: sum(r["label"] == label for r in train))
    identity = [{"id": r["id"], "label": majority, "target": r["source"], "parse_error": None} for r in validation]
    identity_metrics = experiment_report(validation, identity)
    selection_key(identity_metrics, selection_metrics)
    resolved = AutoConfig.from_pretrained(settings["name"], revision=settings["revision"])
    revision = resolved._commit_hash or settings["revision"]
    settings["revision"] = revision
    lora_config.revision = revision
    tokenizer = AutoTokenizer.from_pretrained(settings["name"], revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    encoded = [encode_record(tokenizer, r, training["max_length"], config["prompt"]) for r in train]
    validation_lengths = [len(encode_record(tokenizer, r, training["max_length"], config["prompt"])["input_ids"]) for r in validation]
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    write_json(output / "train.json", train)
    write_json(output / "validation.json", validation)
    write_json(output / "prompt.json", {"system": config["prompt"]})
    write_json(output / "identity_metrics.json", identity_metrics)
    model, dtype = load_base(settings, device)

    def predict(current_model, current_tokenizer):
        return generate_predictions(current_model, current_tokenizer, validation,
                                    system_prompt=config["prompt"], generation=config["generation"])

    def save_adapter(current_model, path):
        current_model.save_pretrained(path)
        tokenizer.save_pretrained(path)
        write_json(path / "loading.json", settings)
        write_json(path / "inference.json", inference)

    print("Evaluating unadapted model", flush=True)
    baseline_predictions = predict(model, tokenizer)
    baseline_metrics = experiment_report(validation, baseline_predictions)
    # Resolve the configured selection paths before any optimizer update.
    selection_key(baseline_metrics, selection_metrics)
    write_json(output / "baseline_predictions.json", baseline_predictions)
    write_json(output / "baseline_metrics.json", baseline_metrics)
    print(json.dumps({"baseline_macro_f1": baseline_metrics["classification"]["macro_f1"]}), flush=True)
    if settings["load_in_4bit"]:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_args.gradient_checkpointing,
            gradient_checkpointing_kwargs=training_args.gradient_checkpointing_kwargs)
    model.config.use_cache = False
    model = get_peft_model(model, lora_config)
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
    model.print_trainable_parameters()
    epoch_history = []
    best = {"key": None, "epoch": None}

    class EpochEvaluation(TrainerCallback):
        def on_epoch_end(self, args, state, control, model=None, **kwargs):
            # Never evaluate or select from nonfinite updates.
            if not all(torch.isfinite(p).all().item() for p in model.parameters() if p.requires_grad):
                raise RuntimeError("Nonfinite adapter parameters")
            if any(not math.isfinite(row["grad_norm"]) for row in state.log_history if "grad_norm" in row):
                raise RuntimeError("Nonfinite gradients; training updates are not trustworthy")
            predictions = predict(model, tokenizer)
            metrics = experiment_report(validation, predictions)
            epoch_dir = output / f"epoch-{len(epoch_history) + 1}"
            epoch_dir.mkdir()
            if config["evaluation"]["save_epoch_adapters"]:
                save_adapter(model, epoch_dir / "adapter")
            write_json(epoch_dir / "predictions.json", predictions)
            write_json(epoch_dir / "metrics.json", metrics)
            epoch_history.append({"epoch": state.epoch, "optimizer_steps": state.global_step, "metrics": metrics})
            write_json(output / "epoch_history.json", epoch_history)
            key = selection_key(metrics, selection_metrics)
            if best["key"] is None or key > best["key"]:
                best.update(key=key, epoch=state.epoch)
                save_adapter(model, adapter_path)
            default_key = selection_key(metrics)
            print(json.dumps({"epoch": state.epoch, "macro_f1": default_key[0],
                              "error_gleu_plus": default_key[1], "selection_values": key,
                              "best_epoch": best["epoch"]}), flush=True)
            model.train()
            return control

    trainer = Trainer(model=model, args=training_args, train_dataset=encoded,
                      data_collator=ResponseCollator(tokenizer.pad_token_id), callbacks=[EpochEvaluation()])
    if training_args.fp16:
        trainer.accelerator.scaler = torch.amp.GradScaler("cuda", init_scale=training["fp16_initial_scale"])
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
    if device == "cuda":
        torch.cuda.empty_cache()
    reloaded, tokenizer = load_adapter(adapter_path, device)
    predictions = predict(reloaded, tokenizer)
    metrics = experiment_report(validation, predictions)
    write_json(output / "predictions.json", predictions)
    write_json(output / "metrics.json", metrics)
    manifest = {
        "config": config, "resolved_model_revision": revision,
        "adapter_path": str(adapter_path), "output_dir": str(output), "seed": seed,
        "data_path": str(data_path), "device": device, "model": settings,
        "inference": inference, "training_args": training_args.to_dict(),
        "gradient_accumulation_steps": accumulation, "selection_metrics": selection_metrics,
        "save_epoch_adapters": config["evaluation"]["save_epoch_adapters"],
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "data_provenance": data_provenance,
        "train_ids": [r["id"] for r in train], "validation_ids": [r["id"] for r in validation],
        "train_max_tokens": max(len(r["input_ids"]) for r in encoded),
        "validation_max_tokens": max(validation_lengths), "optimizer_steps": result.global_step,
        "training_loss": result.training_loss, "world_size": 1,
        "effective_batch_size": training["effective_batch_size"], "compute_dtype": str(dtype),
        "trainable_parameters": trainable_parameters, "best_epoch": best["epoch"],
        "selection_rule": "highest " + ", then ".join(selection_metrics) + "; exact ties keep earlier epoch",
        "versions": {name: importlib.metadata.version(name) for name in ["torch", "transformers", "peft", "accelerate", "bitsandbytes"]},
        "devices": [torch.cuda.get_device_name(0)] if device == "cuda" else ["cpu"],
        "peak_allocated_vram_bytes_rank0": torch.cuda.max_memory_allocated() if device == "cuda" else None,
        "saved_adapter_reloaded_for_evaluation": True}
    write_json(output / "run_manifest.json", manifest)
    print(json.dumps({"output_dir": str(output), "adapter_path": str(adapter_path),
                      "best_epoch": best["epoch"], "macro_f1": metrics["classification"]["macro_f1"]}), flush=True)


if __name__ == "__main__":
    main()
