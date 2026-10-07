"""Response-only LoRA training; launch with torchrun for one replica per GPU."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = Path(args.output_dir)
    # Refuse stale adapters/reports masquerading as the current run.
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
    from peft import LoraConfig, TaskType, get_peft_model
    from .data import load_records, split_records
    from .metrics import score
    from .modeling import ResponseCollator, encode_record, generate_predictions

    use_cuda = args.device == "cuda"
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu explicitly for a CPU diagnostic")
    if config["bf16"]:
        raise ValueError("This T4 baseline uses fp16, not bf16")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if use_cuda:
        torch.cuda.set_device(local_rank)
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
    # Check validation gold too; no silent truncation in this baseline.
    validation_lengths = [len(encode_record(tokenizer, r, config["max_length"])["input_ids"]) for r in validation]
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"], revision=revision,
        # FP32 stored weights with Trainer's FP16 autocast/scaler; T4-compatible.
        torch_dtype=torch.float32,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=config["lora_rank"],
        lora_alpha=2 * config["lora_rank"], lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none", revision=revision,
    ))
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    training_args = TrainingArguments(
        output_dir=str(output), use_cpu=not use_cuda,
        num_train_epochs=config["epochs"], max_steps=config["max_steps"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"], warmup_ratio=0.1, weight_decay=0.01,
        fp16=use_cuda and config["fp16"], bf16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Other ranks wait while rank 0 reloads the adapter and evaluates.
        ddp_timeout=3600, ddp_find_unused_parameters=False, optim="adamw_torch",
        logging_steps=1, save_strategy="no", eval_strategy="no", report_to="none",
        label_names=["labels"],
        seed=config["seed"], data_seed=config["seed"], dataloader_num_workers=0,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=encoded,
                      data_collator=ResponseCollator(tokenizer.pad_token_id))
    if use_cuda and config["fp16"]:
        # Qwen's early gradients overflow AMP's default 65536 scale on this data.
        trainer.accelerator.scaler = torch.amp.GradScaler("cuda", init_scale=4096.0)
    result = trainer.train()
    if not math.isfinite(result.training_loss):
        raise RuntimeError(f"Nonfinite training loss: {result.training_loss}")
    gradient_norms = [entry["grad_norm"] for entry in trainer.state.log_history if "grad_norm" in entry]
    if not all(math.isfinite(norm) for norm in gradient_norms):
        raise RuntimeError("Nonfinite gradients: training skipped updates; inspect FP16 loss scaling")
    if trainer.is_world_process_zero():
        output.mkdir(parents=True, exist_ok=True)
        adapter = output / "adapter"
        unwrapped = trainer.accelerator.unwrap_model(trainer.model)
        trainable = [p for p in unwrapped.parameters() if p.requires_grad]
        if not all(torch.isfinite(p).all().item() for p in trainable):
            raise RuntimeError("Nonfinite adapter parameters")
        unwrapped.save_pretrained(adapter)
        tokenizer.save_pretrained(adapter)
        write_json(output / "train.json", train)
        write_json(output / "validation.json", validation)
        write_json(output / "trainer_history.json", trainer.state.log_history)
        majority = max(range(5), key=lambda label: sum(r["label"] == label for r in train))
        identity = [{"id": r["id"], "label": majority, "target": r["source"]} for r in validation]
        write_json(output / "identity_metrics.json", score(validation, identity))
        # Reload the saved artifact, proving it is sufficient for inference.
        from peft import PeftModel
        del unwrapped, model
        trainer.model = None
        trainer.model_wrapped = None
        trainer.optimizer = None
        trainer.accelerator.free_memory()
        import gc
        gc.collect()
        if use_cuda:
            torch.cuda.empty_cache()
        base = AutoModelForCausalLM.from_pretrained(
            config["model_name"], revision=revision,
            torch_dtype=torch.float16 if use_cuda else torch.float32,
            attn_implementation="sdpa",
        ).to(torch.device("cuda", local_rank) if use_cuda else "cpu")
        reloaded = PeftModel.from_pretrained(base, adapter)
        predictions = generate_predictions(reloaded, tokenizer, validation, config["max_new_tokens"])
        write_json(output / "predictions.json", predictions)
        write_json(output / "metrics.json", score(validation, predictions))
        manifest = {
            "config": config, "resolved_model_revision": revision,
            "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
            "train_ids": [r["id"] for r in train], "validation_ids": [r["id"] for r in validation],
            "train_max_tokens": max(len(r["input_ids"]) for r in encoded),
            "validation_max_tokens": max(validation_lengths),
            "optimizer_steps": result.global_step, "training_loss": result.training_loss,
            "world_size": int(os.environ.get("WORLD_SIZE", 1)),
            "effective_batch_size": config["batch_size"] * config["gradient_accumulation_steps"] * int(os.environ.get("WORLD_SIZE", 1)),
            "versions": {name: importlib.metadata.version(name) for name in ["torch", "transformers", "peft", "accelerate"]},
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if use_cuda else ["cpu"],
            "peak_allocated_vram_bytes_rank0": torch.cuda.max_memory_allocated(local_rank) if use_cuda else None,
            "saved_adapter_reloaded_for_evaluation": True,
        }
        write_json(output / "run_manifest.json", manifest)
        print(json.dumps({"output_dir": str(output), "training_loss": result.training_loss,
                          "optimizer_steps": result.global_step}, ensure_ascii=False), flush=True)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
