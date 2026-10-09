"""Full-finetuned raw-text encoders for the fixed-split heterogeneous ensemble."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from .classification import _class_weights, _classification_metrics, _finite_parameters
from .data import LABELS, _id_key, _source_key, _validate_records
from .prediction_export import comparison_rows
from .runtime import write_json


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def _load_tokenizer(settings):
    # Both published backends include exact normalization, segmentation and
    # special-token processing. BamiBERT's XLM-R class metadata describes a
    # Unigram tokenizer but its actual tokenizer.json is byte-level BPE.
    from huggingface_hub import hf_hub_download
    from transformers import PreTrainedTokenizerFast

    path = hf_hub_download(settings["name"], "tokenizer.json", revision=settings["revision"])
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=path, bos_token="<s>", eos_token="</s>", unk_token="<unk>",
        sep_token="</s>", pad_token="<pad>", cls_token="<s>", mask_token="<mask>",
        model_input_names=["input_ids", "attention_mask"])
    tokenizer.padding_side = "right"
    return tokenizer


def weighted_loss(logits, labels, weights, denominator=None):
    """Keep sample weights comparable across full and partial accumulation windows."""
    import torch.nn.functional as F

    losses = F.cross_entropy(logits.float(), labels, reduction="none")
    return (losses * weights[labels]).sum() / (labels.numel() if denominator is None else denominator)


def _predict(model, tokenizer, records, settings, device, batch_size):
    import torch

    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    model.eval()
    predictions = []
    dtype = getattr(torch, settings["compute_dtype"])
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            rows = records[start:start + batch_size]
            inputs = tokenizer([r["source"] for r in rows], padding=True, truncation=True,
                               max_length=settings["max_length"], return_tensors="pt").to(device)
            context = torch.autocast("cuda", dtype=dtype) if str(device).startswith("cuda") and dtype != torch.float32 else nullcontext()
            with context:
                logits = model(**inputs).logits
            probabilities = logits.float().softmax(-1)
            if not torch.isfinite(probabilities).all():
                raise RuntimeError("Nonfinite encoder probabilities")
            for row, values in zip(rows, probabilities.cpu().tolist()):
                predictions.append({"id": row["id"], "label": max(LABELS, key=values.__getitem__),
                                    "probabilities": values})
    return predictions


def classify(checkpoint, records, device="cuda", batch_size=8):
    """Load one standalone encoder checkpoint; inference never consumes gold targets."""
    import torch
    from transformers import AutoModelForSequenceClassification, PreTrainedTokenizerFast

    checkpoint = Path(checkpoint)
    settings = _read(checkpoint / "encoder_config.json")
    if settings["labels"] != list(LABELS) or settings["preprocessing"] != "raw_text":
        raise ValueError("Unsupported encoder labels or preprocessing")
    if settings["compute_dtype"] == "float16":
        raise ValueError("Encoder serving supports BF16 or FP32")
    _validate_records(records, require_labels=False)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)
    try:
        return _predict(model, tokenizer, records, settings, device, batch_size)
    finally:
        del model, tokenizer
        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()


def train(config, control, output, device="cuda"):
    import torch
    from transformers import (AutoModelForSequenceClassification,
                              DataCollatorWithPadding, Trainer, TrainerCallback,
                              TrainingArguments, set_seed)

    output, control = Path(output).resolve(), Path(control).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Refusing to overwrite existing run: {output}")
    train_rows, validation = _read(control / "train.json"), _read(control / "validation.json")
    for rows in (train_rows, validation):
        _validate_records(rows)
        if len({_id_key(row["id"]) for row in rows}) != len(rows):
            raise ValueError("Duplicate split IDs")
    if {_id_key(r["id"]) for r in train_rows} & {_id_key(r["id"]) for r in validation}:
        raise ValueError("Training/validation ID overlap")
    if {_source_key(r["source"]) for r in train_rows} & {_source_key(r["source"]) for r in validation}:
        raise ValueError("Training/validation source overlap")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "classifier"
    set_seed(config["seed"])
    settings = {**config["model"], "labels": list(LABELS), "preprocessing": "raw_text",
                "max_length": config["max_length"],
                "compute_dtype": "bfloat16" if str(device).startswith("cuda") and torch.cuda.is_bf16_supported() else "float32"}
    tokenizer = _load_tokenizer(settings)
    model = AutoModelForSequenceClassification.from_pretrained(
        settings["name"], revision=settings["revision"], num_labels=len(LABELS),
        id2label={k: str(k) for k in LABELS}, label2id={str(k): k for k in LABELS},
        problem_type="single_label_classification").to(device)
    if settings["max_length"] > model.config.max_position_embeddings - model.config.pad_token_id - 1:
        raise ValueError("Requested encoder length exceeds position capacity")
    weighting = _class_weights(train_rows, config["class_weight_power"])
    weights = torch.tensor(weighting["weights"], dtype=torch.float32, device=device)
    tokens = tokenizer([row["source"] for row in train_rows], truncation=True,
                       max_length=settings["max_length"], padding=False)
    encoded = [{**{k: values[i] for k, values in tokens.items()}, "labels": row["label"]}
               for i, row in enumerate(train_rows)]
    microbatch = config["microbatch"]
    if config["effective_batch_size"] % microbatch:
        raise ValueError("Effective batch size must be divisible by microbatch")
    args = TrainingArguments(
        output_dir=str(output / "trainer"), use_cpu=str(device) == "cpu",
        num_train_epochs=config["epochs"], learning_rate=config["learning_rate"],
        per_device_train_batch_size=microbatch,
        gradient_accumulation_steps=config["effective_batch_size"] // microbatch,
        weight_decay=0.01, warmup_steps=0, lr_scheduler_type="linear", optim="adamw_torch",
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0, bf16=settings["compute_dtype"] == "bfloat16",
        save_strategy="no", eval_strategy="no", report_to="none", logging_steps=10,
        seed=config["seed"], data_seed=config["seed"], label_names=["labels"])
    args._n_gpu = 0 if str(device) == "cpu" else 1
    best = {"epoch": None, "macro_f1": None}
    history = []

    class WeightedTrainer(Trainer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.model_accepts_loss_kwargs = True

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs["labels"]
            outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
            loss = weighted_loss(outputs.logits, labels, weights, num_items_in_batch)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite encoder training loss")
            return (loss, outputs) if return_outputs else loss

    class EpochEvaluation(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):
            _finite_parameters(model)
            predictions = _predict(model, tokenizer, validation, settings, device, config["eval_batch_size"])
            metrics = _classification_metrics(validation, predictions)
            entry = {"epoch": state.epoch, "optimizer_steps": state.global_step, "classification": metrics}
            history.append(entry)
            print(json.dumps(entry), flush=True)
            if best["macro_f1"] is None or metrics["macro_f1"] > best["macro_f1"]:
                best.update(epoch=state.epoch, macro_f1=metrics["macro_f1"])
                checkpoint.mkdir(exist_ok=True)
                model.save_pretrained(checkpoint)
                tokenizer.save_pretrained(checkpoint)
                write_json(checkpoint / "encoder_config.json", settings)
            model.train()
            write_json(output / "epoch_history.json", history)
            return control

    for name, value in (("config.json", config), ("train.json", train_rows), ("validation.json", validation)):
        write_json(output / name, value)
    trainer = WeightedTrainer(model=model, args=args, train_dataset=encoded,
                              data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
                              callbacks=[EpochEvaluation()])
    started = time.monotonic()
    result = trainer.train()
    _finite_parameters(model)
    parameters = sum(p.numel() for p in model.parameters())
    write_json(output / "trainer_history.json", trainer.state.log_history)
    manifest = {"task": "encoder_classification", "config": config, "model": settings,
                "resolved_model_revision": settings["revision"], "seed": config["seed"],
                "control_run": str(control), "train_count": len(train_rows), "validation_count": len(validation),
                "classifier_path": str(checkpoint),
                "train_sha256": hashlib.sha256((control / "train.json").read_bytes()).hexdigest(),
                "validation_sha256": hashlib.sha256((control / "validation.json").read_bytes()).hexdigest(),
                "class_weighting": weighting, "training_args": args.to_dict(),
                "optimizer_steps": result.global_step, "best_epoch": best["epoch"],
                "training_loss": result.training_loss, "parameters": parameters,
                "objective": "flat five-class weighted CE / accumulation-window row count",
                "full_finetuning": True, "training_seconds": time.monotonic() - started,
                "versions": {n: importlib.metadata.version(n) for n in ("torch", "transformers", "tokenizers")},
                "limitations": "Same repeatedly selected holdout; architecture and objective differ from Qwen; not independent-test evidence"}
    del trainer, model, tokenizer, weights
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    predictions = classify(checkpoint, validation, device, config["eval_batch_size"])
    metrics = _classification_metrics(validation, predictions)
    if metrics["macro_f1"] != best["macro_f1"]:
        raise RuntimeError("Saved encoder reload changed selected holdout decisions")
    manifest["saved_classifier_reloaded_for_evaluation"] = True
    write_json(output / "predictions.json", comparison_rows(validation, predictions))
    write_json(output / "metrics.json", {"count": len(validation), "classification": metrics})
    write_json(output / "run_manifest.json", manifest)
    print(json.dumps({"output": str(output), "best_epoch": best["epoch"], "classification": metrics}), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--control-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()
    train(_read(args.config), args.control_run, args.output_dir, args.device)


if __name__ == "__main__":
    main()
