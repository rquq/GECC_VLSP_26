"""Sailor2 QLoRA soft hierarchical classification baseline."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path

from .data import LABELS, _id_key, _source_key, load_records, split_records
from .metrics import score
from .runtime import load_config, resolve_data, stage_output, write_json
from .prediction_export import comparison_rows
from .data import load_classifier_split


def _architecture():
    return {
        "name": "soft_hierarchical_error_type", "version": 2,
        "score_width": 5,
        "score_outputs": ["error_logit", "sentence_logit", "phrase_logit",
                          "function_word_logit", "punctuation_logit"],
        "conditional_labels": [1, 2, 3, 4],
        "joint_log_probabilities": "[logsigmoid(-z0), logsigmoid(z0) + log_softmax(z1..z4)]",
        "subtype_loss_weight": 2.0,
        "objective": "weighted binary + 2 * conditional subtype CE / accumulation-window row count",
        "decision": "argmax joint probabilities; no hard gate",
    }


def _require_hierarchy(saved):
    if saved.get("architecture") != _architecture():
        raise ValueError("Baseline requires lambda-two soft hierarchical architecture metadata; "
                         "reload earlier classifiers with their original experiment code")


def _joint_log_probabilities(logits):
    """Compose P(clean)=1-q and P(type=k)=q*r_k stably in FP32."""
    import torch
    import torch.nn.functional as F

    logits = logits.float()
    error = logits[..., :1]
    return torch.cat((F.logsigmoid(-error),
                      F.logsigmoid(error) + F.log_softmax(logits[..., 1:], dim=-1)), dim=-1)


def _preprocessing(config):
    max_length = config["training"]["max_length"]
    if type(max_length) is not int or max_length < 1:
        raise ValueError("Classifier max_length must be a positive integer")
    return {
        "word_segmentation": "none",
        "implementation": "raw_text",
        "max_length": max_length,
        "padding_side": "right",
        "truncation": True,
        "labels": list(LABELS),
    }


def _texts(records, preprocessing):
    if (preprocessing["word_segmentation"] != "none"
            or preprocessing["implementation"] != "raw_text"
            or preprocessing["padding_side"] != "right"):
        raise ValueError("Unsupported saved classifier preprocessing")
    return [row["source"] for row in records]


def placement_preflight(settings, device="cuda"):
    """Resolve single-device placement and supported precision without loading weights."""
    import torch

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Classification uses a single-process QLoRA fit, not DDP")
    device = torch.device(device)
    quantized = settings["load_in_4bit"]
    if device.type == "cpu":
        if quantized:
            raise ValueError("Quantized classifier requires CUDA; use load_in_4bit=false for a small CPU test")
        return {"device": "cpu", "compute_dtype": "float32", "quantized": False}
    if device.type != "cuda":
        raise ValueError("Classifier device must be CUDA or CPU")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    index = torch.cuda.current_device() if device.index is None else device.index
    requested = settings["compute_dtype"]
    if requested not in ("bfloat16", "float16", "float32"):
        raise ValueError("model.compute_dtype must be bfloat16, float16, or float32")
    with torch.cuda.device(index):
        actual = "float16" if requested == "bfloat16" and not torch.cuda.is_bf16_supported() else requested
    return {"device": f"cuda:{index}", "compute_dtype": actual, "quantized": quantized}


def _configure_padding(tokenizer):
    """Decoder pooling must distinguish real EOS tokens from right-padding tokens."""
    tokenizer.padding_side = "right"
    added = False
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        # Qwen already has a vocabulary-resident end-of-text token distinct from im_end.
        candidate = "<|endoftext|>"
        if candidate in tokenizer.get_vocab() and tokenizer.convert_tokens_to_ids(candidate) != tokenizer.eos_token_id:
            tokenizer.pad_token = candidate
        else:
            added = bool(tokenizer.add_special_tokens({"pad_token": "<|classification_pad|>"}))
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        raise ValueError("Sequence classification needs a pad token distinct from EOS")
    return added


def _load_tokenizer(path, **kwargs):
    """Reload either legacy vocabularies or a complete saved tokenizer backend."""
    from transformers import AutoTokenizer
    directory = Path(path)
    metadata = directory / "tokenizer_config.json"
    saved = json.loads(metadata.read_text()) if metadata.is_file() else {}
    generic_fast = saved.get("tokenizer_class") in {"PreTrainedTokenizer", "PreTrainedTokenizerFast"}
    # Transformers 5 writes a list here; 4.x expects a mapping. These tokens
    # already have IDs in the backend, so register them without changing IDs.
    if isinstance(saved.get("extra_special_tokens"), list):
        kwargs.setdefault("extra_special_tokens", {})
        kwargs.setdefault("additional_special_tokens", list(dict.fromkeys(
            saved.get("additional_special_tokens", []) + saved["extra_special_tokens"])))
    # Newer Transformers can save Qwen's backend without vocab.json/merges.txt,
    # while its metadata still names Qwen2Tokenizer. The slow loader cannot open
    # that layout. Use the saved backend directly, without rebuilding its rules.
    backend_only = ((directory / "tokenizer.json").is_file() and
                    not (directory / "vocab.json").is_file() and
                    not (directory / "merges.txt").is_file())
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=generic_fast or backend_only, **kwargs)
    # Decoder classifiers have no token-type input; generic tokenizer reloads
    # otherwise silently restore the encoder-style default input names.
    tokenizer.model_input_names = ["input_ids", "attention_mask"]
    return tokenizer


def _load_base(settings, tokenizer, placement):
    import torch
    from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig

    dtype = getattr(torch, placement["compute_dtype"])
    kwargs = {
        "revision": settings["revision"], "num_labels": len(LABELS),
        "id2label": {label: str(label) for label in LABELS},
        "label2id": {str(label): label for label in LABELS},
        "problem_type": "single_label_classification", "torch_dtype": dtype,
        "attn_implementation": settings["attn_implementation"],
    }
    if placement["quantized"]:
        kwargs.update(
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type=settings["quant_type"],
                bnb_4bit_use_double_quant=settings["use_double_quant"],
                llm_int8_skip_modules=["score"]),
            device_map={"": torch.device(placement["device"]).index})
    model = AutoModelForSequenceClassification.from_pretrained(settings["name"], **kwargs)
    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    # QLoRA preparation casts some weights to FP32; model.dtype alone would then
    # misidentify the Trainer's mixed-precision forward context on reload.
    model.config.classifier_compute_dtype = placement["compute_dtype"]
    if not placement["quantized"]:
        model.to(placement["device"])
    return model


def _encoded(tokenizer, records, preprocessing):
    tokens = tokenizer(_texts(records, preprocessing), truncation=True,
                       max_length=preprocessing["max_length"], padding=False)
    return [{**{key: values[index] for key, values in tokens.items()}, "labels": row["label"]}
            for index, row in enumerate(records)]


def _predict(model, tokenizer, records, preprocessing, device, batch_size):
    import torch
    from contextlib import nullcontext

    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    model.eval()
    predictions = []
    dtype = getattr(model.config, "classifier_compute_dtype", None)
    compute_dtype = getattr(torch, dtype) if dtype is not None else model.dtype
    use_autocast = torch.device(device).type == "cuda" and compute_dtype in (torch.bfloat16, torch.float16)
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            tokens = tokenizer(_texts(batch, preprocessing), truncation=True,
                               max_length=preprocessing["max_length"], padding=True,
                               return_tensors="pt")
            context = torch.autocast("cuda", dtype=compute_dtype) if use_autocast else nullcontext()
            with context:
                logits = model(**{key: value.to(device) for key, value in tokens.items()}).logits
            probabilities = _joint_log_probabilities(logits).exp()
            if not torch.isfinite(probabilities).all().item():
                raise RuntimeError("Classifier produced nonfinite probabilities")
            for row, values in zip(batch, probabilities.cpu().tolist()):
                predictions.append({"id": row["id"], "label": max(LABELS, key=values.__getitem__),
                                    "probabilities": values})
    return predictions


def _release(device):
    import torch

    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()


def classify(checkpoint, records, device="cuda", batch_size=8):
    """Reload the saved hierarchical LoRA/score head and return five joint scores."""
    from peft import PeftModel, prepare_model_for_kbit_training

    checkpoint = Path(checkpoint)
    preprocessing = json.loads((checkpoint / "preprocessing.json").read_text(encoding="utf-8"))
    saved = json.loads((checkpoint / "classifier_config.json").read_text(encoding="utf-8"))
    _require_hierarchy(saved)
    if preprocessing["labels"] != list(LABELS) or saved["labels"] != list(LABELS):
        raise ValueError("Saved classifier label mapping must be 0..4")
    placement = placement_preflight(saved["model"], device)
    print(json.dumps({"classifier_placement": placement}), flush=True)
    tokenizer = _load_tokenizer(checkpoint)
    _configure_padding(tokenizer)
    if tokenizer.pad_token_id != saved["pad_token_id"] or len(tokenizer) != saved["tokenizer_length"]:
        raise ValueError("Saved classifier tokenizer vocabulary or padding changed")
    base = _load_base(saved["model"], tokenizer, placement)
    if placement["quantized"]:
        # Match the FP32 norms/embeddings used by QLoRA preparation during fit.
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
    model = PeftModel.from_pretrained(base, checkpoint, is_trainable=False)
    if model.config.num_labels != len(LABELS):
        raise ValueError("Saved classifier must predict five labels")
    try:
        return _predict(model, tokenizer, records, preprocessing, placement["device"], batch_size)
    finally:
        del model, base, tokenizer
        _release(placement["device"])


def _classification_metrics(records, predictions):
    # Classification alone is reported; correction targets are exact identities.
    gold = [{**row, "target": row["source"]} for row in records]
    sources = {_id_key(row["id"]): row["source"] for row in records}
    identity = [{**row, "target": sources[_id_key(row["id"])]} for row in predictions]
    return score(gold, identity)["classification"]


def _finite_parameters(model):
    import torch

    if not all(torch.isfinite(parameter).all().item() for parameter in model.parameters()
               if parameter.requires_grad and parameter.is_floating_point()):
        raise RuntimeError("Classifier has nonfinite trainable parameters")


def _class_weights(train, power):
    """Use fit-local mixed frequencies; normalize expected sample weight to one."""
    if type(power) not in (int, float) or not math.isfinite(power) or not 0 <= power <= 1:
        raise ValueError("class_weight_power must be a finite number in [0, 1]")
    counts = Counter(row["label"] for row in train)
    if any(counts[label] == 0 for label in LABELS):
        raise ValueError("Every classifier fit needs all five training classes")
    raw = [counts[label] ** (-power) for label in LABELS]
    normalizer = sum(counts[label] * value for label, value in zip(LABELS, raw)) / len(train)
    return {"power": power, "counts": [counts[label] for label in LABELS],
            "weights": [value / normalizer for value in raw],
            "normalization": "weighted sum divided by accumulation-batch row count"}


def _weighted_hierarchical_loss(logits, labels, weights, num_items_in_batch=None):
    """Weighted binary + twice subtype CE; clean rows have no subtype gradient."""
    import torch.nn.functional as F

    logits = logits.float()
    binary = F.binary_cross_entropy_with_logits(
        logits[..., 0], (labels > 0).float(), reduction="none")
    # Ignore clean rows without selecting a variable-size CUDA tensor. Their
    # conditional target is -1; cross_entropy's ignore_index produces zero loss.
    subtype = F.cross_entropy(logits[..., 1:], labels - 1, ignore_index=-1, reduction="none")
    loss = (weights[labels] * (binary + 2.0 * subtype)).sum()
    denominator = labels.numel() if num_items_in_batch is None else num_items_in_batch
    return loss / denominator


def _hierarchical_trainer_class():
    """Keep optional Transformers imports lazy for dataset-only consumers."""
    import torch
    from transformers import Trainer

    class HierarchicalTrainer(Trainer):
        def __init__(self, *args, loss_weights, **kwargs):
            super().__init__(*args, **kwargs)
            self.loss_weights = loss_weights
            # We consume Trainer's full/partial accumulation-window row count.
            # This also disables its second division by accumulation steps;
            # kwargs are consumed here, never forwarded to the Qwen model.
            self.model_accepts_loss_kwargs = True

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs["labels"]
            features = {key: value for key, value in inputs.items() if key != "labels"}
            output = model(**features)
            loss = _weighted_hierarchical_loss(output.logits, labels, self.loss_weights,
                                               num_items_in_batch)
            if not torch.isfinite(loss).all().item():
                raise RuntimeError("Nonfinite classifier training loss")
            return (loss, output) if return_outputs else loss

    return HierarchicalTrainer


def _training_algorithm(config, model):
    if "algorithm" in config["training"]:
        raise ValueError("training.algorithm is obsolete; classifier training uses single-forward hierarchical loss")
    return {
        "name": "hierarchical", "supervised_objective": "weighted BCE + 2*CE",
        "normalization": "weighted sum divided by full/partial accumulation-window rows",
        "training_forwards": 1, "inference_forwards": 1,
        "lora_dropout": config["lora"]["lora_dropout"],
        "base_attention_dropout": getattr(model.config, "attention_dropout", None),
        "seed": "fit seed recorded in run manifest",
    }


def _fit(config, train, validation, checkpoint, artifact_dir, device, seed, preprocessing):
    """Fresh mixed-data LoRA/head fit with outer-holdout epoch selection."""
    import torch
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import (DataCollatorWithPadding, TrainerCallback,
                              TrainingArguments, set_seed)

    if not train:
        raise ValueError("Every classifier fit needs training rows")
    if "algorithm" in config["training"]:
        raise ValueError("training.algorithm is obsolete; classifier training uses single-forward hierarchical loss")
    weighting = _class_weights(train, config["training"].get("class_weight_power", 0.0))
    set_seed(seed)
    placement = placement_preflight(config["model"], device)
    device = placement["device"]
    print(json.dumps({"classifier_placement": placement}), flush=True)
    settings = {**config["model"], "compute_dtype": placement["compute_dtype"]}
    tokenizer = _load_tokenizer(settings["name"], revision=settings["revision"])
    added_pad = _configure_padding(tokenizer)
    model = _load_base(settings, tokenizer, placement)
    checkpoint_kwargs = {"use_reentrant": False}
    gradient_checkpointing = config["training"]["args"].get("gradient_checkpointing", True)
    if placement["quantized"]:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=gradient_checkpointing,
            gradient_checkpointing_kwargs=checkpoint_kwargs)
    lora = {**config["lora"], "task_type": TaskType.SEQ_CLS, "modules_to_save": ["score"],
            "revision": settings["revision"]}
    model = get_peft_model(model, LoraConfig(**lora))
    # Trainer otherwise wraps any visible multi-GPU host in DataParallel.
    model.is_parallelizable = True
    model.model_parallel = True
    classifier_config = {
        "model": settings, "requested_compute_dtype": config["model"]["compute_dtype"],
        "lora": {**config["lora"], "task_type": "SEQ_CLS", "modules_to_save": ["score"],
                 "revision": settings["revision"]},
        "num_labels": len(LABELS), "labels": list(LABELS),
        "architecture": _architecture(),
        "inference_precision": "CUDA autocast at model.compute_dtype; FP32 composition; no CPU autocast",
        "pad_token_id": tokenizer.pad_token_id, "tokenizer_length": len(tokenizer),
        "added_pad_token": added_pad,
        "class_weighting": weighting,
        "training_algorithm": _training_algorithm(config, model),
    }
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    encoded = _encoded(tokenizer, train, preprocessing)
    training = config["training"]
    args = dict(training["args"])
    microbatch = args["per_device_train_batch_size"]
    effective = training["effective_batch_size"]
    if type(microbatch) is not int or microbatch < 1 or type(effective) is not int or effective < microbatch or effective % microbatch:
        raise ValueError("effective_batch_size must be a positive multiple of the microbatch")
    epochs = float(args["num_train_epochs"])
    if args.get("max_steps", -1) < 1 and epochs >= 1 and epochs.is_integer():
        # Trainer 4.x floors the estimated updates per epoch, although its loop
        # updates the final partial accumulation window. Resolve an explicit
        # budget so integer epoch requests consume every batch in every epoch.
        microbatches = math.ceil(len(encoded) / microbatch)
        args["max_steps"] = math.ceil(microbatches / (effective // microbatch)) * int(epochs)
    args.update(bf16=device.startswith("cuda") and placement["compute_dtype"] == "bfloat16",
                fp16=device.startswith("cuda") and placement["compute_dtype"] == "float16",
                gradient_checkpointing=gradient_checkpointing,
                gradient_checkpointing_kwargs=checkpoint_kwargs)
    training_args = TrainingArguments(
        **args, output_dir=str(artifact_dir / "trainer"), use_cpu=device == "cpu",
        gradient_accumulation_steps=effective // microbatch,
        eval_strategy="no", save_strategy="no", report_to="none",
        logging_nan_inf_filter=False, seed=seed, data_seed=seed, label_names=["labels"])
    # Prevent Trainer's implicit DataParallel when more than one GPU is visible.
    training_args._n_gpu = 0 if device == "cpu" else 1
    if str(training_args.device) != device:
        raise ValueError("Pin the classifier GPU with CUDA_VISIBLE_DEVICES before starting the process")
    history = []
    best = {"epoch": None, "macro_f1": None}
    batch_size = args.get("per_device_eval_batch_size", 8)

    def save_model():
        checkpoint.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint, save_embedding_layers=added_pad)
        tokenizer.save_pretrained(checkpoint)
        write_json(checkpoint / "preprocessing.json", preprocessing)
        write_json(checkpoint / "classifier_config.json", classifier_config)

    loss_weights = torch.tensor(weighting["weights"], dtype=torch.float32, device=device)

    HierarchicalTrainer = _hierarchical_trainer_class()

    class EpochEvaluation(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):
            _finite_parameters(model)
            entry = {"epoch": state.epoch, "optimizer_steps": state.global_step}
            if validation is not None:
                predictions = _predict(model, tokenizer, validation, preprocessing, device, batch_size)
                metrics = _classification_metrics(validation, predictions)
                entry["classification"] = metrics
                if best["macro_f1"] is None or metrics["macro_f1"] > best["macro_f1"]:
                    best.update(epoch=state.epoch, macro_f1=metrics["macro_f1"])
                    save_model()
            history.append(entry)
            write_json(artifact_dir / "epoch_history.json", history)
            print(json.dumps(entry, ensure_ascii=False), flush=True)
            model.train()
            return control

    trainer = HierarchicalTrainer(model=model, args=training_args, train_dataset=encoded,
                          data_collator=DataCollatorWithPadding(tokenizer), callbacks=[EpochEvaluation()],
                          loss_weights=loss_weights)
    if training_args.fp16:
        # Qwen3 NF4 backward overflowed at scale 4096 despite a finite forward loss.
        # Start unscaled; retain dynamic scaling and the nonfinite-statistic guard.
        trainer.accelerator.scaler = torch.amp.GradScaler(
            "cuda", init_scale=training.get("fp16_initial_scale", 1.0))
    result = trainer.train()
    _finite_parameters(model)
    if result.global_step < 1 or not math.isfinite(result.training_loss):
        raise RuntimeError("Classifier training did not produce finite optimizer updates")
    for entry in trainer.state.log_history:
        for key in ("loss", "grad_norm", "learning_rate", "train_loss"):
            if key in entry and not math.isfinite(entry[key]):
                raise RuntimeError(f"Nonfinite classifier training statistic: {key}")
    if validation is None:
        save_model()
    elif best["epoch"] is None:
        raise RuntimeError("Main classifier did not evaluate or save an epoch")
    write_json(artifact_dir / "trainer_history.json", trainer.state.log_history)
    stats = {
        "training_loss": result.training_loss, "optimizer_steps": result.global_step,
        "trainable_parameters": trainable_parameters, "total_parameters": total_parameters,
        "classifier_config": classifier_config, "placement": placement,
        "parameters_finite": True, "loss_finite": True,
        "gradient_accumulation_steps": effective // microbatch,
        "effective_batch_size": effective, "training_args": training_args.to_dict(),
        "class_weighting": weighting,
        "training_algorithm": classifier_config["training_algorithm"],
        "fp16_initial_scale": training.get("fp16_initial_scale", 1.0) if training_args.fp16 else None,
        "best_epoch": best["epoch"], "selection_macro_f1": best["macro_f1"],
        "selection_rule": ("highest outer-validation macro F1; ties retain earlier epoch"
                           if validation is not None else "fixed epochs; no outer-validation evaluation or selection"),
        "train_max_tokens": max(len(row["input_ids"]) for row in encoded),
        "train_ids": [row["id"] for row in train],
        "validation_ids": [row["id"] for row in validation] if validation is not None else [],
    }
    trainer.model = None
    trainer.model_wrapped = None
    trainer.optimizer = None
    trainer.accelerator.free_memory()
    del trainer, model, tokenizer
    _release(device)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", choices=["cuda", "cpu"])
    args = parser.parse_args()
    config = load_config(args.config)
    device = args.device or config["run"]["device"]
    config["run"]["device"] = device
    seed = config["run"]["seed"]
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Classification uses a single-process QLoRA fit, not DDP")
    import torch

    if device not in ("cuda", "cpu"):
        raise ValueError("run.device must be cuda or cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    path = resolve_data(config, args.data)
    fraction = config["data"]["validation_fraction"]
    original_train, validation = split_records(load_records(path), fraction, seed)
    mixed_config = copy.deepcopy(config)
    mixed_config["data"].update(path=config["data"].get("augmentation_path"),
                                filename=config["data"]["augmentation_filename"])
    mixed_path = resolve_data(mixed_config, None)
    train, classifier_info = load_classifier_split(path, mixed_path, fraction, seed)
    preprocessing = _preprocessing(config)
    output = stage_output(config, args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "classifier"
    config["data"]["path"] = str(path)
    config["data"]["augmentation_path"] = str(mixed_path)
    write_json(output / "config.json", config)
    write_json(output / "train.json", train)
    write_json(output / "validation.json", validation)
    write_json(output / "preprocessing.json", preprocessing)
    stats = _fit(config, train, validation, checkpoint, output, device, seed, preprocessing)
    batch_size = config["training"]["args"].get("per_device_eval_batch_size", 8)
    predictions = classify(checkpoint, validation, device, batch_size)
    metrics = {"count": len(validation), "classification": _classification_metrics(validation, predictions),
               "training": stats}
    write_json(output / "predictions.json", comparison_rows(validation, predictions))
    write_json(output / "metrics.json", metrics)
    manifest = {
        "task": "classification", "config": config, "seed": seed, "validation_fraction": fraction,
        "device": device, "world_size": 1, "output_dir": str(output), "classifier_path": str(checkpoint),
        "data_path": str(path), "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "classifier_data_path": str(mixed_path),
        "classifier_data_sha256": hashlib.sha256(mixed_path.read_bytes()).hexdigest(),
        "original_train_count": len(original_train),
        "retained_original_train_count": len(classifier_info["provenance"]["retained_original_train_ids"]),
        "quarantined_original_train_count": len(classifier_info["provenance"]["quarantined_ids"]),
        "synthetic_train_count": len(classifier_info["provenance"]["selected_synthetic_ids"]),
        "model": config["model"], "resolved_model_revision": config["model"]["revision"],
        "preprocessing": preprocessing, "saved_classifier_reloaded_for_evaluation": True,
        "train_count": len(train), "validation_count": len(validation),
        **stats,
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "accelerate", "peft", "bitsandbytes")},
    }
    write_json(output / "run_manifest.json", manifest)
    print(json.dumps({"output_dir": str(output), "classifier_path": str(checkpoint),
                      "macro_f1": metrics["classification"]["macro_f1"]}), flush=True)


if __name__ == "__main__":
    main()
