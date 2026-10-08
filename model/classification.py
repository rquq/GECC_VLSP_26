"""Qwen QLoRA five-class classification and group-safe, train-only correction hints."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random

from .data import LABELS, _id_key, _source_key, load_records, load_training_split, split_records
from .metrics import score
from .runtime import write_json
from .prediction_export import comparison_rows
from .data import load_classifier_split


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

    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            tokens = tokenizer(_texts(batch, preprocessing), truncation=True,
                               max_length=preprocessing["max_length"], padding=True,
                               return_tensors="pt")
            logits = model(**{key: value.to(device) for key, value in tokens.items()}).logits
            probabilities = logits.float().softmax(dim=-1)
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
    """Reload the exact base plus saved LoRA/score head and return five softmax scores."""
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoTokenizer

    checkpoint = Path(checkpoint)
    preprocessing = json.loads((checkpoint / "preprocessing.json").read_text(encoding="utf-8"))
    saved = json.loads((checkpoint / "classifier_config.json").read_text(encoding="utf-8"))
    if preprocessing["labels"] != list(LABELS) or saved["labels"] != list(LABELS):
        raise ValueError("Saved classifier label mapping must be 0..4")
    placement = placement_preflight(saved["model"], device)
    print(json.dumps({"classifier_placement": placement}), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)
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


def _weighted_cross_entropy(logits, labels, weights, num_items_in_batch=None):
    """Do not divide by sum(weights): that cancels weighting for microbatch one."""
    import torch.nn.functional as F

    loss = F.cross_entropy(logits.float(), labels, weight=weights, reduction="sum")
    denominator = labels.numel() if num_items_in_batch is None else num_items_in_batch
    return loss / denominator


def _optimizer_groups(model, decay_names, lora_lr, head_lr, weight_decay):
    """Partition only trainable adapters/head; retain Trainer's decay policy."""
    for name, value in (("LoRA learning_rate", lora_lr), ("head_learning_rate", head_lr)):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    buckets = {(role, decay): [] for role in ("lora", "head") for decay in (True, False)}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "score" in name.split("."):
            role = "head"
        elif "lora_" in name:
            role = "lora"
        else:
            raise ValueError(f"Unexpected trainable classifier parameter: {name}")
        buckets[role, name in decay_names].append(parameter)
    if any(not any(buckets[role, decay] for decay in (True, False)) for role in ("lora", "head")):
        raise ValueError("Classifier optimizer requires trainable LoRA and score head parameters")
    return [{"params": parameters, "lr": head_lr if role == "head" else lora_lr,
             "weight_decay": weight_decay if decay else 0.0, "role": role}
            for (role, decay), parameters in buckets.items() if parameters]


def _fit(config, train, validation, checkpoint, artifact_dir, device, seed, preprocessing,
         fixed_epochs=None):
    """Fresh mixed-data LoRA/head fit; only main fit selects on outer holdout."""
    import torch
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoTokenizer, DataCollatorWithPadding, Trainer,
                              TrainerCallback, TrainingArguments, set_seed)

    if not train:
        raise ValueError("Every classifier fit needs training rows")
    weighting = _class_weights(train, config["training"].get("class_weight_power", 0.0))
    set_seed(seed)
    placement = placement_preflight(config["model"], device)
    device = placement["device"]
    print(json.dumps({"classifier_placement": placement}), flush=True)
    settings = {**config["model"], "compute_dtype": placement["compute_dtype"]}
    tokenizer = AutoTokenizer.from_pretrained(settings["name"], revision=settings["revision"], use_fast=False)
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
        "pad_token_id": tokenizer.pad_token_id, "tokenizer_length": len(tokenizer),
        "added_pad_token": added_pad,
        "class_weighting": weighting,
        "learning_rates": {"lora": config["training"]["args"]["learning_rate"],
                           "head": config["training"]["head_learning_rate"]},
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
    if fixed_epochs is not None:
        args.update(num_train_epochs=fixed_epochs, max_steps=-1)
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

    def classification_loss(outputs, labels, num_items_in_batch=None):
        return _weighted_cross_entropy(outputs.logits, labels, loss_weights, num_items_in_batch)

    class FiniteTrainer(Trainer):
        def create_optimizer(self, model=None):
            if self.optimizer is None:
                if self.args.optim != "adamw_torch":
                    raise ValueError("Separate head/LoRA rates require optim=adamw_torch")
                opt_model = self.model if model is None else model
                groups = _optimizer_groups(
                    opt_model, self.get_decay_parameter_names(opt_model), self.args.learning_rate,
                    training["head_learning_rate"], self.args.weight_decay)
                self.optimizer = torch.optim.AdamW(
                    groups, lr=self.args.learning_rate,
                    betas=(self.args.adam_beta1, self.args.adam_beta2), eps=self.args.adam_epsilon)
            return self.optimizer

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            result = super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
            loss = result[0] if return_outputs else result
            if not torch.isfinite(loss).all().item():
                raise RuntimeError("Nonfinite classifier training loss")
            return result

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

    trainer = FiniteTrainer(model=model, args=training_args, train_dataset=encoded,
                            data_collator=DataCollatorWithPadding(tokenizer), callbacks=[EpochEvaluation()],
                            compute_loss_func=classification_loss)
    # The custom loss receives the full accumulation-window row count, including
    # partial final windows; Trainer must not divide it by accumulation again.
    trainer.model_accepts_loss_kwargs = False
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
        "learning_rates": classifier_config["learning_rates"],
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


def _connected_folds(original_train, correction_train, provenance, folds, seed, classifier_train):
    """Keep mixed classifier rows, correction rows and synthetic parents together."""
    if type(folds) is not int or folds < 2:
        raise ValueError("OOF folds must be an integer of at least two")
    originals = {_id_key(row["id"]): row for row in original_train}
    corrections = {_id_key(row["id"]): row for row in correction_train}
    training = {_id_key(row["id"]): row for row in classifier_train}
    parent = {key: key for key in originals.keys() | corrections.keys() | training.keys()}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(first, second):
        parent[find(second)] = find(first)

    sources = {}
    for row in original_train + correction_train + classifier_train:
        key = _id_key(row["id"])
        source = _source_key(row["source"])
        if source in sources:
            union(key, sources[source])
        else:
            sources[source] = key
    for row in provenance.get("train_rows", []):
        key = _id_key(row["id"])
        for value in row.get("parent_ids", []):
            original = _id_key(value)
            if original not in originals:
                raise ValueError("Synthetic OOF parent is not an original training ID")
            union(key, original)
    components = defaultdict(list)
    for key in parent:
        components[find(key)].append(key)
    groups = []
    for keys in components.values():
        original_keys = [key for key in keys if key in originals]
        correction_keys = [key for key in keys if key in corrections]
        counts = Counter(originals[key]["label"] for key in original_keys)
        groups.append({"keys": keys, "original_keys": original_keys,
                       "correction_keys": correction_keys, "counts": counts})
    if sum(bool(group["correction_keys"]) for group in groups) < folds:
        raise ValueError("Too few independent correction source/parent groups for the requested OOF folds")
    # Stabilize before seeded tie breaking; typed IDs remain distinct throughout.
    groups.sort(key=lambda group: min(json.dumps([key[0].__name__, key[1]], ensure_ascii=False) for key in group["keys"]))
    rng = random.Random(seed)
    rng.shuffle(groups)
    totals = Counter(row["label"] for row in original_train)
    groups.sort(key=lambda group: (max((count / totals[label] for label, count in group["counts"].items()), default=0),
                                   len(group["original_keys"]), len(group["correction_keys"])), reverse=True)
    fold_counts = [Counter() for _ in range(folds)]
    fold_original_sizes = [0] * folds
    fold_correction_sizes = [0] * folds
    assignments = {}
    tie_order = list(range(folds))
    rng.shuffle(tie_order)
    for group in groups:
        candidates = tie_order
        if group["correction_keys"] and any(size == 0 for size in fold_correction_sizes):
            candidates = [fold for fold in tie_order if fold_correction_sizes[fold] == 0]

        def balance_cost(fold):
            cost = sum(((fold_counts[fold][label] + group["counts"][label]) ** 2
                        - fold_counts[fold][label] ** 2) / max(1, totals[label]) ** 2 for label in LABELS)
            cost += ((fold_original_sizes[fold] + len(group["original_keys"])) ** 2
                     - fold_original_sizes[fold] ** 2) / len(original_train) ** 2
            cost += ((fold_correction_sizes[fold] + len(group["correction_keys"])) ** 2
                     - fold_correction_sizes[fold] ** 2) / len(correction_train) ** 2
            return cost

        fold = min(candidates, key=balance_cost)
        fold_counts[fold].update(group["counts"])
        fold_original_sizes[fold] += len(group["original_keys"])
        fold_correction_sizes[fold] += len(group["correction_keys"])
        assignments.update((key, fold) for key in group["keys"])
    plans = []
    for fold in range(folds):
        train = [row for row in classifier_train if assignments[_id_key(row["id"])] != fold]
        # Group exclusion can change ratios; cap again within each OOF fit.
        original_counts = Counter(row["label"] for row in train if _id_key(row["id"]) in originals)
        synthetic_counts = Counter()
        capped = []
        for row in train:
            if _id_key(row["id"]) not in originals:
                label = row["label"]
                if synthetic_counts[label] >= original_counts[label]:
                    continue
                synthetic_counts[label] += 1
            capped.append(row)
        train = capped
        heldout = [row for row in original_train if assignments[_id_key(row["id"])] == fold]
        predict = [row for row in correction_train if assignments[_id_key(row["id"])] == fold]
        if not train or not predict:
            raise ValueError("Every OOF fold must have classifier training and correction prediction rows")
        plans.append((train, heldout, predict))
    return plans


def _correction_split(config, original_path, original_train, original_validation, explicit):
    from .two_phase import resolve_data

    correction_config = copy.deepcopy(config)
    correction_config["data"].update(path=config["data"].get("correction_path"),
                                     filename=config["data"].get("correction_filename", "VGEC_balanced.json"))
    path = resolve_data(correction_config, explicit)
    seed = config["run"]["seed"]
    fraction = config["data"]["validation_fraction"]
    train, validation, info = load_training_split(path, fraction, seed)
    if info["format"] != "vgec-balanced-v3":
        raise ValueError("Correction hints require the seed-42 balanced v3 bundle")
    expected = {_id_key(row["id"]): (row["source"], row["label"], row["target"]) for row in original_validation}
    actual = {_id_key(row["id"]): (row["source"], row["label"], row["target"]) for row in validation}
    if actual != expected:
        raise ValueError("Original and correction holdouts must have identical typed IDs, sources, labels, and targets")
    provenance = info["provenance"]
    if {_id_key(value) for value in provenance["original_train_ids"]} != {_id_key(row["id"]) for row in original_train}:
        raise ValueError("Correction bundle original_train_ids differ from the original classifier training partition")
    original_hash = hashlib.sha256(original_path.read_bytes()).hexdigest()
    if info.get("original_sha256") not in (None, original_hash):
        raise ValueError("Correction bundle was built from different original data")
    original_rows = {_id_key(row["id"]): row for row in original_train}
    for row in train:
        key = _id_key(row["id"])
        if key in original_rows and (row["source"], row["label"]) != (original_rows[key]["source"], original_rows[key]["label"]):
            raise ValueError("Retained correction originals must preserve original sources and labels")
    return path, train, validation, provenance


def main():
    from .two_phase import load_stage_config, resolve_data, stage_output

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data")
    parser.add_argument("--correction-data")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", choices=["cuda", "cpu"])
    parser.add_argument("--skip-hints", action="store_true", help="Train the main classifier only; not a connected pipeline run.")
    args = parser.parse_args()
    config = load_stage_config(args.config, expected_task="classification")
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
    enabled = config.get("oof", {}).get("enabled", True) and not args.skip_hints
    plans = []
    if enabled:
        correction_path, correction_train, correction_validation, provenance = _correction_split(
            config, path, original_train, validation, args.correction_data)
        epochs = config["oof"]["epochs"]
        if type(epochs) is not int or epochs < 1:
            raise ValueError("oof.epochs must be a positive integer")
        grouping = {"train_rows": provenance["train_rows"] + classifier_info["provenance"]["train_rows"]}
        plans = _connected_folds(original_train, correction_train, grouping, config["oof"]["folds"], seed, train)
    output = stage_output(config, args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "classifier"
    config["data"]["path"] = str(path)
    config["data"]["augmentation_path"] = str(mixed_path)
    if enabled:
        config["data"]["correction_path"] = str(correction_path)
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
        "original_train_count": len(original_train), "synthetic_train_count": len(train) - len(original_train),
        "model": config["model"], "resolved_model_revision": config["model"]["revision"],
        "preprocessing": preprocessing, "saved_classifier_reloaded_for_evaluation": True,
        "train_count": len(train), "validation_count": len(validation),
        "hints_enabled": bool(enabled), **stats,
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "accelerate", "peft", "bitsandbytes")},
    }
    if enabled:
        hints = {"seed": seed, "validation_fraction": fraction,
                 "classifier_provenance": classifier_info["provenance"],
                 "correction_data_sha256": hashlib.sha256(correction_path.read_bytes()).hexdigest(),
                 "train": [], "validation": classify(checkpoint, correction_validation, device, batch_size), "folds": []}
        for fold, (fold_train, heldout_originals, fold_predict) in enumerate(plans):
            fold_dir = output / "oof" / f"fold-{fold}"
            fold_dir.mkdir(parents=True)
            fold_checkpoint = fold_dir / "classifier"
            artifact = {"fold": fold, "train_ids": [row["id"] for row in fold_train],
                        "predict_ids": [row["id"] for row in fold_predict],
                        "heldout_original_ids": [row["id"] for row in heldout_originals],
                        "epochs": epochs, "seed": seed + fold + 1,
                        "grouping": "normalized source union synthetic parents across both mixed datasets",
                        "training_labels": "original plus capped synthetic labels", "outer_validation_used": False}
            write_json(fold_dir / "fold.json", artifact)
            fold_stats = _fit(config, fold_train, None, fold_checkpoint, fold_dir,
                              device, artifact["seed"], preprocessing, fixed_epochs=epochs)
            fold_predictions = classify(fold_checkpoint, fold_predict, device, batch_size)
            hints["train"].extend({**row, "fold": fold} for row in fold_predictions)
            hints["folds"].append(artifact)
            write_json(fold_dir / "predictions.json", comparison_rows(fold_predict, fold_predictions))
            write_json(fold_dir / "run_manifest.json", {**artifact, **fold_stats,
                       "model": config["model"], "preprocessing": preprocessing,
                       "saved_classifier_reloaded_for_prediction": True})
        by_id = {_id_key(row["id"]): row for row in hints["train"]}
        if len(hints["train"]) != len(correction_train) or set(by_id) != {_id_key(row["id"]) for row in correction_train}:
            raise RuntimeError("OOF predictions must cover every correction training row exactly once")
        hints["train"] = [by_id[_id_key(row["id"])] for row in correction_train]
        write_json(output / "hints.json", hints)
        manifest.update(correction_data_path=str(correction_path),
                        correction_data_sha256=hints["correction_data_sha256"],
                        hints_path=str(output / "hints.json"), oof_folds=hints["folds"])
    write_json(output / "run_manifest.json", manifest)
    print(json.dumps({"output_dir": str(output), "classifier_path": str(checkpoint),
                      "macro_f1": metrics["classification"]["macro_f1"], "hints_enabled": bool(enabled)}), flush=True)


if __name__ == "__main__":
    main()
