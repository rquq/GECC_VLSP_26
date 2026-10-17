"""LoRA/QLoRA soft hierarchical classification with a saved score head."""
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
from .runtime import CompleteEpochMixin, load_config, resolve_data, stage_output, write_json
from .prediction_export import comparison_rows
from .data import load_classifier_split
from .tracking import TrackingRun, publish_stage_result, records_identity, tracked_entrypoint


def _organizer_partition(path, fraction, seed, classifier_info):
    """Use the bundle's grouped holdout instead of regenerating the legacy split."""
    if classifier_info.get("format") in ("vgec-grouped-validation-v1", "vgec-quality-mixture-v1", "vgec-diverse-mixture-v1"):
        train_ids = {_id_key(i) for i in classifier_info["provenance"]["original_train_ids"]}
        return ([r for r in load_records(path) if _id_key(r["id"]) in train_ids],
                classifier_info["validation_records"])
    return ((load_records(path), []) if fraction == 0 else
            split_records(load_records(path), fraction, seed))


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


def placement_preflight(settings, device="cuda", *, training=True):
    """Resolve rank-local placement and supported precision without loading weights."""
    import torch

    device = torch.device(device)
    if settings.get("device_map", "single") == "balanced":
        if settings.get("load_in_4bit"):
            raise ValueError("Balanced classifier placement supports nonquantized LoRA")
        from .correction import placement_preflight as decoder_placement
        placement = decoder_placement(settings, device.type, task="classification", training=training)
        return {**placement, "device": "cuda:0",
                "compute_dtype": placement["compute_dtype"].removeprefix("torch.")}
    quantized = settings["load_in_4bit"]
    if device.type == "cpu":
        if quantized:
            raise ValueError("Quantized classifier requires CUDA; use load_in_4bit=false for a small CPU test")
        return {"device": "cpu", "compute_dtype": "float32", "quantized": False}
    if device.type != "cuda":
        raise ValueError("Classifier device must be CUDA or CPU")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    index = int(os.environ.get("LOCAL_RANK", "0")) if device.index is None and int(os.environ.get("WORLD_SIZE", "1")) > 1 else (
        torch.cuda.current_device() if device.index is None else device.index)
    requested = settings["compute_dtype"]
    if requested not in ("bfloat16", "float16", "float32"):
        raise ValueError("model.compute_dtype must be bfloat16, float16, or float32")
    with torch.cuda.device(index):
        from .runtime import native_bf16_supported
        actual = "float16" if requested == "bfloat16" and not native_bf16_supported() else requested
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
    from transformers.models.auto.tokenization_auto import get_tokenizer_config
    directory = Path(path)
    metadata = directory / "tokenizer_config.json"
    saved = (json.loads(metadata.read_text()) if metadata.is_file() else
             get_tokenizer_config(path, **kwargs) if not directory.is_dir() else {})
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
    # Native Mistral/Llama checkpoints provide a serialized fast backend.
    native_fast = saved.get("tokenizer_class") in {"LlamaTokenizer", "LlamaTokenizerFast"}
    if native_fast and not directory.is_dir():
        from huggingface_hub import hf_hub_download
        download_kwargs = {key: kwargs[key] for key in (
            "revision", "cache_dir", "token", "local_files_only", "force_download", "subfolder"
        ) if key in kwargs}
        # Fetch the native backend and metadata explicitly so cold workers use
        # one complete pinned snapshot, with no remote-resolution fallback.
        assets = [Path(hf_hub_download(str(path), filename, **download_kwargs))
                  for filename in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")]
        directory = assets[0].parent
        if any(not asset.is_file() or asset.parent != directory for asset in assets):
            raise ValueError("Native fast tokenizer assets must exist in one pinned snapshot")
        path = str(directory)
        kwargs.update(local_files_only=True, from_slow=False)
    if native_fast and (directory / "tokenizer.json").is_file():
        # Transformers 4.51 LlamaTokenizerFast forces slow conversion whenever
        # add_prefix_space is specified. Its serialized backend already encodes
        # that behavior; None preserves it without requiring SentencePiece.
        kwargs.update(add_prefix_space=None, from_slow=False)
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=generic_fast or backend_only or native_fast, **kwargs)
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
    if placement.get("mode") == "balanced":
        kwargs.update(device_map=placement["device_map"], low_cpu_mem_usage=True)
    model = AutoModelForSequenceClassification.from_pretrained(settings["name"], **kwargs)
    model.config.classifier_embeddings_resized = len(tokenizer) > model.get_input_embeddings().num_embeddings
    if model.config.classifier_embeddings_resized:
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    # QLoRA preparation casts some weights to FP32; model.dtype alone would then
    # misidentify the Trainer's mixed-precision forward context on reload.
    model.config.classifier_compute_dtype = placement["compute_dtype"]
    if not placement["quantized"] and placement.get("mode") != "balanced":
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
    if (Path(checkpoint) / "ensemble.json").is_file():
        from .classifier_ensemble import classify as ensemble_classify
        return ensemble_classify(checkpoint, records, device, batch_size)
    from .reproducibility import seed_everything
    seed_everything(42)
    from peft import PeftModel, prepare_model_for_kbit_training

    checkpoint = Path(checkpoint)
    preprocessing = json.loads((checkpoint / "preprocessing.json").read_text(encoding="utf-8"))
    saved = json.loads((checkpoint / "classifier_config.json").read_text(encoding="utf-8"))
    _require_hierarchy(saved)
    if preprocessing["labels"] != list(LABELS) or saved["labels"] != list(LABELS):
        raise ValueError("Saved classifier label mapping must be 0..4")
    placement = placement_preflight(saved["model"], device, training=False)
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

    class HierarchicalTrainer(CompleteEpochMixin, Trainer):
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
            head_device = output.logits.device
            labels = labels.to(head_device)
            if torch.is_tensor(num_items_in_batch):
                num_items_in_batch = num_items_in_batch.to(head_device)
            loss = _weighted_hierarchical_loss(output.logits, labels, self.loss_weights.to(head_device),
                                               num_items_in_batch)
            loss = loss.to(self.args.device)
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


def _fit(config, train, validation, checkpoint, artifact_dir, device, seed, preprocessing,
         fixed_epochs=None, *, tracking):
    """Fresh mixed-data LoRA/head fit; only main fit selects on outer holdout."""
    import torch
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import (DataCollatorWithPadding, TrainerCallback,
                              TrainingArguments)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    tracking.start(config, artifact_dir, "classification-oof" if fixed_epochs is not None else "classification",
                   group=config.get("_tracking_group"),
                   metadata={"seed": seed, "fixed_epochs": fixed_epochs,
                             "fold": artifact_dir.name if fixed_epochs is not None else None,
                             "train": records_identity(train),
                             "validation": records_identity(validation or [])})

    def barrier():
        if world_size > 1:
            torch.distributed.barrier()

    if not train:
        raise ValueError("Every classifier fit needs training rows")
    if "algorithm" in config["training"]:
        raise ValueError("training.algorithm is obsolete; classifier training uses single-forward hierarchical loss")
    weighting = _class_weights(train, config["training"].get("class_weight_power", 0.0))
    from .reproducibility import seed_everything
    reproducibility = seed_everything(seed)
    placement = placement_preflight(config["model"], device)
    device = placement["device"]
    print(json.dumps({"classifier_placement": placement}), flush=True)
    settings = {**config["model"], "compute_dtype": placement["compute_dtype"]}
    tokenizer = _load_tokenizer(settings["name"], revision=settings["revision"])
    added_pad = _configure_padding(tokenizer)
    model = _load_base(settings, tokenizer, placement)
    resized_embeddings = model.config.classifier_embeddings_resized
    checkpoint_kwargs = {"use_reentrant": False}
    gradient_checkpointing = config["training"]["args"].get("gradient_checkpointing", True)
    if placement["quantized"]:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=gradient_checkpointing,
            gradient_checkpointing_kwargs=checkpoint_kwargs)
    lora = {**config["lora"], "task_type": TaskType.SEQ_CLS, "modules_to_save": ["score"],
            "revision": settings["revision"]}
    model = get_peft_model(model, LoraConfig(**lora))
    from .runtime import promote_trainable_parameters
    promote_trainable_parameters(model)
    if gradient_checkpointing:
        model.enable_input_require_grads()
    if world_size == 1:
        # Avoid implicit DataParallel for explicitly single-process runs.
        model.is_parallelizable = True
        model.model_parallel = True
    classifier_config = {
        "model": settings, "requested_compute_dtype": config["model"]["compute_dtype"],
        "lora": {**config["lora"], "task_type": "SEQ_CLS", "modules_to_save": ["score"],
                 "revision": settings["revision"]},
        "num_labels": len(LABELS), "labels": list(LABELS),
        "architecture": _architecture(),
        "inference_precision": "CUDA autocast at model.compute_dtype; FP32 composition; no CPU autocast",
        "training_seed": seed,
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
    if type(microbatch) is not int or microbatch < 1 or type(effective) is not int or effective < microbatch * world_size or effective % (microbatch * world_size):
        raise ValueError("effective_batch_size must be a positive multiple of microbatch * WORLD_SIZE")
    accumulation = effective // (microbatch * world_size)
    if fixed_epochs is not None:
        args.update(num_train_epochs=fixed_epochs, max_steps=-1)
    epochs = float(args["num_train_epochs"])
    if fixed_epochs is None and args.get("max_steps", -1) < 1 and epochs >= 1 and epochs.is_integer():
        # Trainer 4.x floors the estimated updates per epoch, although its loop
        # updates the final partial accumulation window. Resolve an explicit
        # budget so integer epoch requests consume every batch in every epoch.
        microbatches = math.ceil(math.ceil(len(encoded) / world_size) / microbatch)
        args["max_steps"] = math.ceil(microbatches / accumulation) * int(epochs)
    args.update(bf16=device.startswith("cuda") and placement["compute_dtype"] == "bfloat16",
                fp16=device.startswith("cuda") and placement["compute_dtype"] == "float16",
                gradient_checkpointing=gradient_checkpointing,
                gradient_checkpointing_kwargs=checkpoint_kwargs)
    if world_size > 1:
        args.update(ddp_find_unused_parameters=False, ddp_broadcast_buffers=False,
                    ddp_timeout=86400, average_tokens_across_devices=False)
    training_args = TrainingArguments(
        **args, output_dir=str(artifact_dir / "trainer"), use_cpu=device == "cpu",
        gradient_accumulation_steps=accumulation,
        eval_strategy="no", save_strategy="no", report_to="none",
        logging_nan_inf_filter=False, seed=seed, data_seed=seed, label_names=["labels"])
    if world_size == 1:
        # Prevent Trainer's implicit DataParallel for a single-process fit.
        training_args._n_gpu = 0 if device == "cpu" else 1
    actual_device = training_args.device
    expected_device = torch.device(device)
    if (actual_device.type != expected_device.type
            or (expected_device.type == "cuda" and actual_device.index != expected_device.index)):
        raise ValueError(f"Classifier device {actual_device} must match rank-local {expected_device}")
    history = []
    best = {"epoch": None, "macro_f1": None}
    batch_size = args.get("per_device_eval_batch_size", 8)
    if rank == 0 and validation is not None:
        baseline = _classification_metrics(validation, _predict(model, tokenizer, validation, preprocessing, device, batch_size))
        tracking.evaluation(baseline, "baseline/classification")
        model.train()

    def save_model():
        checkpoint.mkdir(parents=True, exist_ok=True)
        trainer.accelerator.unwrap_model(model).save_pretrained(checkpoint, save_embedding_layers=resized_embeddings)
        tokenizer.save_pretrained(checkpoint)
        write_json(checkpoint / "preprocessing.json", preprocessing)
        write_json(checkpoint / "classifier_config.json", classifier_config)

    loss_weights = torch.tensor(weighting["weights"], dtype=torch.float32, device=device)

    HierarchicalTrainer = _hierarchical_trainer_class()

    class EpochEvaluation(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):
            barrier()
            _finite_parameters(model)
            if rank == 0:
                unwrapped = trainer.accelerator.unwrap_model(model)
                entry = {"epoch": state.epoch, "optimizer_steps": state.global_step}
                if validation is not None:
                    predictions = _predict(unwrapped, tokenizer, validation, preprocessing, device, batch_size)
                    metrics = _classification_metrics(validation, predictions)
                    entry["classification"] = metrics
                    tracking.evaluation(metrics, "validation/classification", optimizer_step=state.global_step, epoch=state.epoch)
                    if best["macro_f1"] is None or metrics["macro_f1"] > best["macro_f1"]:
                        best.update(epoch=state.epoch, macro_f1=metrics["macro_f1"])
                        save_model()
                history.append(entry)
                write_json(artifact_dir / "epoch_history.json", history)
                print(json.dumps(entry, ensure_ascii=False), flush=True)
                unwrapped.train()
            if world_size > 1:
                selected = [best if rank == 0 else None]
                torch.distributed.broadcast_object_list(selected, src=0)
                best.update(selected[0])
            barrier()
            return control

    trainer = HierarchicalTrainer(model=model, args=training_args, train_dataset=encoded,
                          data_collator=DataCollatorWithPadding(tokenizer), callbacks=[EpochEvaluation(), tracking.callback()],
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
    barrier()
    if rank == 0:
        if validation is None:
            save_model()
        elif best["epoch"] is None:
            raise RuntimeError("Main classifier did not evaluate or save an epoch")
        write_json(artifact_dir / "trainer_history.json", trainer.state.log_history)
    barrier()
    stats = {
        "reproducibility": reproducibility,
        "training_loss": result.training_loss, "optimizer_steps": result.global_step,
        "trainable_parameters": trainable_parameters, "total_parameters": total_parameters,
        "classifier_config": classifier_config, "placement": placement,
        "parameters_finite": True, "loss_finite": True,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": effective, "world_size": world_size, "training_args": training_args.to_dict(),
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
    if rank == 0:
        tracking.log({"training_loss": result.training_loss}, "training")
        if best["epoch"] is not None:
            tracking.summary({"best_epoch": best["epoch"]})
        tracking.model(checkpoint, "classifier", metadata={"seed": seed, "best_epoch": best["epoch"],
                                                        "selection_macro_f1": best["macro_f1"]}, aliases=["latest", "selected"])
    trainer.model = None
    trainer.model_wrapped = None
    trainer.callback_handler.model = None
    trainer.callback_handler.optimizer = None
    trainer.callback_handler.lr_scheduler = None
    trainer.optimizer = None
    trainer.accelerator.free_memory()
    del trainer, model, tokenizer
    _release(device)
    return stats


def _connected_folds(original_train, correction_train, provenance, folds, seed, classifier_train):
    """Keep source duplicates, synthetic parents and contrast/template families together."""
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
    families = {}
    for row in provenance.get("train_rows", []):
        key = _id_key(row["id"])
        metadata = row.get("authored_provenance", {})
        for field in ("family_id", "template_id"):
            value = metadata.get(field)
            if value:
                family = (field, value)
                if family in families:
                    union(key, families[family])
                else:
                    families[family] = key
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
                limit = min(40, original_counts[label]) if label == 0 else original_counts[label]
                if synthetic_counts[label] >= limit:
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
    from .runtime import resolve_data

    correction_config = copy.deepcopy(config)
    correction_config["data"].update(path=config["data"].get("correction_path"),
                                     filename=config["data"].get("correction_filename", "VGEC_balanced.json"))
    path = resolve_data(correction_config, explicit)
    seed = config["run"]["seed"]
    fraction = config["data"]["validation_fraction"]
    train, validation, info = load_training_split(path, fraction, seed)
    if info["format"] not in {"vgec-balanced-v3", "vgec-reviewed-v1"}:
        raise ValueError("Correction hints require a seed-42 balanced or reviewed bundle")
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
    reviewed_labels = {}
    if info["format"] == "vgec-reviewed-v1":
        original_raw = {_id_key(row["ID"]): row for row in json.loads(original_path.read_text())}
        for review in provenance["authored_overrides"]:
            key = _id_key(review["before"]["ID"])
            if review["before"] != original_raw.get(key):
                raise ValueError("Correction review original differs from organizer JSON")
            reviewed_labels[key] = review["after"]["Output"]["Error_Type_Id"]
    for row in train:
        key = _id_key(row["id"])
        if key in original_rows and (row["source"], row["label"]) != (original_rows[key]["source"], reviewed_labels.get(key, original_rows[key]["label"])):
            raise ValueError("Retained correction originals must preserve original sources and labels")
    return path, train, validation, provenance


@tracked_entrypoint
def main(*, tracking):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data")
    parser.add_argument("--correction-data")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", choices=["cuda", "cpu"])
    parser.add_argument("--skip-hints", action="store_true", help="Train the main classifier only; not a connected pipeline run.")
    parser.add_argument("--fit-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = load_config(args.config, expected_task="classification")
    device = args.device or config["run"]["device"]
    config["run"]["device"] = device
    seed = config["run"]["seed"]
    import torch
    from datetime import timedelta

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    requested_world_size = config["run"].get("world_size", 1)
    if type(requested_world_size) is not int or requested_world_size not in (1, 2):
        raise ValueError("run.world_size must be 1 or 2")
    if world_size != requested_world_size:
        raise ValueError(f"run.world_size={requested_world_size} requires torchrun --standalone --nproc_per_node={requested_world_size} -m model.classification; actual WORLD_SIZE={world_size}")

    def barrier():
        if world_size > 1:
            torch.distributed.barrier()

    if device not in ("cuda", "cpu"):
        raise ValueError("run.device must be cuda or cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device == "cuda":
        if torch.cuda.device_count() < world_size:
            raise RuntimeError(f"Classification requires {world_size} visible CUDA devices")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if world_size > 1:
        torch.distributed.init_process_group(
            backend="nccl" if device == "cuda" else "gloo", timeout=timedelta(seconds=86400))
    path = resolve_data(config, args.data)
    fraction = config["data"]["validation_fraction"]
    final_fit = fraction == 0
    mixed_config = copy.deepcopy(config)
    mixed_config["data"].update(path=config["data"].get("augmentation_path"),
                                filename=config["data"]["augmentation_filename"])
    mixed_path = resolve_data(mixed_config, None)
    from .runtime import split_seed
    data_seed = split_seed(config)
    train, classifier_info = load_classifier_split(path, mixed_path, fraction, data_seed)
    original_train, validation = _organizer_partition(path, fraction, data_seed, classifier_info)
    preprocessing = _preprocessing(config)
    enabled = config.get("oof", {}).get("enabled", False) and not args.skip_hints
    if enabled and classifier_info["format"] in ("vgec-grouped-validation-v1", "vgec-quality-mixture-v1", "vgec-diverse-mixture-v1"):
        raise ValueError("Grouped validation recipe uses one holdout without OOF hints")
    plans = []
    if final_fit and enabled:
        raise ValueError("Final fits have no validation or OOF hints")
    if enabled:
        correction_path, correction_train, correction_validation, provenance = _correction_split(
            config, path, original_train, validation, args.correction_data)
        epochs = config["oof"]["epochs"]
        if type(epochs) is not int or epochs < 1:
            raise ValueError("oof.epochs must be a positive integer")
        grouping = {"train_rows": provenance["train_rows"] + classifier_info["provenance"]["train_rows"]}
        plans = _connected_folds(original_train, correction_train, grouping, config["oof"]["folds"], seed, train)
    resolved_output = [str(stage_output(config, args.output_dir)) if rank == 0 else None]
    if world_size > 1:
        torch.distributed.broadcast_object_list(resolved_output, src=0)
    output = Path(resolved_output[0])
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    barrier()
    checkpoint = output / "classifier"
    config["data"]["path"] = str(path)
    config["data"]["augmentation_path"] = str(mixed_path)
    if enabled:
        config["data"]["correction_path"] = str(correction_path)
    if rank == 0:
        write_json(output / "config.json", config)
        write_json(output / "train.json", train)
        write_json(output / "validation.json", validation)
        write_json(output / "preprocessing.json", preprocessing)
    barrier()
    config["_tracking_group"] = os.environ.get("WANDB_RUN_GROUP") or output.name
    if args.fit_only and (enabled or world_size != 1):
        raise ValueError("Separate classifier verification requires one process and no hints")
    stats = _fit(config, train, None if final_fit else validation, checkpoint, output, device, seed, preprocessing, tracking=tracking)
    batch_size = config["training"]["args"].get("per_device_eval_batch_size", 8)
    if rank == 0:
        manifest = {
            "task": "classification", "config": config, "seed": seed, "split_seed": data_seed, "validation_fraction": fraction,
            "device": device, "world_size": world_size, "output_dir": str(output), "classifier_path": str(checkpoint),
            "data_path": str(path), "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "classifier_data_path": str(mixed_path),
            "classifier_data_sha256": hashlib.sha256(mixed_path.read_bytes()).hexdigest(),
            "original_train_count": len(original_train),
            "retained_original_train_count": len(classifier_info["provenance"]["retained_original_train_ids"]),
            "quarantined_original_train_count": len(classifier_info["provenance"]["quarantined_ids"]),
            "synthetic_train_count": len(classifier_info["provenance"]["selected_synthetic_ids"]),
            "external_train_count": len(classifier_info["provenance"].get("external_train_ids", [])),
            "resampled_train_count": len(classifier_info["provenance"].get("resampled_train_ids", [])),
            "model": config["model"], "resolved_model_revision": config["model"]["revision"],
            "preprocessing": preprocessing, "saved_classifier_reloaded_for_evaluation": False,
            "saved_classifier_reload_verified": False, "final_fit": final_fit,
            "train_count": len(train), "validation_count": len(validation),
            "hints_enabled": bool(enabled), **stats,
            "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "accelerate", "peft", "bitsandbytes")},
        }
        write_json(output / "training_complete.json", manifest)
    if args.fit_only:
        return
    if rank == 0:
        from .classifier_finalize import finalize
        metrics = finalize(output, device)
        tracking.complete("classifier", metrics, best_epoch=stats["best_epoch"])
        manifest.update(saved_classifier_reloaded_for_evaluation=not final_fit,
                        saved_classifier_reload_verified=True)
    barrier()
    if enabled:
        if rank == 0:
            hints = {"seed": seed, "validation_fraction": fraction,
                     "classifier_provenance": classifier_info["provenance"],
                     "correction_data_sha256": hashlib.sha256(correction_path.read_bytes()).hexdigest(),
                     "train": [], "validation": classify(checkpoint, correction_validation, device, batch_size), "folds": []}
        barrier()
        for fold, (fold_train, heldout_originals, fold_predict) in enumerate(plans):
            fold_dir = output / "oof" / f"fold-{fold}"
            if rank == 0:
                fold_dir.mkdir(parents=True)
            barrier()
            fold_checkpoint = fold_dir / "classifier"
            artifact = {"fold": fold, "train_ids": [row["id"] for row in fold_train],
                        "predict_ids": [row["id"] for row in fold_predict],
                        "heldout_original_ids": [row["id"] for row in heldout_originals],
                        "epochs": epochs, "seed": seed + fold + 1,
                        "grouping": "normalized source union synthetic parents and contrast/template families across both mixed datasets",
                        "training_labels": "quality-selected originals plus capped synthetic contrasts", "outer_validation_used": False}
            if rank == 0:
                write_json(fold_dir / "fold.json", artifact)
            barrier()
            with TrackingRun() as fold_tracking:
                fold_stats = _fit(config, fold_train, None, fold_checkpoint, fold_dir,
                                  device, artifact["seed"], preprocessing, fixed_epochs=epochs,
                                  tracking=fold_tracking)
                if rank == 0:
                    fold_predictions = classify(fold_checkpoint, fold_predict, device, batch_size)
                    fold_metrics = _classification_metrics(fold_predict, fold_predictions)
                    hints["train"].extend({**row, "fold": fold} for row in fold_predictions)
                    hints["folds"].append(artifact)
                    write_json(fold_dir / "predictions.json", comparison_rows(fold_predict, fold_predictions))
                    write_json(fold_dir / "run_manifest.json", {**artifact, **fold_stats,
                               "model": config["model"], "preprocessing": preprocessing,
                               "saved_classifier_reloaded_for_prediction": True})
                    fold_report = {"count": len(fold_predict), "classification": fold_metrics}
                    write_json(fold_dir / "metrics.json", fold_report)
                    fold_tracking.complete(f"fold-{fold + 1}", fold_report)
                barrier()
        if rank == 0:
            by_id = {_id_key(row["id"]): row for row in hints["train"]}
            if len(hints["train"]) != len(correction_train) or set(by_id) != {_id_key(row["id"]) for row in correction_train}:
                raise RuntimeError("OOF predictions must cover every correction training row exactly once")
            hints["train"] = [by_id[_id_key(row["id"])] for row in correction_train]
            write_json(output / "hints.json", hints)
            oof_metrics = {"count": len(correction_train),
                           "classification": _classification_metrics(correction_train, hints["train"])}
            write_json(output / "oof_metrics.json", oof_metrics)
            publish_stage_result("oof", oof_metrics)
            manifest.update(correction_data_path=str(correction_path),
                            correction_data_sha256=hints["correction_data_sha256"],
                            hints_path=str(output / "hints.json"), oof_folds=hints["folds"])
    if rank == 0:
        write_json(output / "run_manifest.json", manifest)
        print(json.dumps({"output_dir": str(output), "classifier_path": str(checkpoint),
                          "macro_f1": (metrics["classification"] or {}).get("macro_f1"), "hints_enabled": bool(enabled)}), flush=True)
    barrier()
    if world_size > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
