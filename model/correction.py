"""Train Sailor2 QLoRA from out-of-fold Qwen3 classifier hints, then reload it."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path

from .data import _id_key, _source_key, _valid_label, load_training_split
from .runtime import balanced_limit, selection_key, write_json
from .prediction_export import comparison_rows

SELECTION_METRICS = (
    "corrections.error_only.gleu_plus", "corrections.error_only.exact_match",
    "clean_preservation.exact_source_preservation",
)


def _hint_index(records, hints):
    if not isinstance(hints, list):
        raise ValueError("Hints must be an array")
    indexed = {}
    for hint in hints:
        if not isinstance(hint, dict) or "id" not in hint or not _valid_label(hint.get("label")):
            raise ValueError("Every hint must contain a typed ID and label in 0..4")
        key = _id_key(hint["id"])
        if key in indexed:
            raise ValueError(f"Duplicate hint ID: {hint['id']!r}")
        probabilities = hint.get("probabilities")
        if (not isinstance(probabilities, list) or len(probabilities) != 5
                or any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1
                       for p in probabilities)
                or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-4)
                or max(range(5), key=lambda label: probabilities[label]) != hint["label"]):
            raise ValueError(f"Invalid classifier probabilities for ID {hint['id']!r}")
        indexed[key] = hint
    keys = [_id_key(row["id"]) for row in records]
    if len(set(keys)) != len(keys) or set(indexed) != set(keys):
        raise ValueError("Hints must exactly cover record IDs, preserving integer/string types")
    return indexed


def validate_hints(artifact, train, validation, provenance, classifier_train, data_path, seed, fraction):
    """Check OOF declarations against connected source/parent groups, not just IDs."""
    if not isinstance(artifact, dict):
        raise ValueError("Classifier hints must be an object")
    if type(artifact.get("seed")) is not int or artifact["seed"] != seed:
        raise ValueError("Classifier hints seed differs from correction seed")
    if artifact.get("validation_fraction") != fraction:
        raise ValueError("Classifier hints validation_fraction differs")
    if artifact.get("correction_data_sha256") != hashlib.sha256(Path(data_path).read_bytes()).hexdigest():
        raise ValueError("Classifier hints do not match the correction dataset SHA256")
    train_hints = _hint_index(train, artifact.get("train"))
    validation_hints = _hint_index(validation, artifact.get("validation"))
    classifier_rows = {_id_key(row["id"]): row for row in classifier_train}
    if len(classifier_rows) != len(classifier_train) or not classifier_rows:
        raise ValueError("Classifier train.json must contain unique mixed training records")
    heldout = {_id_key(row["id"]) for row in validation}
    if classifier_rows.keys() & heldout:
        raise ValueError("Classifier training IDs overlap outer validation")
    original_ids = provenance.get("provenance", {}).get("original_train_ids")
    mixed = artifact.get("classifier_provenance")
    if not isinstance(mixed, dict) or original_ids is None:
        raise ValueError("Mixed classifier hints require original and classifier provenance")
    original_keys = {_id_key(value) for value in original_ids}
    if ({_id_key(value) for value in mixed["retained_original_train_ids"]} != original_keys
            or {_id_key(value) for value in mixed["original_train_ids"]} != original_keys
            or original_keys | {_id_key(value) for value in mixed["selected_synthetic_ids"]} != set(classifier_rows)):
        raise ValueError("Classifier train.json differs from declared mixed partition")
    originals = {key: classifier_rows[key] for key in original_keys}
    parents = {}
    sources = {}

    def find(key):
        parents.setdefault(key, key)
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    def union(left, right):
        parents[find(left)] = find(right)

    all_rows = dict(classifier_rows)
    for row in train:
        key = _id_key(row["id"])
        if key in all_rows and all_rows[key] != row:
            raise ValueError("Classifier original and correction record with the same ID differ")
        all_rows[key] = row
    for key, row in all_rows.items():
        find(key)
        source = _source_key(row["source"])
        if source in sources:
            union(key, sources[source])
        else:
            sources[source] = key
    if set(sources) & {_source_key(row["source"]) for row in validation}:
        raise ValueError("Classifier training sources overlap outer validation")
    for metadata in provenance.get("provenance", {}).get("train_rows", []) + mixed["train_rows"]:
        key = _id_key(metadata["id"])
        for parent in metadata.get("parent_ids", []):
            parent_key = _id_key(parent)
            if parent_key not in originals or parent_key in heldout:
                raise ValueError("Synthetic parent is not in the classifier training partition")
            union(key, parent_key)
    components = {key: find(key) for key in parents}
    folds = artifact.get("folds")
    if not isinstance(folds, list) or len(folds) < 2:
        raise ValueError("OOF hints require at least two declared folds")
    seen_folds, predicted = set(), set()
    for fold in folds:
        number = fold.get("fold")
        if type(number) is not int or number < 0 or number in seen_folds:
            raise ValueError("Fold numbers must be unique nonnegative integers")
        seen_folds.add(number)
        training_ids, prediction_ids = fold.get("train_ids"), fold.get("predict_ids")
        if not isinstance(training_ids, list) or not isinstance(prediction_ids, list):
            raise ValueError("Fold train_ids and predict_ids must be arrays")
        training_keys = {_id_key(value) for value in training_ids}
        prediction_keys = {_id_key(value) for value in prediction_ids}
        if (not training_keys or not prediction_keys or len(training_keys) != len(training_ids)
                or len(prediction_keys) != len(prediction_ids) or not training_keys <= classifier_rows.keys()
                or not prediction_keys <= train_hints.keys() or predicted & prediction_keys):
            raise ValueError("Fold declarations have invalid, duplicate, or overlapping IDs")
        excluded_components = {components[key] for key in prediction_keys}
        if any(components[key] in excluded_components for key in training_keys):
            raise ValueError("OOF training leaks a held-out normalized source or synthetic parent group")
        for key in prediction_keys:
            if type(train_hints[key].get("fold")) is not int or train_hints[key]["fold"] != number:
                raise ValueError("Training hint fold differs from its prediction declaration")
        predicted.update(prediction_keys)
    if predicted != set(train_hints):
        raise ValueError("OOF fold prediction declarations must exactly cover correction training")
    # A group cannot be predicted across multiple folds, even when no original is retained.
    group_folds = {}
    for key, hint in train_hints.items():
        group = components[key]
        if group in group_folds and group_folds[group] != hint["fold"]:
            raise ValueError("Connected source/parent group is split across OOF folds")
        group_folds[group] = hint["fold"]
    return train_hints, validation_hints


def _prompt_ids(tokenizer, source, label, inference):
    descriptions = inference["hint_descriptions"]
    content = f"Gợi ý nhãn dự đoán: {label} — {descriptions[label]}.\nCâu gốc:\n{source}"
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": inference["prompt"]}, {"role": "user", "content": content}],
        tokenize=True, add_generation_prompt=True, return_dict=False)


def encode_record(tokenizer, record, hint, max_length, inference):
    prompt = _prompt_ids(tokenizer, record["source"], hint["label"], inference)
    answer = json.dumps({"corrected_sentence": record["target"]}, ensure_ascii=False)
    if tokenizer.eos_token_id is None:
        raise ValueError("Correction tokenizer must have an EOS token")
    completion = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
    ids = prompt + completion
    if len(ids) > max_length:
        raise ValueError(f"ID {record['id']!r}: {len(ids)} tokens exceeds max_length={max_length}; targets are never truncated")
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": [-100] * len(prompt) + completion}


def _parse_correction(text):
    try:
        value = json.loads(text)
        if (not isinstance(value, dict) or set(value) != {"corrected_sentence"}
                or not isinstance(value["corrected_sentence"], str) or not value["corrected_sentence"].strip()):
            raise ValueError("Expected JSON with only a nonblank corrected_sentence string")
        return value["corrected_sentence"], None
    except (ValueError, TypeError, KeyError) as error:
        return "", str(error)


def correct(model, tokenizer, records, hints, inference):
    """Generate for every row, including class 0; never override classifier labels."""
    import torch
    indexed = _hint_index(records, hints)
    generation = {"max_new_tokens": 192, "do_sample": False, "use_cache": True,
                  "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id}
    generation.update(inference["generation"])
    device = model.get_input_embeddings().weight.device
    model.eval()
    predictions = []
    with torch.inference_mode():
        for record in records:
            hint = indexed[_id_key(record["id"])]
            prompt = _prompt_ids(tokenizer, record["source"], hint["label"], inference)
            if len(prompt) > inference["max_length"]:
                raise ValueError(f"ID {record['id']!r}: correction prompt exceeds max_length; truncation is forbidden")
            ids = torch.tensor([prompt], device=device)
            output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), **generation)
            sequences = output.sequences if hasattr(output, "sequences") else output
            text = tokenizer.decode(sequences[0, len(prompt):], skip_special_tokens=True).strip()
            target, error = _parse_correction(text)
            predictions.append({"id": record["id"], "label": hint["label"], "target": target,
                                "raw_generation": text, "parse_error": error})
    return predictions


def _dtype(settings, device):
    import torch
    value = settings.get("compute_dtype", "auto")
    if value == "auto":
        if device == "cpu":
            return torch.float32
        supported = []
        count = 2 if settings.get("device_map", "balanced") == "balanced" else 1
        for index in range(count):
            with torch.cuda.device(index):
                supported.append(torch.cuda.is_bf16_supported())
        return torch.bfloat16 if all(supported) else torch.float16
    if value not in ("float16", "bfloat16", "float32"):
        raise ValueError("compute_dtype must be auto, float16, bfloat16, or float32")
    return getattr(torch, value)


def placement_preflight(settings, device="cuda"):
    """Build an explicit Qwen layer map and budget weights before loading them."""
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Correction uses single-process model parallelism, never torchrun/DDP")
    quantized = settings.get("load_in_4bit", False)
    if settings.get("load_in_8bit") or settings.get("quantization_config") is not None:
        raise ValueError("Use load_in_4bit for NF4 QLoRA; 8-bit/prequantized loading is unsupported")
    if quantized and device != "cuda":
        raise ValueError("Correction QLoRA requires CUDA")
    mode = settings.get("device_map", "balanced")
    if mode not in ("balanced", "single"):
        raise ValueError("device_map must be balanced (explicit two-GPU map) or single")
    if device not in ("cpu", "cuda"):
        raise ValueError("Correction device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if mode == "balanced" and (device != "cuda" or torch.cuda.device_count() < 2):
        raise RuntimeError("Balanced correction needs two CUDA GPUs; no CPU offload")
    dtype = _dtype(settings, device)
    config = AutoConfig.from_pretrained(settings["name"], revision=settings["revision"])
    if getattr(config, "quantization_config", None) is not None:
        raise ValueError("Use an unquantized base checkpoint and quantize it during loading")
    with init_empty_weights(include_buffers=True):
        skeleton = AutoModelForCausalLM.from_config(config, torch_dtype=dtype)

    def weight_bytes(module):
        if not quantized:
            return sum(p.numel() * p.element_size() for p in module.parameters())
        # NF4 packed weights plus metadata; k-bit preparation promotes other weights to FP32.
        total = 0
        for name, parameter in module.named_parameters():
            parent = module.get_submodule(name.rsplit(".", 1)[0]) if "." in name else module
            packed = isinstance(parent, torch.nn.Linear) and parent is not skeleton.lm_head and name.endswith("weight")
            total += math.ceil(parameter.numel() * (0.625 if packed else 4))
        return total
    if mode == "single":
        mapping = {"": "cuda:0" if device == "cuda" else "cpu"}
    else:
        if config.model_type not in ("qwen2", "qwen3") or config.tie_word_embeddings:
            raise ValueError("Balanced placement requires an untied Qwen decoder architecture")
        layer_sizes = [weight_bytes(layer) for layer in skeleton.model.layers]
        embedding = weight_bytes(skeleton.model.embed_tokens)
        ending = weight_bytes(skeleton.model.norm) + weight_bytes(skeleton.lm_head)
        split = min(range(1, len(layer_sizes)),
                    key=lambda index: abs(embedding + sum(layer_sizes[:index]) - ending - sum(layer_sizes[index:])))
        mapping = {"model.embed_tokens": "cuda:0", "model.rotary_emb": "cuda:0",
                   "model.norm": "cuda:1", "lm_head": "cuda:1"}
        mapping.update({f"model.layers.{index}": f"cuda:{int(index >= split)}" for index in range(len(layer_sizes))})
    weights = {}
    for name, parameter in skeleton.named_parameters():
        prefixes = [prefix for prefix in mapping if not prefix or name == prefix or name.startswith(prefix + ".")]
        if not prefixes:
            raise ValueError(f"Explicit device map does not cover {name}")
        destination = mapping[max(prefixes, key=len)]
        parent = skeleton.get_submodule(name.rsplit(".", 1)[0]) if "." in name else skeleton
        packed = quantized and isinstance(parent, torch.nn.Linear) and parent is not skeleton.lm_head and name.endswith("weight")
        size = math.ceil(parameter.numel() * (0.625 if packed else 4)) if quantized else parameter.numel() * parameter.element_size()
        weights[destination] = weights.get(destination, 0) + size
    del skeleton
    # Allow frozen weights plus FP32 LoRA/Adam states, checkpointed activations,
    # full-vocabulary FP32 loss logits, generation KV cache, and CUDA workspaces.
    reserve = 4 * 1024 ** 3
    budgets = {}
    for destination, size in weights.items():
        required = math.ceil(size * 1.10) + reserve
        free = None
        if destination.startswith("cuda"):
            index = int(destination.split(":")[1])
            free, total = torch.cuda.mem_get_info(index)
            if required > free:
                raise RuntimeError(f"{destination} has {free / 1024**3:.2f} GiB free; estimated weights plus conservative training reserve need {required / 1024**3:.2f} GiB. Use sufficient GPU memory or balanced placement; no offload.")
        else:
            total = None
        budgets[destination] = {"weight_bytes": size, "required_bytes": required,
                                "free_bytes": free, "total_bytes": total}
    return {"device": device, "mode": mode, "device_map": mapping, "compute_dtype": str(dtype),
            "quantized": quantized, "memory_budget": budgets,
            "resolved_revision": config._commit_hash or settings["revision"]}


def _load_base(settings, device, placement=None):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    placement = placement or placement_preflight(settings, device)
    dtype = _dtype(settings, device)
    quantization = None
    if placement["quantized"]:
        quantization = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type=settings["quant_type"],
            bnb_4bit_use_double_quant=settings["use_double_quant"])
    model = AutoModelForCausalLM.from_pretrained(
        settings["name"], revision=placement["resolved_revision"], torch_dtype=dtype,
        device_map=placement["device_map"], attn_implementation=settings.get("attn_implementation", "sdpa"),
        quantization_config=quantization, low_cpu_mem_usage=True)
    if bool(getattr(model, "is_loaded_in_4bit", False)) != placement["quantized"]:
        raise RuntimeError("Loaded correction base quantization differs from its configuration")
    root_hook = getattr(model, "_hf_hook", None)
    if root_hook is not None:
        # The root hook must not move loss labels or the normalizer to GPU 0,
        # or copy full-vocabulary logits away from the output head on GPU 1.
        skip = root_hook.skip_keys or []
        if isinstance(skip, str):
            skip = [skip]
        root_hook.skip_keys = list(dict.fromkeys([*skip, "labels", "num_items_in_batch", "logits"]))
    expected_devices = set(placement["device_map"].values())
    if any(str(parameter.device) not in expected_devices for parameter in model.parameters()):
        raise RuntimeError("Correction weights do not match the explicit map; offload is forbidden")
    # Trainer detects these flags and avoids both .to(cuda:0) and DataParallel.
    model.is_parallelizable = True
    model.model_parallel = True
    return model


def load_corrector(adapter, device="cuda"):
    """Reload the saved base placement, tokenizer, adapter and exact inference prompt."""
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoTokenizer
    adapter = Path(adapter)
    settings = json.loads((adapter / "loading.json").read_text(encoding="utf-8"))
    inference = json.loads((adapter / "inference.json").read_text(encoding="utf-8"))
    base = _load_base(settings, device)
    if settings.get("load_in_4bit"):
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
    model = PeftModel.from_pretrained(base, adapter, is_trainable=False)
    model.is_parallelizable = True
    model.model_parallel = True
    tokenizer = AutoTokenizer.from_pretrained(adapter)
    model.eval()
    return model, tokenizer, inference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data")
    parser.add_argument("--classifier-run", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--device", choices=("cuda", "cpu"))
    args = parser.parse_args()
    from .two_phase import load_stage_config, resolve_data, stage_output
    config = load_stage_config(args.config, "correction")
    device = args.device or config["run"]["device"]
    config["run"]["device"] = device
    seed = config["run"]["seed"]
    fraction = config["data"]["validation_fraction"]
    data_path = resolve_data(config, args.data)
    config["data"]["path"] = str(data_path)
    output = stage_output(config, args.output_dir)
    classifier_run = Path(args.classifier_run).expanduser().resolve()
    artifact = json.loads((classifier_run / "hints.json").read_text(encoding="utf-8"))
    classifier_train = json.loads((classifier_run / "train.json").read_text(encoding="utf-8"))
    train, validation, provenance = load_training_split(data_path, fraction, seed)
    train_hints, validation_hints = validate_hints(
        artifact, train, validation, provenance, classifier_train, data_path, seed, fraction)
    # Limits are diagnostics only, after validating the entire connected pipeline.
    train = balanced_limit(train, config["data"].get("train_limit"), seed)
    validation = balanced_limit(validation, config["data"].get("validation_limit"), seed + 1)
    if not train or not validation:
        raise ValueError("Correction training and validation must be nonempty")
    training, settings = config["training"], config["model"]
    if tuple(config["evaluation"]["selection_metrics"]) != SELECTION_METRICS:
        raise ValueError("Correction selects error-only GLEU+, then error-only exact, then clean preservation")
    if training["args"].get("gradient_checkpointing") and training["args"].get("gradient_checkpointing_kwargs", {}).get("use_reentrant") is not False:
        raise ValueError("Correction gradient checkpointing must explicitly use_reentrant=false")
    placement = placement_preflight(settings, device)
    settings["revision"] = placement["resolved_revision"]
    import torch
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed
    from .metrics import experiment_report
    from .modeling import ResponseCollator
    dtype = _dtype(settings, device)
    settings["compute_dtype"] = str(dtype).removeprefix("torch.")
    set_seed(seed)
    inference = {**config["inference"], "generation": config["generation"],
                 "max_length": training["max_length"]}
    tokenizer = AutoTokenizer.from_pretrained(settings["name"], revision=settings["revision"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    encoded = [encode_record(tokenizer, row, train_hints[_id_key(row["id"])], training["max_length"], inference) for row in train]
    validation_lengths = [len(encode_record(tokenizer, row, validation_hints[_id_key(row["id"])], training["max_length"], inference)["input_ids"]) for row in validation]
    accumulation = training["effective_batch_size"] // training["args"]["per_device_train_batch_size"]
    training_args = TrainingArguments(
        **training["args"], output_dir=str(output), use_cpu=device == "cpu",
        gradient_accumulation_steps=accumulation, fp16=device == "cuda" and dtype == torch.float16,
        bf16=device == "cuda" and dtype == torch.bfloat16, save_strategy="no", eval_strategy="no",
        report_to="none", label_names=["labels"], seed=seed, data_seed=seed,
        logging_nan_inf_filter=False)
    # Single-map diagnostics must not replicate weights onto other visible GPUs.
    training_args._n_gpu = 1 if device == "cuda" else 0
    lora = LoraConfig(**config["lora"], task_type=TaskType.CAUSAL_LM, revision=settings["revision"])
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    write_json(output / "train.json", train)
    write_json(output / "validation.json", validation)
    write_json(output / "inference.json", inference)
    write_json(output / "placement.json", placement)
    model = _load_base(settings, device, placement)
    selected_hints = [validation_hints[_id_key(row["id"])] for row in validation]

    def predict(current):
        return correct(current, tokenizer, validation, selected_hints, inference)

    def save_adapter(current, path):
        current.save_pretrained(path)
        tokenizer.save_pretrained(path)
        write_json(path / "loading.json", settings)
        write_json(path / "inference.json", inference)

    baseline_predictions = predict(model)
    baseline_metrics = experiment_report(validation, baseline_predictions)
    write_json(output / "baseline_predictions.json", comparison_rows(validation, baseline_predictions))
    write_json(output / "baseline_metrics.json", baseline_metrics)
    model.config.use_cache = False
    if placement["quantized"]:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_args.gradient_checkpointing,
            gradient_checkpointing_kwargs=training_args.gradient_checkpointing_kwargs)
    model = get_peft_model(model, lora, autocast_adapter_dtype=True)
    model.is_parallelizable = True
    model.model_parallel = True
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
    model.print_trainable_parameters()
    history, best = [], {"key": None, "epoch": None}
    adapter_path = output / "adapter"

    class CorrectionTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            # Decoder/head dispatch is explicit; loss labels and its normalizer
            # belong on the output head, even though input IDs start on GPU 0.
            inputs = dict(inputs)
            head_device = model.get_output_embeddings().weight.device
            inputs["labels"] = inputs["labels"].to(head_device)
            if torch.is_tensor(num_items_in_batch):
                num_items_in_batch = num_items_in_batch.to(head_device)
            loss = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
            # Trainer accumulates its detached logging loss on args.device.
            if return_outputs:
                return loss[0].to(self.args.device), loss[1]
            return loss.to(self.args.device)

    class EpochEvaluation(TrainerCallback):
        def on_epoch_end(self, args, state, control, model=None, **kwargs):
            if not all(torch.isfinite(p).all().item() for p in model.parameters() if p.requires_grad):
                raise RuntimeError("Nonfinite correction adapter parameters")
            if any(not math.isfinite(row["grad_norm"]) for row in state.log_history if "grad_norm" in row):
                raise RuntimeError("Nonfinite correction gradients")
            if any(not math.isfinite(row["loss"]) for row in state.log_history if "loss" in row):
                raise RuntimeError("Nonfinite correction training loss")
            predictions = predict(model)
            metrics = experiment_report(validation, predictions)
            epoch_dir = output / f"epoch-{len(history) + 1}"
            epoch_dir.mkdir()
            write_json(epoch_dir / "predictions.json", comparison_rows(validation, predictions))
            write_json(epoch_dir / "metrics.json", metrics)
            if config["evaluation"].get("save_epoch_adapters", False):
                save_adapter(model, epoch_dir / "adapter")
            key = selection_key(metrics, SELECTION_METRICS)
            history.append({"epoch": state.epoch, "optimizer_steps": state.global_step,
                            "metrics": metrics, "selection_values": key})
            write_json(output / "epoch_history.json", history)
            if best["key"] is None or key > best["key"]:
                best.update(key=key, epoch=state.epoch)
                save_adapter(model, adapter_path)
            print(json.dumps({"epoch": state.epoch, "selection_values": key, "best_epoch": best["epoch"]}), flush=True)
            model.train()
            return control

    trainer = CorrectionTrainer(model=model, args=training_args, train_dataset=encoded,
                                data_collator=ResponseCollator(tokenizer.pad_token_id), callbacks=[EpochEvaluation()])
    if (trainer.accelerator.num_processes != 1 or trainer.args.n_gpu > 1
            or (settings["device_map"] == "balanced" and not trainer.is_model_parallel)):
        raise RuntimeError("Correction must train one process without DataParallel/DDP")
    if training_args.fp16:
        trainer.accelerator.scaler = torch.amp.GradScaler("cuda", init_scale=training["fp16_initial_scale"])
    result = trainer.train()
    if not math.isfinite(result.training_loss) or best["epoch"] is None:
        raise RuntimeError("Correction did not complete a finite evaluated training epoch")
    write_json(output / "trainer_history.json", trainer.state.log_history)
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_parameters = sum(p.numel() for p in model.parameters())
    peak_memory = {f"cuda:{i}": torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())} if device == "cuda" else {}
    trainer.model = trainer.model_wrapped = trainer.optimizer = trainer.lr_scheduler = None
    trainer.accelerator.free_memory()
    del trainer, model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    reloaded, reloaded_tokenizer, saved_inference = load_corrector(adapter_path, device)
    predictions = correct(reloaded, reloaded_tokenizer, validation, selected_hints, saved_inference)
    metrics = experiment_report(validation, predictions)
    write_json(output / "predictions.json", comparison_rows(validation, predictions))
    write_json(output / "metrics.json", metrics)
    manifest = {
        "config": config, "resolved_model_revision": settings["revision"], "model": settings,
        "adapter_path": str(adapter_path), "output_dir": str(output), "seed": seed,
        "data_path": str(data_path), "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "classifier_run": str(classifier_run),
        "hints_sha256": hashlib.sha256((classifier_run / "hints.json").read_bytes()).hexdigest(),
        "data_provenance": provenance, "train_ids": [row["id"] for row in train],
        "validation_ids": [row["id"] for row in validation], "inference": inference,
        "device": device, "placement": placement, "device_map": placement["device_map"],
        "quantized": placement["quantized"], "load_in_4bit": placement["quantized"], "cpu_offload": False,
        "parallelism": "single-process explicit device map", "world_size": 1,
        "compute_dtype": str(dtype), "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters, "training_args": training_args.to_dict(),
        "effective_batch_size": training["effective_batch_size"], "gradient_accumulation_steps": accumulation,
        "fp16_initial_scale": training["fp16_initial_scale"] if training_args.fp16 else None,
        "train_max_tokens": max(len(row["input_ids"]) for row in encoded),
        "validation_max_tokens": max(validation_lengths), "optimizer_steps": result.global_step,
        "training_loss": result.training_loss, "best_epoch": best["epoch"],
        "selection_metrics": list(SELECTION_METRICS),
        "selection_rule": "highest error-only GLEU+, then error-only exact, then clean preservation; ties keep earlier epoch",
        "saved_adapter_reloaded_for_evaluation": True, "peak_allocated_vram_bytes": peak_memory,
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes")},
        "devices": [torch.cuda.get_device_name(i) for i in range(2 if settings["device_map"] == "balanced" else 1)] if device == "cuda" else ["cpu"],
    }
    write_json(output / "run_manifest.json", manifest)
    print(json.dumps({"output_dir": str(output), "adapter_path": str(adapter_path),
                      "best_epoch": best["epoch"], "error_gleu_plus": metrics["corrections"]["error_only"]["gleu_plus"]}), flush=True)


if __name__ == "__main__":
    main()
