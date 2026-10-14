"""Train response-only LoRA/QLoRA correction and reload the saved adapter."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
from datetime import timedelta

from .data import _id_key, _source_key, _valid_label, load_training_split
from .runtime import CompleteEpochMixin, balanced_limit, load_config, resolve_data, selection_key, stage_output, write_json
from .prediction_export import comparison_rows
from .correction_edits import EDIT_PROMPT, SENTENCE_NO_HINT_PROMPT, encode_edits, parse_edits
from .tracking import records_identity, tracked_entrypoint


def prepare_training_inference(config):
    """Choose an explicit representation only for fresh training, never rewrite saved prompts."""
    inference = {**config["inference"], "generation": dict(config["generation"]),
                 "max_length": config["training"]["max_length"]}
    inference.setdefault("output_format", "sentence")
    inference.setdefault("hint_mode", "predicted")
    if inference["output_format"] == "edits":
        inference["prompt"] = EDIT_PROMPT
    elif inference["hint_mode"] == "none":
        inference["prompt"] = SENTENCE_NO_HINT_PROMPT
    validate_inference(inference)
    return inference


def validate_inference(inference):
    if inference.get("output_format", "sentence") not in ("sentence", "edits"):
        raise ValueError("output_format must be sentence or edits")
    if inference.get("hint_mode", "predicted") not in ("predicted", "none"):
        raise ValueError("hint_mode must be predicted or none")
    if inference.get("user_role", "user") not in ("user", "question"):
        raise ValueError("user_role must match the base tokenizer's user or question role")
    if not isinstance(inference.get("prompt"), str) or not inference["prompt"]:
        raise ValueError("Correction inference requires a saved prompt")
    if inference.get("hint_mode", "predicted") == "predicted":
        descriptions = inference.get("hint_descriptions")
        if not isinstance(descriptions, list) or len(descriptions) != 5 or any(not isinstance(x, str) for x in descriptions):
            raise ValueError("Predicted hints require five description strings")
    if inference.get("reranking") is not None:
        from .correction_reranking import reranking_config
        reranking_config(inference["reranking"])


def validate_initial_adapter(adapter, config, inference=None):
    """Validate a continuation before weights load; optimizer state is intentionally new."""
    adapter = Path(adapter).expanduser().resolve()
    for filename in ("loading.json", "inference.json", "adapter_config.json"):
        if not (adapter / filename).is_file():
            raise ValueError(f"Initial correction adapter is missing {filename}")
    if not any((adapter / filename).is_file() for filename in ("adapter_model.safetensors", "adapter_model.bin")):
        raise ValueError("Initial correction adapter has no saved weights")
    loading = json.loads((adapter / "loading.json").read_text(encoding="utf-8"))
    saved = json.loads((adapter / "inference.json").read_text(encoding="utf-8"))
    peft = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    requested = inference or prepare_training_inference(config)
    validate_inference(saved)
    if peft.get("task_type") != "CAUSAL_LM" or peft.get("peft_type") != "LORA":
        raise ValueError("Initial adapter must be a PEFT LoRA CAUSAL_LM correction adapter")
    for field in ("name", "revision"):
        if loading.get(field) != config["model"].get(field):
            raise ValueError(f"Initial adapter base {field} differs")
    if peft.get("base_model_name_or_path") != loading["name"] or peft.get("revision") != loading["revision"]:
        raise ValueError("Initial PEFT base model/revision disagrees with loading.json")
    structural_defaults = {"use_dora": False, "use_rslora": False, "fan_in_fan_out": False,
                           "rank_pattern": {}, "alpha_pattern": {}, "layers_to_transform": None,
                           "layers_pattern": None, "modules_to_save": None, "lora_bias": False}
    expected_lora = {**structural_defaults, **config["lora"]}
    for field, expected in expected_lora.items():
        actual = peft.get(field, structural_defaults.get(field))
        if field == "target_modules" and isinstance(expected, list) and isinstance(actual, list):
            actual, expected = sorted(actual), sorted(expected)
        if actual != expected:
            raise ValueError(f"Initial adapter LoRA {field} differs")
    for field, default in (("output_format", "sentence"), ("hint_mode", "predicted")):
        if saved.get(field, default) != requested.get(field, default):
            raise ValueError(f"Initial adapter {field} contract differs; no silent migration")
    if saved.get("user_role", "user") != requested.get("user_role", "user"):
        raise ValueError("Initial adapter user_role interface differs")
    for field in ("prompt", "hint_descriptions"):
        if saved.get(field) != requested.get(field):
            raise ValueError(f"Initial adapter {field} interface differs")
    return {"path": str(adapter), "loading": loading, "inference": saved,
            "continuation": "trainable adapter weights with a fresh optimizer, not exact resume"}

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
    """Check OOF declarations against connected source, parent and contrast families."""
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
    retained = {_id_key(value) for value in mixed["retained_original_train_ids"]}
    quarantined = {_id_key(value) for value in mixed["quarantined_ids"]}
    synthetic = {_id_key(value) for value in mixed["selected_synthetic_ids"]}
    if (not retained <= original_keys or quarantined != original_keys - retained
            or {_id_key(value) for value in mixed["original_train_ids"]} != original_keys
            or synthetic & original_keys or retained | synthetic != set(classifier_rows)):
        raise ValueError("Classifier train.json differs from declared mixed partition")
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
    families = {}
    for metadata in provenance.get("provenance", {}).get("train_rows", []) + mixed["train_rows"]:
        key = _id_key(metadata["id"])
        authored = metadata.get("authored_provenance", {})
        for field in ("family_id", "template_id"):
            value = authored.get(field)
            if value:
                family = (field, value)
                if family in families:
                    union(key, families[family])
                else:
                    families[family] = key
        for parent in metadata.get("parent_ids", []):
            parent_key = _id_key(parent)
            if parent_key not in original_keys or parent_key in heldout:
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
            raise ValueError("OOF training leaks a held-out source, parent or contrast/template family")
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
            raise ValueError("Connected source/parent/family group is split across OOF folds")
        group_folds[group] = hint["fold"]
    return train_hints, validation_hints


def _prompt_ids(tokenizer, source, label, inference):
    mode = inference.get("hint_mode", "predicted")
    if mode == "none":
        content = f"Câu gốc:\n{source}"
        prompt = EDIT_PROMPT if inference.get("output_format", "sentence") == "edits" else SENTENCE_NO_HINT_PROMPT
    elif mode == "predicted":
        descriptions = inference["hint_descriptions"]
        content = f"Gợi ý nhãn dự đoán: {label} — {descriptions[label]}.\nCâu gốc:\n{source}"
        prompt = inference["prompt"]
    else:
        raise ValueError("hint_mode must be predicted or none")
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": prompt},
         {"role": inference.get("user_role", "user"), "content": content}],
        tokenize=True, add_generation_prompt=True, return_dict=False)


def encode_record(tokenizer, record, hint, max_length, inference):
    prompt = _prompt_ids(tokenizer, record["source"], hint["label"], inference)
    answer_value = (encode_edits(record["source"], record["target"])
                    if inference.get("output_format", "sentence") == "edits"
                    else {"corrected_sentence": record["target"]})
    answer = json.dumps(answer_value, ensure_ascii=False)
    if tokenizer.eos_token_id is None:
        raise ValueError("Correction tokenizer must have an EOS token")
    completion = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
    ids = prompt + completion
    if len(ids) > max_length:
        raise ValueError(f"ID {record['id']!r}: {len(ids)} tokens exceeds max_length={max_length}; targets are never truncated")
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": [-100] * len(prompt) + completion}


def _parse_correction(text, source=None, output_format="sentence"):
    if output_format == "edits":
        if not isinstance(source, str):
            raise ValueError("Edit parsing requires the original source")
        return parse_edits(source, text)
    if output_format != "sentence":
        raise ValueError("output_format must be sentence or edits")
    try:
        value = json.loads(text)
        if (not isinstance(value, dict) or set(value) != {"corrected_sentence"}
                or not isinstance(value["corrected_sentence"], str) or not value["corrected_sentence"].strip()):
            raise ValueError("Expected JSON with only a nonblank corrected_sentence string")
        return value["corrected_sentence"], None
    except (ValueError, TypeError, KeyError) as error:
        return "", str(error)


def correct(model, tokenizer, records, hints, inference):
    """Generate every row, including class 0; optional ranking never gates on labels."""
    import torch
    validate_inference(inference)
    indexed = _hint_index(records, hints)
    generation = {"max_new_tokens": 192, "do_sample": False, "use_cache": True,
                  "pad_token_id": tokenizer.pad_token_id,
                  "eos_token_id": model.generation_config.eos_token_id or tokenizer.eos_token_id}
    generation.update(inference["generation"])
    reranker, ranking = None, None
    if (inference.get("reranking") or {}).get("enabled", False):
        from .correction_reranking import FrozenReranker, reranking_config
        ranking = reranking_config(inference["reranking"])
        generation.update(do_sample=False, num_beams=ranking["num_candidates"],
                          num_return_sequences=ranking["num_candidates"])
        reranker = FrozenReranker(model, tokenizer, ranking, inference["max_length"],
                                 user_role=inference.get("user_role", "user"))
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
            candidates = []
            for sequence in sequences:
                text = tokenizer.decode(sequence[len(prompt):], skip_special_tokens=True).strip()
                target, error = _parse_correction(text, record["source"], inference.get("output_format", "sentence"))
                candidates.append({"target": target, "raw_generation": text, "parse_error": error,
                                   "origin": "beam" if ranking else "generation"})
            # Preserve every valid greedy prediction. Recover only malformed full-
            # sentence JSON with the same bounded method used for the public run.
            if not ranking and candidates[0]["parse_error"] and inference.get("output_format", "sentence") == "sentence":
                from lmformatenforcer import JsonSchemaParser
                from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
                schema = {"type": "object", "properties": {"corrected_sentence": {"type": "string", "minLength": 1}},
                          "required": ["corrected_sentence"], "additionalProperties": False}
                constrained = {**generation, "do_sample": False, "num_beams": 4, "num_return_sequences": 1,
                               "length_penalty": 0.0, "early_stopping": True,
                               "max_new_tokens": max(generation["max_new_tokens"], inference["max_length"] + 64),
                               "prefix_allowed_tokens_fn": build_transformers_prefix_allowed_tokens_fn(tokenizer, JsonSchemaParser(schema))}
                retry = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), **constrained)
                sequence = retry.sequences[0] if hasattr(retry, "sequences") else retry[0]
                text = tokenizer.decode(sequence[len(prompt):], skip_special_tokens=True).strip()
                target, error = _parse_correction(text)
                candidates[0] = {"target": target, "raw_generation": text, "parse_error": error,
                                 "origin": "schema_constrained_retry"}
            selected = 0
            if reranker is not None:
                candidates.append({"target": record["source"], "raw_generation": None,
                                   "parse_error": None, "origin": "unchanged_source"})
                selected = reranker.score(record["source"], candidates)
            chosen = candidates[selected]
            prediction = {"id": record["id"], "label": hint["label"], "target": chosen["target"],
                          "raw_generation": chosen["raw_generation"], "parse_error": chosen["parse_error"]}
            if chosen["origin"] == "schema_constrained_retry":
                prediction["regenerated"] = True
            if ranking:
                prediction.update(candidates=candidates, selected_candidate=selected, reranking=ranking)
            predictions.append(prediction)
    return predictions


def _dtype(settings, device):
    import torch
    value = settings.get("compute_dtype", "auto")
    if value == "auto":
        if device == "cpu":
            return torch.float32
        supported = []
        indices = range(2) if settings.get("device_map", "single") == "balanced" else [int(os.environ.get("LOCAL_RANK", "0"))]
        for index in indices:
            with torch.cuda.device(index):
                from .runtime import native_bf16_supported
                supported.append(native_bf16_supported())
        return torch.bfloat16 if all(supported) else torch.float16
    if value not in ("float16", "bfloat16", "float32"):
        raise ValueError("compute_dtype must be auto, float16, bfloat16, or float32")
    if value == "bfloat16" and device == "cuda":
        from .runtime import native_bf16_supported
        with torch.cuda.device(int(os.environ.get("LOCAL_RANK", "0"))):
            if not native_bf16_supported():
                return torch.float16
    return getattr(torch, value)


def placement_preflight(settings, device="cuda", *, local_rank=None, training=True, task="correction"):
    """Budget native decoder weights and build an explicit two-GPU layer map."""
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSequenceClassification
    factory = AutoModelForSequenceClassification if task == "classification" else AutoModelForCausalLM
    head_name = "score" if task == "classification" else "lm_head"
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0")) if local_rank is None else local_rank
    quantized = settings.get("load_in_4bit", False)
    if settings.get("load_in_8bit") or settings.get("quantization_config") is not None:
        raise ValueError("Use load_in_4bit for NF4 QLoRA; 8-bit/prequantized loading is unsupported")
    if quantized and device != "cuda":
        raise ValueError("Correction QLoRA requires CUDA")
    mode = settings.get("device_map", "single")
    if mode not in ("balanced", "single"):
        raise ValueError("device_map must be balanced (explicit two-GPU map) or single")
    if world_size > 1 and mode != "single":
        raise ValueError("Correction DDP requires a complete single-device replica on each rank")
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
    # Resolve lazy model imports before the meta-device context: importing
    # torch attention/Dynamo inside torch.device("meta") can recurse during init.
    factory._model_mapping[type(config)]
    with init_empty_weights(include_buffers=True):
        if task == "classification":
            config.num_labels = 5
        skeleton = factory.from_config(config, torch_dtype=dtype)
    head = getattr(skeleton, head_name)

    def weight_bytes(module):
        if not quantized:
            return sum(p.numel() * p.element_size() for p in module.parameters())
        # NF4 packed weights plus metadata; k-bit preparation promotes other weights to FP32.
        total = 0
        for name, parameter in module.named_parameters():
            parent = module.get_submodule(name.rsplit(".", 1)[0]) if "." in name else module
            packed = isinstance(parent, torch.nn.Linear) and parent is not head and name.endswith("weight")
            total += math.ceil(parameter.numel() * (0.625 if packed else 4))
        return total
    if mode == "single":
        mapping = {"": f"cuda:{local_rank}" if device == "cuda" else "cpu"}
    else:
        eligible = ("llama", "qwen2", "qwen3") if task == "classification" else ("qwen2", "qwen3")
        if config.model_type not in eligible or (task == "correction" and config.tie_word_embeddings):
            raise ValueError("Balanced placement requires a supported decoder architecture with a separate head")
        layer_sizes = [weight_bytes(layer) for layer in skeleton.model.layers]
        embedding = weight_bytes(skeleton.model.embed_tokens)
        ending = weight_bytes(skeleton.model.norm) + weight_bytes(head)
        split = min(range(1, len(layer_sizes)),
                    key=lambda index: abs(embedding + sum(layer_sizes[:index]) - ending - sum(layer_sizes[index:])))
        mapping = {"model.embed_tokens": "cuda:0", "model.rotary_emb": "cuda:0",
                   "model.norm": "cuda:1", head_name: "cuda:1"}
        mapping.update({f"model.layers.{index}": f"cuda:{int(index >= split)}" for index in range(len(layer_sizes))})
    weights = {}
    for name, parameter in skeleton.named_parameters():
        prefixes = [prefix for prefix in mapping if not prefix or name == prefix or name.startswith(prefix + ".")]
        if not prefixes:
            raise ValueError(f"Explicit device map does not cover {name}")
        destination = mapping[max(prefixes, key=len)]
        parent = skeleton.get_submodule(name.rsplit(".", 1)[0]) if "." in name else skeleton
        packed = quantized and isinstance(parent, torch.nn.Linear) and parent is not head and name.endswith("weight")
        size = math.ceil(parameter.numel() * (0.625 if packed else 4)) if quantized else parameter.numel() * parameter.element_size()
        weights[destination] = weights.get(destination, 0) + size
    del skeleton
    # Training also needs optimizer states and loss/activation workspaces.
    # Reloaded inference preserves FP32 prepared weights but needs only its
    # generation KV cache and CUDA workspaces, plus the weight margin below.
    reserve = (4 if training else 1) * 1024 ** 3
    budgets = {}
    for destination, size in weights.items():
        required = math.ceil(size * 1.10) + reserve
        free = None
        if destination.startswith("cuda"):
            index = int(destination.split(":")[1])
            free, total = torch.cuda.mem_get_info(index)
            if required > free:
                purpose = "training" if training else "inference"
                raise RuntimeError(f"{destination} has {free / 1024**3:.2f} GiB free; estimated weights plus conservative {purpose} reserve need {required / 1024**3:.2f} GiB. Use sufficient GPU memory or balanced placement; no offload.")
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
    # Single-device replicas must remain eligible for Trainer's DDP wrapping.
    model.is_parallelizable = placement["mode"] == "balanced"
    model.model_parallel = placement["mode"] == "balanced"
    return model


def load_corrector(adapter, device="cuda"):
    """Reload the saved base placement, tokenizer, adapter and exact inference prompt."""
    adapter = Path(adapter)
    settings = json.loads((adapter / "loading.json").read_text(encoding="utf-8"))
    inference = json.loads((adapter / "inference.json").read_text(encoding="utf-8"))
    validate_inference(inference)
    placement = placement_preflight(settings, device, training=False)
    return _attach_corrector(_load_base(settings, device, placement), adapter, settings, inference)


def _attach_corrector(base, adapter, settings, inference):
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoTokenizer
    if settings.get("load_in_4bit"):
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
    model = PeftModel.from_pretrained(
        base, adapter, is_trainable=False,
        torch_device=str(base.get_input_embeddings().weight.device),
        device_map=getattr(base, "hf_device_map", None))
    model.is_parallelizable = settings.get("device_map", "single") == "balanced"
    model.model_parallel = settings.get("device_map", "single") == "balanced"
    tokenizer = AutoTokenizer.from_pretrained(adapter)
    model.eval()
    return model, tokenizer, inference


def _reload_selected_corrector(model, adapter):
    """Replace training LoRA with saved selected weights; never merge or load a second base."""
    adapter = Path(adapter)
    settings = json.loads((adapter / "loading.json").read_text(encoding="utf-8"))
    inference = json.loads((adapter / "inference.json").read_text(encoding="utf-8"))
    validate_inference(inference)
    base = model.unload()
    # PEFT unload removes the layers but leaves the old adapter registry behind.
    del base.peft_config
    base.gradient_checkpointing_disable()
    if hasattr(base, "_require_grads_hook"):
        base.disable_input_require_grads()
    return _attach_corrector(base, adapter, settings, inference)


@tracked_entrypoint
def main(*, tracking):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data")
    parser.add_argument("--classifier-run", required=True)
    parser.add_argument("--initial-adapter", help="Compatible correction LoRA weights; starts a fresh optimizer")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", choices=("cuda", "cpu"))
    args = parser.parse_args()
    config = load_config(args.config, "correction")
    device = args.device or config["run"]["device"]
    config["run"]["device"] = device
    inference = prepare_training_inference(config)
    initial = validate_initial_adapter(args.initial_adapter, config, inference) if args.initial_adapter else None
    import torch
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    expected_world = config["run"].get("world_size", 1)
    if world_size != expected_world:
        raise ValueError(f"run.world_size={expected_world} requires a matching torchrun launch, got {world_size}")
    if device == "cuda":
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if world_size > 1:
        torch.distributed.init_process_group(backend="nccl" if device == "cuda" else "gloo",
                                             timeout=timedelta(hours=2))

    def barrier():
        if world_size > 1:
            torch.distributed.barrier()

    seed = config["run"]["seed"]
    fraction = config["data"]["validation_fraction"]
    data_path = resolve_data(config, args.data)
    config["data"]["path"] = str(data_path)
    shared_output = [str(stage_output(config, args.output_dir)) if rank == 0 else None]
    if world_size > 1:
        torch.distributed.broadcast_object_list(shared_output, src=0)
    output = Path(shared_output[0])
    from .run_reuse import resolve_run, validate_reuse_data, validation_predictions
    classifier_run = resolve_run(args.classifier_run, stage="classification")
    train, validation, provenance = load_training_split(data_path, fraction, seed)
    hints_sha256 = None
    validation_predictions_sha256 = None
    if inference["hint_mode"] == "predicted":
        artifact = json.loads((classifier_run / "hints.json").read_text(encoding="utf-8"))
        classifier_train = json.loads((classifier_run / "train.json").read_text(encoding="utf-8"))
        train_hints, validation_hints = validate_hints(
            artifact, train, validation, provenance, classifier_train, data_path, seed, fraction)
        hints_sha256 = hashlib.sha256((classifier_run / "hints.json").read_bytes()).hexdigest()
    else:
        validate_reuse_data(classifier_run, config, data_path)
        # IDs/placeholder scores only: no label is included in the training prompt.
        train_hints = {_id_key(row["id"]): {"id": row["id"], "label": 0,
                       "probabilities": [1, 0, 0, 0, 0]} for row in train}
        validation_hints = validation_predictions(classifier_run, validation)
        validation_predictions_sha256 = hashlib.sha256((classifier_run / "predictions.json").read_bytes()).hexdigest()
    tracking.start(config, output, "correction", group=output.parent.name,
                   metadata={"train": records_identity(train), "validation": records_identity(validation),
                             "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
                             "hint_mode": inference["hint_mode"], "hints_sha256": hints_sha256})
    tracking.use_model(classifier_run / "classifier", "classifier")
    if initial:
        tracking.use_model(initial["path"], "corrector")
    # Limits are diagnostics only, after validating the entire connected pipeline.
    train = balanced_limit(train, config["data"].get("train_limit"), seed)
    validation = balanced_limit(validation, config["data"].get("validation_limit"), seed + 1)
    final_fit = provenance.get("format") == "vgec-final-training-v1"
    if not train or (not validation and not final_fit):
        raise ValueError("Correction training and validation must be nonempty except verified final fits")
    if final_fit and (validation or inference["hint_mode"] != "none"):
        raise ValueError("Final fits require no validation and no hints")
    training, settings = config["training"], config["model"]
    if tuple(config["evaluation"]["selection_metrics"]) != SELECTION_METRICS:
        raise ValueError("Correction selects error-only GLEU+, then error-only exact, then clean preservation")
    if training["args"].get("gradient_checkpointing") and training["args"].get("gradient_checkpointing_kwargs", {}).get("use_reentrant") is not False:
        raise ValueError("Correction gradient checkpointing must explicitly use_reentrant=false")
    placement = placement_preflight(settings, device)
    settings["revision"] = placement["resolved_revision"]
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed
    from .metrics import experiment_report
    from .modeling import ResponseCollator
    dtype = _dtype(settings, device)
    settings["compute_dtype"] = str(dtype).removeprefix("torch.")
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(initial["path"] if initial else settings["name"],
                                             **({} if initial else {"revision": settings["revision"]}))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    encoded = [encode_record(tokenizer, row, train_hints[_id_key(row["id"])], training["max_length"], inference) for row in train]
    validation_lengths = [len(encode_record(tokenizer, row, validation_hints[_id_key(row["id"])], training["max_length"], inference)["input_ids"]) for row in validation]
    trainer_args = dict(training["args"])
    microbatch = trainer_args["per_device_train_batch_size"]
    global_microbatch = microbatch * world_size
    if training["effective_batch_size"] % global_microbatch or training["effective_batch_size"] < global_microbatch:
        raise ValueError("effective_batch_size must be a positive multiple of microbatch * world_size")
    accumulation = training["effective_batch_size"] // global_microbatch
    training_args = TrainingArguments(
        **trainer_args, output_dir=str(output), use_cpu=device == "cpu",
        gradient_accumulation_steps=accumulation, fp16=device == "cuda" and dtype == torch.float16,
        bf16=device == "cuda" and dtype == torch.bfloat16, save_strategy="no", eval_strategy="no",
        report_to="none", label_names=["labels"], seed=seed, data_seed=seed,
        logging_nan_inf_filter=False, ddp_find_unused_parameters=False, ddp_broadcast_buffers=False,
        ddp_timeout=7200)
    # Keep one device per process; multiple ranks use DDP, never DataParallel.
    training_args._n_gpu = 1 if device == "cuda" else 0
    lora = LoraConfig(**config["lora"], task_type=TaskType.CAUSAL_LM, revision=settings["revision"])
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "config.json", config)
        write_json(output / "train.json", train)
        write_json(output / "validation.json", validation)
        write_json(output / "inference.json", inference)
        write_json(output / "placement.json", placement)
        write_json(output / "validation_hints.json", [validation_hints[_id_key(row["id"])] for row in validation])
    barrier()
    model = _load_base(settings, device, placement)
    selected_hints = [validation_hints[_id_key(row["id"])] for row in validation]

    def predict(current):
        return correct(current, tokenizer, validation, selected_hints, inference)

    def save_adapter(current, path):
        current.save_pretrained(path)
        tokenizer.save_pretrained(path)
        write_json(path / "loading.json", settings)
        write_json(path / "inference.json", inference)

    if rank == 0 and not final_fit:
        baseline_predictions = predict(model)
        baseline_metrics = experiment_report(validation, baseline_predictions)
        write_json(output / "baseline_predictions.json", comparison_rows(validation, baseline_predictions))
        write_json(output / "baseline_metrics.json", baseline_metrics)
        tracking.evaluation(baseline_metrics, "baseline")
    barrier()
    model.config.use_cache = False
    if placement["quantized"]:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_args.gradient_checkpointing,
            gradient_checkpointing_kwargs=training_args.gradient_checkpointing_kwargs)
    model = (PeftModel.from_pretrained(model, initial["path"], is_trainable=True, autocast_adapter_dtype=True)
             if initial else get_peft_model(model, lora, autocast_adapter_dtype=True))
    from .runtime import promote_trainable_parameters
    promote_trainable_parameters(model)
    model.is_parallelizable = settings["device_map"] == "balanced"
    model.model_parallel = settings["device_map"] == "balanced"
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
    model.print_trainable_parameters()
    history, best = [], {"key": None, "epoch": None}
    adapter_path = output / "adapter"

    class CorrectionTrainer(CompleteEpochMixin, Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            # Decoder/head dispatch is explicit; loss labels and its normalizer
            # belong on the output head, even though input IDs start on GPU 0.
            inputs = dict(inputs)
            unwrapped = self.accelerator.unwrap_model(model)
            head_device = unwrapped.get_output_embeddings().weight.device
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
            barrier()
            if final_fit:
                best.update(epoch=state.epoch)
                if rank == 0:
                    history.append({"epoch": state.epoch, "optimizer_steps": state.global_step,
                                    "evaluation_status": "unavailable_full_training"})
                    write_json(output / "epoch_history.json", history)
                barrier()
                model.train()
                return control
            if rank == 0:
                current = trainer.accelerator.unwrap_model(model)
                predictions = predict(current)
                metrics = experiment_report(validation, predictions)
                epoch_dir = output / f"epoch-{len(history) + 1}"
                epoch_dir.mkdir()
                write_json(epoch_dir / "predictions.json", comparison_rows(validation, predictions))
                write_json(epoch_dir / "metrics.json", metrics)
                tracking.evaluation(metrics, "validation", optimizer_step=state.global_step, epoch=state.epoch)
                if config["evaluation"].get("save_epoch_adapters", False):
                    save_adapter(current, epoch_dir / "adapter")
                    tracking.model(epoch_dir / "adapter", f"corrector-epoch-{len(history) + 1}",
                                   metadata={"epoch": state.epoch}, aliases=["latest"])
                key = selection_key(metrics, SELECTION_METRICS)
                history.append({"epoch": state.epoch, "optimizer_steps": state.global_step,
                                "metrics": metrics, "selection_values": key})
                write_json(output / "epoch_history.json", history)
                if best["key"] is None or key > best["key"]:
                    best.update(key=key, epoch=state.epoch)
                    save_adapter(current, adapter_path)
                print(json.dumps({"epoch": state.epoch, "selection_values": key, "best_epoch": best["epoch"],
                                  "combined_score": metrics["combined_score"],
                                  "combined_score_metadata": metrics["combined_score_metadata"]}), flush=True)
                tracking.summary({"best_epoch": best["epoch"]})
            if world_size > 1:
                shared_best = [best if rank == 0 else None]
                torch.distributed.broadcast_object_list(shared_best, src=0)
                best.update(shared_best[0])
            barrier()
            model.train()
            return control

    trainer = CorrectionTrainer(model=model, args=training_args, train_dataset=encoded,
                                data_collator=ResponseCollator(tokenizer.pad_token_id),
                                callbacks=[EpochEvaluation(), tracking.callback()])
    if trainer.accelerator.num_processes != world_size or trainer.args.n_gpu > 1:
        raise RuntimeError("Correction requires one replica per DDP rank, never DataParallel")
    if training_args.fp16:
        trainer.accelerator.scaler = torch.amp.GradScaler("cuda", init_scale=training["fp16_initial_scale"])
    result = trainer.train()
    if final_fit and rank == 0:
        save_adapter(trainer.accelerator.unwrap_model(model), adapter_path)
    barrier()
    if not math.isfinite(result.training_loss) or best["epoch"] is None:
        raise RuntimeError("Correction did not complete a finite evaluated training epoch")
    if rank == 0:
        write_json(output / "trainer_history.json", trainer.state.log_history)
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_parameters = sum(p.numel() for p in model.parameters())
    peak_memory = {f"cuda:{i}": torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())} if device == "cuda" else {}
    # Keep the frozen base, but release DDP wrappers and optimizer state before
    # replacing the final-epoch LoRA with the selected adapter from disk.
    trainer.callback_handler.model = None
    trainer.callback_handler.optimizer = None
    trainer.callback_handler.lr_scheduler = None
    trainer.model = trainer.model_wrapped = trainer.optimizer = trainer.lr_scheduler = None
    trainer.accelerator.free_memory()
    trainer.accelerator.scaler = None
    del trainer
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    barrier()
    if rank != 0:
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        barrier()
        torch.distributed.destroy_process_group()
        return
    reloaded, reloaded_tokenizer, saved_inference = _reload_selected_corrector(model, adapter_path)
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    if final_fit:
        # Probe saved-adapter generation on a training input, discard its result:
        # this establishes reload mechanics, never an independent score.
        correct(reloaded, reloaded_tokenizer, train[:1],
                [train_hints[_id_key(train[0]["id"])]], saved_inference)
        from .runtime import unscored_final_report
        predictions, metrics = [], unscored_final_report()
    else:
        predictions = correct(reloaded, reloaded_tokenizer, validation, selected_hints, saved_inference)
        metrics = experiment_report(validation, predictions)
    write_json(output / "predictions.json", comparison_rows(validation, predictions))
    write_json(output / "metrics.json", metrics)
    tracking.model(adapter_path, "corrector", metadata={"best_epoch": best["epoch"],
                                                     "selection_values": best["key"]}, aliases=["latest", "selected"])
    del reloaded, reloaded_tokenizer
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    manifest = {
        "config": config, "resolved_model_revision": settings["revision"], "model": settings,
        "adapter_path": str(adapter_path), "output_dir": str(output), "seed": seed,
        "initial_adapter": initial["path"] if initial else None,
        "continuation": initial["continuation"] if initial else None,
        "data_path": str(data_path), "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "classifier_run": str(classifier_run),
        "hint_mode": inference["hint_mode"], "hints_sha256": hints_sha256,
        "validation_predictions_sha256": validation_predictions_sha256,
        "data_provenance": provenance, "train_ids": [row["id"] for row in train],
        "validation_ids": [row["id"] for row in validation], "inference": inference,
        "device": device, "placement": placement, "device_map": placement["device_map"],
        "quantized": placement["quantized"], "load_in_4bit": placement["quantized"], "cpu_offload": False,
        "parallelism": "DDP single-device replicas" if world_size > 1 else "single-process explicit device map",
        "world_size": world_size,
        "compute_dtype": str(dtype), "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters, "training_args": training_args.to_dict(),
        "effective_batch_size": training["effective_batch_size"], "gradient_accumulation_steps": accumulation,
        "fp16_initial_scale": training["fp16_initial_scale"] if training_args.fp16 else None,
        "train_max_tokens": max(len(row["input_ids"]) for row in encoded),
        "validation_max_tokens": max(validation_lengths, default=None), "optimizer_steps": result.global_step,
        "training_loss": result.training_loss, "best_epoch": best["epoch"],
        "selection_metrics": [] if final_fit else list(SELECTION_METRICS),
        "selection_rule": ("fixed final epoch; no independent validation" if final_fit else
                           "highest error-only GLEU+, then error-only exact, then clean preservation; ties keep earlier epoch"),
        "final_fit": final_fit, "saved_adapter_reload_verified": True,
        "saved_adapter_reloaded_for_evaluation": not final_fit, "peak_allocated_vram_bytes": peak_memory,
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes")},
        "devices": [torch.cuda.get_device_name(i) for i in range(world_size if settings["device_map"] == "single" else 2)] if device == "cuda" else ["cpu"] * world_size,
    }
    write_json(output / "run_manifest.json", manifest)
    tracking.log({"training_loss": result.training_loss}, "training")
    tracking.complete("corrector", metrics, best_epoch=best["epoch"])
    print(json.dumps({"output_dir": str(output), "adapter_path": str(adapter_path),
                      "best_epoch": best["epoch"], "error_gleu_plus": metrics.get("corrections", {}).get("error_only", {}).get("gleu_plus"),
                      "combined_score": metrics["combined_score"],
                      "combined_score_metadata": metrics["combined_score_metadata"]}), flush=True)
    barrier()
    if world_size > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
