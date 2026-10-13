"""Read-only discovery and validation of portable completed stage artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import struct

from .runtime import write_json

HANDOFF_FORMAT = "vgec-run-handoff-v1"
CLASSIFICATION_FILES = ("run_manifest.json", "train.json", "validation.json", "metrics.json", "config.json")


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _manifest_stage(root, stage):
    manifest = _json(root / "run_handoff.json")
    if manifest.get("format") != HANDOFF_FORMAT:
        raise ValueError(f"Unsupported run_handoff.json format at {root}")
    if stage not in manifest.get("completed_stages", []):
        raise ValueError(f"Prior run has no completed {stage} stage: {root}")
    relative = manifest.get("stages", {}).get(stage)
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Handoff stage paths must be nonempty relative paths")
    result = (root / relative).resolve()
    if not result.is_relative_to(root.resolve()) or not result.is_dir():
        raise ValueError(f"Handoff {stage} path escapes its root or is missing: {relative}")
    return result


def _is_stage(root, stage):
    checkpoint = "classifier" if stage == "classification" else "adapter"
    return (root / "run_manifest.json").is_file() and (root / checkpoint / "adapter_config.json").is_file()


def resolve_run(path, stage="classification"):
    """Resolve a stage, pipeline root, or uniquely populated Kaggle input mount."""
    if stage not in ("classification", "correction"):
        raise ValueError("stage must be classification or correction")
    root = Path(path).expanduser().resolve()
    if root.name == "run_handoff.json" and root.is_file():
        root = root.parent
    if not root.is_dir():
        raise FileNotFoundError(f"Prior run directory does not exist: {root}")
    if (root / "run_handoff.json").is_file():
        return _manifest_stage(root, stage)
    if _is_stage(root, stage):
        return root
    if _is_stage(root / stage, stage):
        return root / stage
    manifests = sorted(root.rglob("run_handoff.json"))
    # A handoff is authoritative; do not also count its enclosed legacy stage.
    candidates = [entry.parent for entry in manifests]
    manifest_roots = set(candidates)
    for entry in sorted(root.rglob("run_manifest.json")):
        parent = entry.parent
        if any(parent.is_relative_to(owner) for owner in manifest_roots):
            continue
        if _is_stage(parent, stage) and "oof" not in parent.relative_to(root).parts:
            candidates.append(parent)
    if len(candidates) != 1:
        if candidates:
            raise ValueError(f"Ambiguous prior run: multiple saved runs under {root}; pass a specific run directory: "
                             + ", ".join(str(value) for value in candidates))
        raise FileNotFoundError(f"No completed {stage} run found under {root}; provide an extracted saved output, not an archive")
    candidate = candidates[0]
    return _manifest_stage(candidate, stage) if (candidate / "run_handoff.json").is_file() else candidate


def resolve_adapter(path):
    """Accept an explicit correction adapter or a completed correction handoff."""
    root = Path(path).expanduser().resolve()
    if (root / "adapter_config.json").is_file():
        return root
    return resolve_run(root, "correction") / "adapter"


def _weight_keys(checkpoint):
    safe = checkpoint / "adapter_model.safetensors"
    binary = checkpoint / "adapter_model.bin"
    if safe.is_file():
        with safe.open("rb") as stream:
            length_bytes = stream.read(8)
            if len(length_bytes) != 8:
                raise ValueError("Saved classifier adapter weights are truncated")
            length = struct.unpack("<Q", length_bytes)[0]
            if length > safe.stat().st_size - 8:
                raise ValueError("Saved classifier adapter weights are truncated")
            header = json.loads(stream.read(length))
        entries = {key: value for key, value in header.items() if key != "__metadata__"}
        payload = safe.stat().st_size - 8 - length
        if not entries or any(not isinstance(value, dict) or not isinstance(value.get("data_offsets"), list)
                              or len(value["data_offsets"]) != 2
                              or not 0 <= value["data_offsets"][0] < value["data_offsets"][1] <= payload
                              for value in entries.values()):
            raise ValueError("Saved classifier adapter tensor payload is missing or invalid")
        return set(entries)
    if binary.is_file():
        import torch
        return set(torch.load(binary, map_location="cpu", weights_only=True))
    raise FileNotFoundError("Saved classifier is missing adapter_model.safetensors or adapter_model.bin (including its score head)")


def validate_classifier_run(classifier_run, classification=None, *, require_hints=True):
    """Validate completed adapter/head and metadata without loading base weights."""
    classifier_run = Path(classifier_run)
    checkpoint = classifier_run / "classifier"
    required = [checkpoint / name for name in ("classifier_config.json", "adapter_config.json", "preprocessing.json", "tokenizer_config.json")]
    required.extend(classifier_run / name for name in CLASSIFICATION_FILES)
    if require_hints:
        required.append(classifier_run / "hints.json")
    missing = [str(path.relative_to(classifier_run)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Completed classifier reuse requires saved adapter/head and metadata; missing: " + ", ".join(missing)
                                + ". For Task2, train Task1 with oof.enabled=true on the same correction dataset.")
    from .classification import _require_hierarchy, _preprocessing
    from .data import LABELS
    saved, adapter, config = (_json(checkpoint / "classifier_config.json"),
                              _json(checkpoint / "adapter_config.json"), _json(classifier_run / "config.json"))
    _require_hierarchy(saved)
    model = saved.get("model", {})
    if (saved.get("num_labels") != 5 or saved.get("labels") != list(LABELS)
            or adapter.get("task_type") != "SEQ_CLS" or "score" not in (adapter.get("modules_to_save") or [])
            or adapter.get("base_model_name_or_path") != model.get("name")
            or adapter.get("revision") != model.get("revision")):
        raise ValueError("Reused classifier must be a compatible five-class QLoRA adapter with its score head and pinned base revision")
    configured_model = config.get("model", {})
    model_fields_match = ({key: value for key, value in model.items() if key != "compute_dtype"}
                          == {key: value for key, value in configured_model.items() if key != "compute_dtype"})
    requested_dtype = saved.get("requested_compute_dtype", model.get("compute_dtype"))
    resolved_dtype_ok = (model.get("compute_dtype") == requested_dtype
                         or requested_dtype == "bfloat16" and model.get("compute_dtype") == "float16")
    if (not model_fields_match or requested_dtype != configured_model.get("compute_dtype") or not resolved_dtype_ok
            or _json(checkpoint / "preprocessing.json") != _preprocessing(config)):
        raise ValueError("Saved classifier configuration/preprocessing differs from its run metadata")
    for key, value in config.get("lora", {}).items():
        actual = adapter.get(key)
        if key == "modules_to_save":
            # SEQ_CLS PEFT adds both standard head names, even if one is absent,
            # and may repeat an explicitly configured head in saved metadata.
            actual, value = set(actual or []), set(value or [])
            if value <= actual <= value | {"classifier", "score"}:
                continue
        elif key == "target_modules":
            actual, value = set(actual or []), set(value or [])
        if actual != value:
            raise ValueError(f"Saved classifier LoRA setting differs: {key}")
    if classification is not None and any(model.get(key) != classification["model"].get(key)
                                          for key in ("name", "revision", "load_in_4bit")):
        raise ValueError("Reused classifier base/revision/quantization differs from requested classification configuration")
    keys = _weight_keys(checkpoint)
    if not any("score" in key.split(".") and key.endswith("weight") for key in keys):
        raise ValueError("Saved classifier adapter weights do not contain the score head")
    if not any("lora_" in key for key in keys):
        raise ValueError("Saved classifier adapter weights do not contain LoRA tensors")
    manifest = _json(classifier_run / "run_manifest.json")
    if (manifest.get("task") != "classification" or not manifest.get("saved_classifier_reloaded_for_evaluation")
            or not manifest.get("classifier_data_sha256")):
        raise ValueError("Reused classifier must be a completed mixed-data classifier run with saved-adapter evaluation")
    if (manifest.get("seed") != config["run"]["seed"]
            or manifest.get("validation_fraction") != config["data"]["validation_fraction"]):
        raise ValueError("Saved classifier split metadata differs from its configuration")
    if require_hints and not manifest.get("hints_enabled"):
        raise ValueError("Task2 requires saved OOF hints; rerun Task1 with oof.enabled=true")
    return config


def validate_reuse_data(classifier_run, correction, data_path, *, original_path=None):
    """Reject population changes before any launch; never regenerate in-sample hints."""
    from .correction import validate_hints
    from .data import load_training_split
    root = Path(classifier_run)
    manifest = _json(root / "run_manifest.json")
    if original_path is not None and manifest.get("data_sha256") != hashlib.sha256(Path(original_path).read_bytes()).hexdigest():
        raise ValueError("Reused classifier original dataset SHA256 differs; train a fresh Task1/OOF run for the changed population")
    seed, fraction = correction["run"]["seed"], correction["data"]["validation_fraction"]
    train, validation, provenance = load_training_split(data_path, fraction, seed)
    if _json(root / "validation.json") != validation:
        raise ValueError("Reused classifier validation split differs from correction holdout; train a fresh Task1/OOF run")
    artifact = _json(root / "hints.json")
    if manifest.get("correction_data_sha256") != artifact.get("correction_data_sha256"):
        raise ValueError("Saved classifier manifest and OOF correction dataset hashes differ")
    if manifest.get("oof_folds") != artifact.get("folds"):
        raise ValueError("Saved classifier manifest and OOF fold declarations differ")
    try:
        validate_hints(artifact, train, validation, provenance, _json(root / "train.json"), data_path, seed, fraction)
    except (ValueError, KeyError, TypeError) as error:
        raise ValueError(f"Saved OOF hints are incompatible with this correction dataset: {error}. "
                         "Train a fresh Task1 with OOF on this exact population; a saved main adapter cannot safely regenerate training hints.") from error
    return {"correction_train": len(train), "validation": len(validation), "seed": seed}


def copy_classifier_run(source, destination):
    """Copy all inference and OOF-validation inputs, without archive dependencies."""
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise ValueError(f"Classifier export destination already exists: {destination}")
    destination.mkdir(parents=True)
    shutil.copytree(source / "classifier", destination / "classifier")
    for name in (*CLASSIFICATION_FILES, "hints.json", "oof_metrics.json", "wandb_run.json",
                 "preprocessing.json", "predictions.json", "source_manifest.json"):
        if (source / name).is_file():
            shutil.copy2(source / name, destination / name)
    manifest = _json(destination / "run_manifest.json")
    manifest.update(output_dir=".", classifier_path="classifier")
    if (destination / "hints.json").is_file():
        manifest["hints_path"] = "hints.json"
    write_json(destination / "run_manifest.json", manifest)


def write_handoff(output, *, correction=False):
    stages = {"classification": "classification"}
    if correction:
        stages["correction"] = "correction"
    write_json(Path(output) / "run_handoff.json", {
        "format": HANDOFF_FORMAT, "completed_stages": list(stages), "stages": stages,
        "reuse_contract": "Completed Task1 adapter/head plus exact-population saved OOF hints; Task2 continuation requires a compatible correction adapter.",
    })
