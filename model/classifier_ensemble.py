"""Three independently fitted classifier models, portable manifests and deterministic voting."""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from .data import _id_key
from .runtime import write_json

NAMES = ["qwen7b", "sailor7b", "mistral7b"]
FORMAT = "vgec-classifier-soft-vote-v1"
RULE = "equal-weight mean of member probabilities; exact ties use lower class ID"


def vote(predictions):
    if len(predictions) != 3:
        raise ValueError("Voting requires exactly three classifier members")
    from .correction import _hint_index

    records = [{"id": r["id"]} for r in predictions[0]]
    indexed = [_hint_index(records, rows) for rows in predictions]
    result = []
    for row in records:
        values = [member[_id_key(row["id"])] for member in indexed]
        counts = Counter(r["label"] for r in values)
        means = [math.fsum(r["probabilities"][i] for r in values) / 3 for i in range(5)]
        label = max(range(5), key=lambda i: (means[i], -i))
        result.append(
            {
                "id": row["id"],
                "label": label,
                "probabilities": means,
                "mean_probabilities": means,
                "votes": [counts[i] for i in range(5)],
            }
        )
    return result


def members(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    value = json.loads((checkpoint / "ensemble.json").read_text())
    if (
        value.get("format") != FORMAT
        or value.get("names") != NAMES
        or value.get("rule") != RULE
        or len(value.get("members", [])) != 3
    ):
        raise ValueError("Invalid three-model ensemble protocol")
    paths = []
    for member_name, entry in zip(NAMES, value["members"]):
        expected = f"../members/{member_name}/classifier"
        if entry.get("name") != member_name or entry.get("path") != expected:
            raise ValueError("Ensemble member path/name differs")
        path = (checkpoint / expected).resolve()
        if (
            not path.is_relative_to(checkpoint.parent)
            or (path / "ensemble.json").exists()
        ):
            raise ValueError("Ensemble member escapes its bundle or nests an ensemble")
        files = entry.get("adapter_sha256", {})
        if not files or any(
            Path(n).name != n or not n.startswith("adapter_model") for n in files
        ):
            raise ValueError("Ensemble must bind member adapter hashes")
        for name, digest in files.items():
            if hashlib.sha256((path / name).read_bytes()).hexdigest() != digest:
                raise ValueError("Ensemble member adapter hash differs")
        paths.append(path)
    return paths


def _member_population(root):
    from .run_reuse import validate_classifier_run

    configs, manifests, populations = [], [], []
    for member_name in NAMES:
        member = root / "members" / member_name
        config = validate_classifier_run(member, require_hints=False)
        manifest = json.loads((member / "run_manifest.json").read_text())
        if (
            config["run"]["seed"] != 42
            or manifest["seed"] != 42
            or config.get("ensemble")
        ):
            raise ValueError("Ensemble member is not an independent seed-42 model fit")
        configs.append(config)
        manifests.append(manifest)
        populations.append(
            [
                json.loads((member / name).read_text())
                for name in ("train.json", "validation.json")
            ]
        )
    from .classifier_resume import data_recipe

    for i in (1, 2):
        if (
            populations[i] != populations[0]
            or configs[i].get("lora") != configs[0].get("lora")
            or configs[i]["training"] != configs[0]["training"]
            or data_recipe(configs[i]) != data_recipe(configs[0])
            or any(
                manifests[i].get(key) != manifests[0].get(key)
                for key in (
                    "data_sha256",
                    "classifier_data_sha256",
                    "validation_fraction",
                )
            )
        ):
            raise ValueError("Ensemble members differ in data or training recipe")
    return configs, manifests, populations[0]


def build(root, config):
    """Create a completed classifier stage after all three independently fitted members."""
    from .classification import _classification_metrics
    from .prediction_export import comparison_rows

    root = Path(root)
    configs, manifests, (train, validation) = _member_population(root)
    for entry, child in zip(config["ensemble"]["members"], configs):
        if child["model"] != entry["model"]:
            raise ValueError("Member differs from its requested pinned model")
    checkpoint = root / "classifier"
    checkpoint.mkdir(parents=True, exist_ok=False)
    entries = []
    member_predictions = []
    for member_name in NAMES:
        member = root / "members" / member_name
        hashes = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((member / "classifier").glob("adapter_model*"))
            if p.is_file()
        }
        entries.append(
            {
                "name": member_name,
                "path": f"../members/{member_name}/classifier",
                "adapter_sha256": hashes,
            }
        )
        rows = json.loads((member / "predictions.json").read_text())
        member_predictions.append(
            [
                {
                    "id": r["id"],
                    "label": r["predicted_class"],
                    "probabilities": r["probabilities"],
                }
                for r in rows
            ]
        )
    predictions = vote(member_predictions)
    write_json(
        checkpoint / "ensemble.json",
        {
            "format": FORMAT,
            "names": NAMES,
            "members": entries,
            "rule": RULE,
            "probabilities": "arithmetic mean of three member joint class probability distributions; weights 1/3 each",
        },
    )
    for name, value in (
        ("config.json", config),
        ("train.json", train),
        ("validation.json", validation),
        ("predictions.json", comparison_rows(validation, predictions)),
        ("voting_details.json", predictions),
    ):
        write_json(root / name, value)
    metrics = {
        "count": len(validation),
        "classification": (
            _classification_metrics(validation, predictions) if validation else None
        ),
        "evaluation_status": "evaluated" if validation else "unavailable_full_training",
        "ensemble_members": NAMES,
        "voting_rule": RULE,
    }
    write_json(root / "metrics.json", metrics)
    manifest = {
        **manifests[0],
        "config": config,
        "output_dir": str(root),
        "classifier_path": str(checkpoint),
        "seed": 42,
        "split_seed": 42,
        "ensemble_members": NAMES,
        "voting_rule": RULE,
        "saved_classifier_reloaded_for_evaluation": bool(validation),
        "saved_classifier_reload_verified": True,
        "final_fit": not bool(validation),
        "train_count": len(train),
        "validation_count": len(validation),
        "hints_enabled": False,
        "best_epoch": None,
        "member_best_epochs": {
            str(s): m.get("best_epoch") for s, m in zip(NAMES, manifests)
        },
        "selection_rule": "Each fresh member selects its own epoch; classifier outcomes use all three members.",
        "training_loss": None,
        "optimizer_steps": sum(m.get("optimizer_steps", 0) for m in manifests),
        "member_training_statistics": {
            str(s): {
                k: m.get(k) for k in ("training_loss", "optimizer_steps", "best_epoch")
            }
            for s, m in zip(NAMES, manifests)
        },
    }
    write_json(root / "run_manifest.json", manifest)
    return metrics


def validate(root, classification=None):
    root = Path(root)
    members(root / "classifier")
    configs, manifests, (train, validation) = _member_population(root)
    config = json.loads((root / "config.json").read_text())
    if [
        r.get("name") for r in config.get("ensemble", {}).get("members", [])
    ] != NAMES or config["model"] != configs[0]["model"]:
        raise ValueError("Ensemble root configuration differs from its members")
    if classification is not None and (
        config["model"] != classification["model"]
        or config.get("ensemble") != classification.get("ensemble")
    ):
        raise ValueError("Requested classifier differs from ensemble base")
    if (
        json.loads((root / "train.json").read_text()) != train
        or json.loads((root / "validation.json").read_text()) != validation
    ):
        raise ValueError("Ensemble root population differs from members")
    from .prediction_export import comparison_rows

    predictions = []
    for member_name in NAMES:
        rows = json.loads(
            (root / "members" / member_name / "predictions.json").read_text()
        )
        predictions.append(
            [
                {
                    "id": r["id"],
                    "label": r["predicted_class"],
                    "probabilities": r["probabilities"],
                }
                for r in rows
            ]
        )
    if json.loads((root / "predictions.json").read_text()) != comparison_rows(
        validation, vote(predictions)
    ):
        raise ValueError("Ensemble root votes differ from saved member predictions")
    for entry, child in zip(config["ensemble"]["members"], configs):
        if child["model"] != entry["model"]:
            raise ValueError("Member differs from its requested pinned model")
    manifest = json.loads((root / "run_manifest.json").read_text())
    if (
        manifest.get("ensemble_members") != NAMES
        or manifest.get("hints_enabled") is not False
        or manifest.get("train_count") != len(train)
        or manifest.get("validation_count") != len(validation)
        or any(
            manifest.get(k) != manifests[0].get(k)
            for k in ("data_sha256", "classifier_data_sha256", "validation_fraction")
        )
    ):
        raise ValueError("Ensemble manifest differs from completed members")
    return config


def classify(checkpoint, records, device, batch_size):
    """For CUDA, reap each prediction worker before the next member is loaded."""
    from .classification import classify as single
    from .pipeline_predict import _classify_isolated

    predictions = []
    for path in members(checkpoint):
        predictions.append(
            _classify_isolated(path, records, device, batch_size)
            if str(device).startswith("cuda")
            else single(path, records, device, batch_size)
        )
    return vote(predictions)
