"""Portable1039 soft voting for fresh Qwen14B, Sailor20B and Mistral12B fits."""
import hashlib
import json
from pathlib import Path

from .classifier_ensemble import vote, voting_rule
from .runtime import write_json

NAMES = ["qwen14b", "sailor20b", "mistral12b"]
FORMAT = "vgec-upgraded-classifier-soft-vote-v1"


def read(path):
    return json.loads(Path(path).read_text())


def build(root, config):
    from .classification import _classification_metrics
    from .prediction_export import comparison_rows
    root = Path(root)
    if config.get("competition_id") != 1039 or [e["name"] for e in config["ensemble"]["members"]] != NAMES:
        raise ValueError("Upgraded ensemble requires the three committed1039 member names")
    manifests, populations, predictions, entries = [], [], [], []
    for entry in config["ensemble"]["members"]:
        member = root / "members" / entry["name"]
        manifest = read(member / "run_manifest.json")
        child = read(member / "config.json")
        if child["model"] != entry["model"] or child.get("ensemble") or manifest["seed"] != 42:
            raise ValueError("Ensemble member differs from its fresh pinned config")
        manifests.append(manifest)
        populations.append([read(member / "train.json"), read(member / "validation.json")])
        rows = read(member / "predictions.json")
        predictions.append([{"id": row["id"], "label": row["predicted_class"],
                             "probabilities": row["probabilities"]} for row in rows])
        weights = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted((member / "classifier").glob("adapter_model*")) if p.is_file()}
        if not weights:
            raise ValueError("Missing saved classifier weights")
        entries.append({"name": entry["name"], "path": f"../members/{entry['name']}/classifier",
                        "adapter_sha256": weights})
    if any(population != populations[0] for population in populations[1:]):
        raise ValueError("All classifier members must use identical training/development rows")
    train, validation = populations[0]
    calibration = config["ensemble"]["calibration"]
    selected = vote(predictions, calibration)
    checkpoint = root / "classifier"
    checkpoint.mkdir(exist_ok=False)
    write_json(checkpoint / "ensemble.json", {"format": FORMAT, "names": NAMES, "members": entries,
               "calibration": calibration, "rule": voting_rule(calibration)})
    metrics = {"count": len(validation), "classification": _classification_metrics(validation, selected),
               "evaluation_status": "selected_epoch_member_predictions", "ensemble_members": NAMES}
    for name, value in [("config.json", config), ("train.json", train), ("validation.json", validation),
                        ("predictions.json", comparison_rows(validation, selected)),
                        ("voting_details.json", selected), ("metrics.json", metrics)]:
        write_json(root / name, value)
    write_json(root / "run_manifest.json", {
        **manifests[0], "config": config, "model": config["model"], "output_dir": str(root),
        "classifier_path": str(checkpoint), "ensemble_members": NAMES, "hints_enabled": False,
        "train_count": len(train), "validation_count": len(validation), "best_epoch": None,
        "training_loss": None, "optimizer_steps": sum(m["optimizer_steps"] for m in manifests),
        "member_best_epochs": {name: m["best_epoch"] for name, m in zip(NAMES, manifests)},
        "member_training_statistics": {name: {key: m[key] for key in ("best_epoch", "training_loss", "optimizer_steps")}
                                       for name, m in zip(NAMES, manifests)},
        "selection_rule": "Each member selects its own best development Macro-F1 epoch; fixed weighted soft vote",
        "voting_rule": voting_rule(calibration), "saved_classifier_reloaded_for_evaluation": False,
        "saved_classifier_reload_verified": False, "verification_status": "skipped_user_request"})
    return metrics


def classify(checkpoint, records, device="cuda", batch_size=8):
    from .classification import classify as single
    checkpoint = Path(checkpoint).resolve()
    descriptor = read(checkpoint / "ensemble.json")
    if descriptor.get("format") != FORMAT or descriptor.get("names") != NAMES or len(descriptor.get("members", [])) != 3:
        raise ValueError("Invalid upgraded classifier descriptor")
    predictions = []
    for name, entry in zip(NAMES, descriptor["members"]):
        expected = f"../members/{name}/classifier"
        path = (checkpoint / expected).resolve()
        if entry.get("name") != name or entry.get("path") != expected or not path.is_relative_to(checkpoint.parent):
            raise ValueError("Upgraded member path differs or escapes its run")
        weights = entry.get("adapter_sha256", {})
        if not weights:
            raise ValueError("Upgraded member has no weight bindings")
        for file, digest in weights.items():
            if Path(file).name != file or not file.startswith("adapter_model") or hashlib.sha256((path / file).read_bytes()).hexdigest() != digest:
                raise ValueError("Upgraded classifier weights differ")
        predictions.append(single(path, records, device, batch_size))
    return vote(predictions, descriptor["calibration"])
