"""Paired saved-adapter hint ablation and opt-in candidate reranking, without training."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path

from .correction import _hint_index, correct, load_corrector
from .data import _id_key, _validate_records
from .metrics import experiment_report
from .prediction_export import comparison_rows
from .runtime import write_json
from .run_reuse import resolve_run, validate_classifier_run, validation_predictions
from .tracking import records_identity, tracked_entrypoint


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_experiment_inputs(run):
    """Bind evaluation labels to the completed classifier and saved prediction hashes."""
    correction = resolve_run(run, stage="correction")
    manifest = _json(correction / "run_manifest.json")
    if not manifest.get("saved_adapter_reloaded_for_evaluation"):
        raise ValueError("Experiments require a completed, reloaded correction adapter")
    records = _json(correction / "validation.json")
    _validate_records(records, require_labels=True)
    if not records or manifest.get("validation_ids") != [row["id"] for row in records]:
        raise ValueError("Saved correction validation IDs differ from run manifest")
    # For a direct correction stage, prefer its enclosing portable handoff; legacy
    # runs can still use their original classifier_run when it remains available.
    if (correction.parent / "run_handoff.json").is_file() or (correction.parent / "classification").is_dir():
        classifier = resolve_run(correction.parent, stage="classification")
    else:
        classifier = resolve_run(manifest["classifier_run"], stage="classification")
    no_hints = manifest.get("inference", {}).get("hint_mode", "predicted") == "none"
    validate_classifier_run(classifier, require_hints=not no_hints)
    if no_hints:
        path = classifier / "predictions.json"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest.get("validation_predictions_sha256"):
            raise ValueError("Saved validation classifier predictions differ from correction evaluation")
        indexed = validation_predictions(classifier, records)
        hints = [indexed[_id_key(row["id"])] for row in records]
        if _json(correction / "validation_hints.json") != hints:
            raise ValueError("Saved correction validation predictions differ")
        return correction, records, hints
    hint_path = classifier / "hints.json"
    if hashlib.sha256(hint_path.read_bytes()).hexdigest() != manifest.get("hints_sha256"):
        raise ValueError("Saved OOF hints SHA256 differs from correction training")
    artifact = _json(hint_path)
    classifier_manifest = _json(classifier / "run_manifest.json")
    if (artifact.get("correction_data_sha256") != manifest.get("data_sha256")
            or classifier_manifest.get("correction_data_sha256") != artifact.get("correction_data_sha256")
            or classifier_manifest.get("oof_folds") != artifact.get("folds")
            or artifact.get("seed") != manifest.get("seed")
            or artifact.get("validation_fraction") != manifest["config"]["data"]["validation_fraction"]):
        raise ValueError("Classifier OOF provenance differs from saved correction run")
    outer_validation = _json(classifier / "validation.json")
    indexed = _hint_index(outer_validation, artifact.get("validation"))
    outer_records = {_id_key(row["id"]): row for row in outer_validation}
    for row in records:
        if outer_records.get(_id_key(row["id"])) != row:
            raise ValueError("Correction validation records differ from saved classifier holdout")
    hints = [indexed[_id_key(row["id"])] for row in records]
    _hint_index(records, hints)
    if (correction / "validation_hints.json").is_file() and _json(correction / "validation_hints.json") != hints:
        raise ValueError("Saved correction validation hints differ from the OOF artifact")
    return correction, records, hints


def experiment_variants(inference, *, hint_ablation=False, rerank=False):
    """Always include the untouched saved inference as a paired baseline."""
    variants = {"saved": copy.deepcopy(inference)}
    if hint_ablation:
        for mode in ("predicted", "none"):
            variant = copy.deepcopy(inference)
            variant["hint_mode"] = mode
            variants[f"hint_{mode}"] = variant
    if rerank:
        from .correction_reranking import DEFAULT_RERANKING
        modes = ("predicted", "none") if hint_ablation else (inference.get("hint_mode", "predicted"),)
        for mode in modes:
            variant = copy.deepcopy(inference)
            variant["hint_mode"] = mode
            variant["reranking"] = {**DEFAULT_RERANKING, **inference.get("reranking", {}), "enabled": True}
            variants[f"rerank_{mode}"] = variant
    return variants


def paired_rows(records, predictions):
    indices = {name: {_id_key(row["id"]): row for row in rows} for name, rows in predictions.items()}
    for rows in predictions.values():
        if len(rows) != len(records) or set(_id_key(row["id"]) for row in rows) != set(_id_key(row["id"]) for row in records):
            raise ValueError("Every experiment must predict the exact same typed validation IDs")
    return [{"id": record["id"], "source": record["source"], "target_class": record["label"],
             "target_correction": record["target"],
             "variants": {name: index[_id_key(record["id"])] for name, index in indices.items()}}
            for record in records]


@tracked_entrypoint
def main(*, tracking):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Completed portable pipeline root or correction stage")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--hint-ablation", action="store_true")
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--tracking-config", help="JSON with top-level tracking or a tracking settings object")
    args = parser.parse_args()
    if not args.hint_ablation and not args.rerank:
        parser.error("Select --hint-ablation and/or --rerank")
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Experiment output directory must be empty")
    stage, records, hints = load_experiment_inputs(args.run)
    config = _json(stage / "config.json")
    if args.tracking_config:
        override = _json(args.tracking_config)
        config["tracking"] = override.get("tracking", override)
    tracking.start(config, output, "correction-diagnostics", group=stage.parent.name,
                   metadata={"validation": records_identity(records), "training_performed": False})
    tracking.use_model(stage / "adapter", "corrector")
    model, tokenizer, inference = load_corrector(stage / "adapter", args.device)
    variants = experiment_variants(inference, hint_ablation=args.hint_ablation, rerank=args.rerank)
    output.mkdir(parents=True, exist_ok=True)
    predictions, reports = {}, {}
    for name, variant in variants.items():
        predictions[name] = correct(model, tokenizer, records, hints, variant)
        reports[name] = experiment_report(records, predictions[name])
        destination = output / name
        destination.mkdir()
        write_json(destination / "inference.json", variant)
        write_json(destination / "predictions.json", comparison_rows(records, predictions[name]))
        write_json(destination / "metrics.json", reports[name])
        tracking.evaluation(reports[name], f"variants/{name}")
    write_json(output / "paired_predictions.json", paired_rows(records, predictions))
    write_json(output / "metrics.json", reports)
    write_json(output / "experiment_manifest.json", {
        "run": str(stage), "adapter": str(stage / "adapter"), "device": args.device,
        "validation_ids": [row["id"] for row in records], "variants": variants,
        "validation_sha256": hashlib.sha256((stage / "validation.json").read_bytes()).hexdigest(),
        "training_performed": False, "hints": "saved outer-holdout predictions from validated OOF artifact",
        "ranking": "deterministic beams plus unchanged source; adapter-disabled base Có/Không teacher-forced judgment; CPU multilingual cosine; weighted scores then edited-character tie-break",
    })
    del model, tokenizer
    print(json.dumps({"output_dir": str(output), "variants": list(variants)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
