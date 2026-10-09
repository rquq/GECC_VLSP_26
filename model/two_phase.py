"""Run independently configured Qwen3 QLoRA classification then Sailor2 QLoRA correction."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from .runtime import PROJECT_ROOT, resolve_output, write_json


def load_stage_config(path, expected_task=None):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    task = config.get("task")
    if task not in ("classification", "correction") or (expected_task and task != expected_task):
        raise ValueError(f"Expected {expected_task or 'classification/correction'} task configuration")
    for section in ("run", "data", "model", "training"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing configuration section: {section}")
    if type(config["run"].get("seed")) is not int or config["run"]["seed"] != 42:
        raise ValueError("run.seed must be 42")
    batch = config["training"]["args"]["per_device_train_batch_size"]
    effective = config["training"]["effective_batch_size"]
    if type(batch) is not int or type(effective) is not int or batch <= 0 or effective < batch or effective % batch:
        raise ValueError("effective_batch_size must be a positive multiple of the microbatch")
    return config


def resolve_data(config, explicit=None):
    from tools.runtime import discover_data
    path = explicit or config["data"].get("path")
    if path is not None:
        path = Path(path).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
    return discover_data(path, PROJECT_ROOT, Path(config["data"]["input_root"]), config["data"]["filename"])


def stage_output(config, explicit=None):
    output = resolve_output(config, explicit)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")
    return output


def validate_classifier_run(classifier_run, classification):
    checkpoint = classifier_run / "classifier"
    required = [checkpoint / "classifier_config.json", checkpoint / "adapter_config.json"]
    required += [classifier_run / filename for filename in
                 ("hints.json", "run_manifest.json", "train.json", "validation.json", "metrics.json", "config.json")]
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("Reused Qwen3 classifier run must contain classifier/classifier_config.json, "
                                "classifier/adapter_config.json, hints.json, and its run metadata; "
                                "historical full-model PhoBERT checkpoints are incompatible")
    saved = json.loads(required[0].read_text(encoding="utf-8"))
    adapter = json.loads(required[1].read_text(encoding="utf-8"))
    from .classification import _require_hierarchy
    _require_hierarchy(saved)
    settings = classification["model"]
    model = saved.get("model", {})
    if (saved.get("num_labels") != 5 or adapter.get("task_type") != "SEQ_CLS"
            or "score" not in (adapter.get("modules_to_save") or [])
            or any(model.get(key) != settings.get(key) for key in ("name", "revision", "load_in_4bit"))):
        raise ValueError("Reused classifier must be a compatible five-class Qwen3 QLoRA adapter with its score head")
    manifest = json.loads((classifier_run / "run_manifest.json").read_text(encoding="utf-8"))
    if "classifier_data_sha256" not in manifest:
        raise ValueError("Reused classifier must train on the capped mixed classifier bundle; retrain old original-only runs")


def preflight(classification, correction, original, bundle):
    """Verify split compatibility and available model placement before weight downloads."""
    import torch
    from .data import load_records, load_training_split, split_records
    if classification["run"]["seed"] != correction["run"]["seed"]:
        raise ValueError("Both task configs must use the same seed")
    fraction = classification["data"]["validation_fraction"]
    if fraction != correction["data"]["validation_fraction"]:
        raise ValueError("Both task configs must use the same validation_fraction")
    seed = classification["run"]["seed"]
    original_train, original_validation = split_records(load_records(original), fraction, seed)
    from .data import load_classifier_split
    import copy
    mixed_config = copy.deepcopy(classification)
    mixed_config["data"].update(path=classification["data"].get("augmentation_path"),
                                filename=classification["data"]["augmentation_filename"])
    mixed_path = resolve_data(mixed_config, None)
    classifier_train, _ = load_classifier_split(original, mixed_path, fraction, seed)
    train, validation, _ = load_training_split(bundle, fraction, seed)
    if original_validation != validation:
        raise ValueError("Correction validation must be the identical original holdout")
    from .classification import placement_preflight as classifier_placement_preflight
    classifier_placement = classifier_placement_preflight(classification["model"], classification["run"]["device"])
    from .correction import placement_preflight
    placement = placement_preflight(correction["model"], correction["run"]["device"])
    report = {"seed": seed, "classification_train": len(classifier_train),
              "classification_original_train": len(original_train),
              "classification_synthetic_train": len(classifier_train) - len(original_train),
              "correction_train": len(train), "validation": len(validation),
              "original_data": str(original), "correction_data": str(bundle),
              "classification_placement": classifier_placement,
              "classification_quantized": classifier_placement["quantized"],
              "correction_placement": placement, "correction_quantized": placement["quantized"]}
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classification-config", default="configs/classification.json")
    parser.add_argument("--correction-config", default="configs/correction.json")
    parser.add_argument("--classification-data")
    parser.add_argument("--correction-data")
    parser.add_argument("--classifier-run", help="Reuse a completed compatible classifier run and its OOF hints")
    parser.add_argument("--output-dir")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    classifier = load_stage_config(args.classification_config, "classification")
    corrector = load_stage_config(args.correction_config, "correction")
    original = resolve_data(classifier, args.classification_data)
    bundle = resolve_data(corrector, args.correction_data)
    classifier_run = Path(args.classifier_run).expanduser().resolve() if args.classifier_run else None
    if classifier_run is not None:
        validate_classifier_run(classifier_run, classifier)
    preflight(classifier, corrector, original, bundle)
    if args.preflight_only:
        return
    output = stage_output({"run": {"name": "qwen3-sailor2", "output_root": corrector["run"]["output_root"]}}, args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    from tools.runtime import source_identity
    write_json(output / "source_manifest.json", source_identity(PROJECT_ROOT))
    write_json(output / "classification_config.json", classifier)
    write_json(output / "correction_config.json", corrector)
    classifier_run = classifier_run or output / "classification"
    if not args.classifier_run:
        subprocess.run([sys.executable, "-m", "model.classification", "--config", str(Path(args.classification_config).resolve()),
                        "--data", str(original), "--correction-data", str(bundle), "--output-dir", str(classifier_run)],
                       cwd=PROJECT_ROOT, check=True)
    subprocess.run([sys.executable, "-m", "model.correction", "--config", str(Path(args.correction_config).resolve()),
                    "--data", str(bundle), "--classifier-run", str(classifier_run), "--output-dir", str(output / "correction")],
                   cwd=PROJECT_ROOT, check=True)
    # Package a reusable classifier alongside the correction adapter, including reuse mode.
    if args.classifier_run:
        shutil.copytree(classifier_run / "classifier", output / "classification" / "classifier")
        for filename in ("hints.json", "run_manifest.json", "train.json", "validation.json", "metrics.json", "config.json"):
            shutil.copy2(classifier_run / filename, output / "classification" / filename)
    manifest = {"classifier": "classification/classifier", "corrector": "correction/adapter",
                "seed": classifier["run"]["seed"], "classifier_source": str(classifier_run),
                "classification_metrics": json.loads((classifier_run / "metrics.json").read_text()),
                "correction_metrics": json.loads((output / "correction" / "metrics.json").read_text())}
    write_json(output / "pipeline_manifest.json", manifest)
    if corrector["run"].get("archive", True):
        archive = shutil.make_archive(str(output) + "-artifacts", "zip", output.parent, output.name)
        print(f"Artifacts: {archive}", flush=True)
    print(f"Completed pipeline: {output}", flush=True)


if __name__ == "__main__":
    main()
