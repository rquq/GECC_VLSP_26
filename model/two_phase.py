"""Run independently configured QLoRA classification and correction stages."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from .runtime import PROJECT_ROOT, load_config, resolve_data, stage_output, write_json




def validate_classifier_run(classifier_run, classification):
    checkpoint = classifier_run / "classifier"
    required = [checkpoint / "classifier_config.json", checkpoint / "adapter_config.json"]
    required += [classifier_run / filename for filename in
                 ("hints.json", "run_manifest.json", "train.json", "validation.json", "metrics.json", "config.json")]
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("Reused classifier run must contain classifier/classifier_config.json, "
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
        raise ValueError("Reused classifier must be a compatible five-class QLoRA adapter with its score head")
    manifest = json.loads((classifier_run / "run_manifest.json").read_text(encoding="utf-8"))
    if "classifier_data_sha256" not in manifest:
        raise ValueError("Reused classifier must train on the capped mixed classifier bundle; retrain old original-only runs")


def preflight(classification, correction, original, bundle, *, fresh_classifier=True):
    """Verify split compatibility and available model placement before weight downloads."""
    if fresh_classifier and not classification.get("oof", {}).get("enabled", False):
        raise ValueError("Fresh two-phase runs require oof.enabled=true for leakage-safe correction hints")
    import torch
    for config in (classification, correction):
        world_size = config["run"].get("world_size", 1)
        if type(world_size) is not int or world_size not in (1, 2):
            raise ValueError("run.world_size must be 1 or 2")
        if config["run"]["device"] == "cuda" and (not torch.cuda.is_available() or torch.cuda.device_count() < world_size):
            raise RuntimeError(f"This stage requires {world_size} visible CUDA GPUs")
        global_microbatch = config["training"]["args"]["per_device_train_batch_size"] * world_size
        effective = config["training"]["effective_batch_size"]
        if effective < global_microbatch or effective % global_microbatch:
            raise ValueError("effective_batch_size must be a positive multiple of microbatch * world_size")
    if correction["run"].get("world_size", 1) > 1 and correction["model"].get("device_map", "single") != "single":
        raise ValueError("DDP correction requires single-device replicas, not balanced model splitting")
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
    classifier_train, classifier_info = load_classifier_split(original, mixed_path, fraction, seed)
    train, validation, _ = load_training_split(bundle, fraction, seed)
    if original_validation != validation:
        raise ValueError("Correction validation must be the identical original holdout")
    from .classification import placement_preflight as classifier_placement_preflight
    classifier_world = classification["run"].get("world_size", 1)
    classifier_placements = [classifier_placement_preflight(
        classification["model"], f"cuda:{rank}" if classification["run"]["device"] == "cuda" else "cpu")
        for rank in range(classifier_world)]
    classifier_placement = classifier_placements[0]
    from .correction import placement_preflight
    corrector_world = correction["run"].get("world_size", 1)
    placements = [placement_preflight(correction["model"], correction["run"]["device"], local_rank=rank)
                  for rank in range(corrector_world)]
    placement = placements[0]
    report = {"seed": seed, "classification_train": len(classifier_train),
              "classification_original_train": len(original_train),
              "classification_retained_original_train": len(classifier_info["provenance"]["retained_original_train_ids"]),
              "classification_quarantined_original_train": len(classifier_info["provenance"]["quarantined_ids"]),
              "classification_synthetic_train": len(classifier_info["provenance"]["selected_synthetic_ids"]),
              "correction_train": len(train), "validation": len(validation),
              "classification_world_size": classifier_world, "correction_world_size": corrector_world,
              "classification_replicas": classifier_placements, "correction_replicas": placements,
              "original_data": str(original), "correction_data": str(bundle),
              "classification_placement": classifier_placement,
              "classification_quantized": classifier_placement["quantized"],
              "correction_placement": placement, "correction_quantized": placement["quantized"]}
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def stage_command(module, config):
    world_size = config["run"].get("world_size", 1)
    if world_size > 1:
        return [sys.executable, "-m", "torch.distributed.run", "--standalone",
                "--nproc_per_node", str(world_size), "-m", module]
    return [sys.executable, "-m", module]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classification-config", default="configs/baseline.json")
    parser.add_argument("--correction-config", default="configs/correction.json")
    parser.add_argument("--classification-data")
    parser.add_argument("--correction-data")
    parser.add_argument("--classifier-run", help="Reuse a completed compatible classifier run and its OOF hints")
    parser.add_argument("--output-dir")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    classifier = load_config(args.classification_config, "classification")
    corrector = load_config(args.correction_config, "correction")
    original = resolve_data(classifier, args.classification_data)
    bundle = resolve_data(corrector, args.correction_data)
    classifier_run = Path(args.classifier_run).expanduser().resolve() if args.classifier_run else None
    if classifier_run is not None:
        validate_classifier_run(classifier_run, classifier)
    preflight(classifier, corrector, original, bundle, fresh_classifier=classifier_run is None)
    if args.preflight_only:
        return
    output = stage_output({"run": {"name": "sailor2-two-phase", "output_root": corrector["run"]["output_root"]}}, args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    from tools.runtime import source_identity
    write_json(output / "source_manifest.json", source_identity(PROJECT_ROOT))
    write_json(output / "classification_config.json", classifier)
    write_json(output / "correction_config.json", corrector)
    classifier_run = classifier_run or output / "classification"
    if not args.classifier_run:
        subprocess.run(stage_command("model.classification", classifier) + ["--config", str(Path(args.classification_config).resolve()),
                        "--data", str(original), "--correction-data", str(bundle), "--output-dir", str(classifier_run)],
                       cwd=PROJECT_ROOT, check=True)
    subprocess.run(stage_command("model.correction", corrector) + ["--config", str(Path(args.correction_config).resolve()),
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
