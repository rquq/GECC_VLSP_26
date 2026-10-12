"""Train Task1, Task2, or both, exporting portable completed-stage handoffs."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

from .runtime import PROJECT_ROOT, load_config, resolve_data, stage_output, write_json
from .run_reuse import (copy_classifier_run, resolve_adapter, resolve_run,
                        validate_classifier_run, validate_reuse_data, write_handoff)


def _stage_placement(config, task):
    import torch
    world_size = config["run"].get("world_size", 1)
    if type(world_size) is not int or world_size not in (1, 2):
        raise ValueError("run.world_size must be 1 or 2")
    device = config["run"]["device"]
    if device not in ("cuda", "cpu"):
        raise ValueError("run.device must be cuda or cpu")
    if device == "cuda" and (not torch.cuda.is_available() or torch.cuda.device_count() < world_size):
        raise RuntimeError(f"{task} requires {world_size} visible CUDA GPUs")
    batch = config["training"]["args"]["per_device_train_batch_size"] * world_size
    effective = config["training"]["effective_batch_size"]
    if effective < batch or effective % batch:
        raise ValueError("effective_batch_size must be a positive multiple of microbatch * world_size")
    if task == "classification":
        from .classification import placement_preflight
        return [placement_preflight(config["model"], f"cuda:{rank}" if device == "cuda" else "cpu")
                for rank in range(world_size)]
    if world_size > 1 and config["model"].get("device_map", "single") != "single":
        raise ValueError("DDP correction requires single-device replicas, not balanced model splitting")
    from .correction import placement_preflight
    return [placement_preflight(config["model"], device, local_rank=rank) for rank in range(world_size)]


def preflight(classification, correction=None, original=None, bundle=None, *, mode="both",
              classifier_run=None, initial_adapter=None):
    """Check data/reuse first, then placement for only the stages being trained."""
    if mode not in ("task1", "task2", "both"):
        raise ValueError("mode must be task1, task2, or both")
    run_corrector = mode != "task1"
    fresh_classifier = classifier_run is None
    if mode == "task2" and fresh_classifier:
        raise ValueError("Task2 requires --classifier-run or --prior-run with a completed Task1 and OOF hints")
    if mode == "task1" and initial_adapter is not None:
        raise ValueError("--initial-adapter continues Task2; use task2 or both")
    report = {"mode": mode, "classifier_reused": not fresh_classifier}
    if fresh_classifier:
        from .classification import _connected_folds, _correction_split
        from .data import load_classifier_split, load_records, split_records
        seed, fraction = classification["run"]["seed"], classification["data"]["validation_fraction"]
        original_train, validation = split_records(load_records(original), fraction, seed)
        mixed_config = copy.deepcopy(classification)
        mixed_config["data"].update(path=classification["data"].get("augmentation_path"),
                                    filename=classification["data"]["augmentation_filename"])
        mixed_path = resolve_data(mixed_config)
        classifier_train, classifier_info = load_classifier_split(original, mixed_path, fraction, seed)
        oof = classification.get("oof", {}).get("enabled", False)
        if run_corrector and not oof:
            raise ValueError("Fresh two-stage runs require oof.enabled=true for leakage-safe correction hints")
        report.update(seed=seed, classification_train=len(classifier_train), validation=len(validation),
                      original_data=str(original), classification_data=str(mixed_path), hints_enabled=bool(oof))
        if oof:
            resolved, train, holdout, provenance = _correction_split(
                classification, Path(original), original_train, validation, bundle)
            epochs = classification["oof"]["epochs"]
            if type(epochs) is not int or epochs < 1:
                raise ValueError("oof.epochs must be a positive integer")
            grouping = {"train_rows": provenance["train_rows"] + classifier_info["provenance"]["train_rows"]}
            _connected_folds(original_train, train, grouping, classification["oof"]["folds"], seed, classifier_train)
            report.update(correction_train=len(train), correction_data=str(resolved))
            if run_corrector and (seed != correction["run"]["seed"]
                                  or fraction != correction["data"]["validation_fraction"]):
                raise ValueError("Both task configs must use the same seed and validation_fraction")
    else:
        saved = validate_classifier_run(classifier_run, require_hints=run_corrector)
        if run_corrector:
            report.update(validate_reuse_data(classifier_run, correction, bundle, original_path=original))
        else:
            report["validation"] = len(json.loads((Path(classifier_run) / "validation.json").read_text()))
        report["classifier_run"] = str(classifier_run)
        report["classification_train"] = json.loads((Path(classifier_run) / "run_manifest.json").read_text())["train_count"]
        if run_corrector and (saved["run"]["seed"] != correction["run"]["seed"]
                              or saved["data"]["validation_fraction"] != correction["data"]["validation_fraction"]):
            raise ValueError("Saved Task1 and correction configs must use the same seed and validation_fraction")
    if initial_adapter is not None:
        from .correction import validate_initial_adapter
        validate_initial_adapter(initial_adapter, correction)
        report["initial_adapter"] = str(initial_adapter)
    if fresh_classifier:
        report["classification_replicas"] = _stage_placement(classification, "classification")
    if run_corrector:
        report["correction_replicas"] = _stage_placement(correction, "correction")
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
    parser.add_argument("--mode", choices=("task1", "task2", "both"), default="both")
    parser.add_argument("--classification-config", default="configs/baseline.json")
    parser.add_argument("--correction-config", default="configs/correction.json")
    parser.add_argument("--classification-data")
    parser.add_argument("--correction-data")
    reuse = parser.add_mutually_exclusive_group()
    reuse.add_argument("--classifier-run", help="Completed classifier stage or enclosing portable/legacy saved output")
    reuse.add_argument("--prior-run", help="Extracted previous-version Kaggle output or unique enclosing input mount")
    parser.add_argument("--initial-adapter", help="Compatible saved correction adapter (or enclosing saved output) for continued fine-tuning")
    parser.add_argument("--output-dir")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    prior = args.classifier_run or args.prior_run
    if args.mode == "task2" and prior is None:
        parser.error("--mode task2 requires --classifier-run or --prior-run; Task1 is never trained in task2 mode")
    if args.mode == "task1" and args.initial_adapter:
        parser.error("--initial-adapter continues Task2; use task2 or both")
    classifier_run = resolve_run(prior) if prior else None
    classifier = (validate_classifier_run(classifier_run, require_hints=args.mode != "task1") if classifier_run else
                  load_config(args.classification_config, "classification"))
    corrector = load_config(args.correction_config, "correction") if args.mode != "task1" else None
    original = resolve_data(classifier, args.classification_data) if classifier_run is None else (
        resolve_data(classifier, args.classification_data) if args.classification_data else None)
    needs_bundle = corrector is not None or (classifier_run is None and classifier.get("oof", {}).get("enabled", False))
    if corrector is not None:
        bundle = resolve_data(corrector, args.correction_data)
    elif needs_bundle:
        hints_config = copy.deepcopy(classifier)
        hints_config["data"].update(path=classifier["data"].get("correction_path"),
                                    filename=classifier["data"].get("correction_filename", "VGEC_balanced.json"))
        bundle = resolve_data(hints_config, args.correction_data)
    else:
        bundle = None
    initial_adapter = resolve_adapter(args.initial_adapter) if args.initial_adapter else None
    preflight(classifier, corrector, original, bundle, mode=args.mode,
              classifier_run=classifier_run, initial_adapter=initial_adapter)
    if args.preflight_only:
        return
    output_config = corrector or classifier
    output = stage_output({"run": {"name": f"sailor2-{args.mode}",
                                    "output_root": output_config["run"]["output_root"]}}, args.output_dir)
    input_roots = [source for source in (classifier_run, initial_adapter) if source is not None]
    for argument in (prior, args.initial_adapter):
        if argument:
            source = Path(argument).expanduser().resolve()
            input_roots.append(source.parent if source.is_file() else source)
    for source in input_roots:
        if output == source or output.is_relative_to(source):
            raise ValueError("Output directory must not be inside read-only reused input artifacts")
    output.mkdir(parents=True, exist_ok=True)
    from tools.runtime import source_identity
    write_json(output / "source_manifest.json", source_identity(PROJECT_ROOT))
    write_json(output / "classification_config.json", classifier)
    if corrector is not None:
        write_json(output / "correction_config.json", corrector)
    packaged_classifier = output / "classification"
    if classifier_run is None:
        command = stage_command("model.classification", classifier) + [
            "--config", str(Path(args.classification_config).resolve()), "--data", str(original),
            "--output-dir", str(packaged_classifier)]
        if bundle is not None:
            command += ["--correction-data", str(bundle)]
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    else:
        copy_classifier_run(classifier_run, packaged_classifier)
    # The first completed stage is already reusable if a later stage fails.
    write_handoff(output)
    if corrector is not None:
        validate_classifier_run(packaged_classifier)
        validate_reuse_data(packaged_classifier, corrector, bundle)
        command = stage_command("model.correction", corrector) + [
            "--config", str(Path(args.correction_config).resolve()), "--data", str(bundle),
            "--classifier-run", str(packaged_classifier), "--output-dir", str(output / "correction")]
        if initial_adapter is not None:
            command += ["--initial-adapter", str(initial_adapter)]
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        write_handoff(output, correction=True)
    manifest = {"mode": args.mode, "classifier": "classification/classifier",
                "seed": classifier["run"]["seed"], "classifier_reused": classifier_run is not None,
                "classification_metrics": json.loads((packaged_classifier / "metrics.json").read_text())}
    if corrector is not None:
        manifest.update(corrector="correction/adapter",
                        correction_metrics=json.loads((output / "correction" / "metrics.json").read_text()))
    write_json(output / "pipeline_manifest.json", manifest)
    if output_config["run"].get("archive", True):
        archive = shutil.make_archive(str(output) + "-artifacts", "zip", output.parent, output.name)
        print(f"Artifacts: {archive}", flush=True)
    print(f"Completed {args.mode}: {output}", flush=True)


if __name__ == "__main__":
    main()
