"""Two quality-data fits and four controlled, separately exported pairings."""

from contextlib import ExitStack
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile

from model.data import _id_key, load_records
from model.prediction_export import read_submission_json
from model.runtime import PROJECT_ROOT, load_config, resolve_output, write_json
from tools.package_focused_training import sha256
from tools.run_focused import materialize_package, bundle_selected, merge_export, stage_timeout


def read(path):
    return json.loads(Path(path).read_text())


def validate(plan):
    if (plan.get("format") != "vgec-quality-trial-v1"
            or not 60 <= plan["max_stage_seconds"] <= plan["max_wall_seconds"] <= 43200):
        raise ValueError("Invalid quality-trial plan/budget")
    recipe = read(PROJECT_ROOT / plan["training_recipe"])
    config = load_config(PROJECT_ROOT / recipe["classification_config"], "classification")
    if (recipe["mode"] != "both" or recipe["classifier_run"] or recipe["initial_adapter"]
            or config.get("competition_id") != 1039 or config.get("ensemble")
            or config.get("oof", {}).get("enabled", False)
            or config["run"]["seed"] != 42 or config["data"]["split_seed"] != 42
            or config["data"]["validation_fraction"] != 0.2
            or config["training"]["args"]["num_train_epochs"] < 5
            or config["training"]["args"].get("max_steps", -1) != -1):
        raise ValueError("Quality trial requires one fresh1039 classifier, split42 and five full epochs")
    correction = load_config(PROJECT_ROOT / recipe["correction_config"], "correction")
    if (correction.get("competition_id") != 1039
            or correction["run"]["seed"] != 42
            or correction["data"]["split_seed"] != 42
            or correction["data"]["validation_fraction"] != 0.2
            or correction["training"]["args"]["num_train_epochs"] < 5
            or correction["training"]["args"].get("max_steps", -1) != -1
            or correction["inference"]["hint_mode"] != "none"):
        raise ValueError("Quality trial requires a fresh no-hint1039 corrector and five full epochs")
    return recipe


def export_frozen_corrector(records, classes, control, output):
    """Change only predictions of class; correction text never enters fitting."""
    corrections = read_submission_json(control, records)
    merge_export(records, classes, corrections, output)


def run(plan, input_root, output_root, *, device="cuda", preflight_only=False,
        source_identity=None):
    from model.two_phase import _run_stage
    from model.reproducibility import process_environment
    from model.tracking import TrackingRun
    from tools.run_experiment import prepare_run
    from tools.experiment_log import record

    recipe = validate(plan)
    work = resolve_output({"run": {"name": plan["name"], "output_root": str(output_root)}})
    output = work / "run"
    output.mkdir(parents=True)
    print("VGEC_RUN_DIR:" + str(output), flush=True)
    proof = read(PROJECT_ROOT / plan["package_proof"])
    package, packet = materialize_package(input_root, work / "package", recipe["data_archive"], proof["manifest_sha256"],
                                          legacy_manifest_sha256=proof.get("legacy_manifest_sha256"),
                                          legacy_control_json_sha256=proof.get("legacy_control_json_sha256"))
    if packet["data_sha256"] != recipe["data_sha256"]:
        raise ValueError("Quality training hashes differ from recipe")
    control_info = packet.get("control_submission", {})
    if (control_info.get("path") != "baseline/quality_control/results.json"
            or control_info.get("validation_predictions") != "baseline/quality_control_validation.json"
            or control_info.get("input_sha256") != plan["inference_sha256"]
            or sha256(package / "private_test.json") != plan["inference_sha256"]):
        raise ValueError("Frozen control requires the exact committed private inference input")
    private = load_records(package / "private_test.json", require_labels=False)
    if any(r["label"] is not None or r["target"] is not None for r in private):
        raise ValueError("Private input must remain inference-only")
    control = package / control_info["path"]
    read_submission_json(control, private)
    destination = output / "quality_control"
    destination.mkdir()
    shutil.copy2(control, destination / "results.json")
    with zipfile.ZipFile(destination / "results.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(destination / "results.json", "results.json")
    training, command = prepare_run(PROJECT_ROOT / plan["training_recipe"], package,
                                    work / "training", world_size=1, device=device)
    write_json(work / "experiment_manifest.json", {"plan": plan, "source": source_identity,
               "package_manifest_sha256": sha256(package / "focused_package_manifest.json"), "training": str(training),
               "model_parameters": read(training / "experiment_manifest.json")["model_parameters"]})
    config = read(training / "classification_config.json")
    environment = {**process_environment(42), "VGEC_CUDA_STAGE": str(int(device == "cuda")),
                   "WANDB_RUN_GROUP": work.name, "VGEC_FORWARD_EVENTS": "1", "VGEC_PIPELINE_TRACKING": "1"}
    started = time.monotonic()
    record(work, "started", kind="quality-trial")
    with TrackingRun() as tracking, ExitStack() as resources:
        if device == "cuda":
            from model.weight_cache import temporary_cache_environment
            environment.update(temporary_cache_environment(resources))
        tracking.start(config, output, "pipeline", group=work.name,
                       metadata={"kind": "quality data; independent classifier and corrector challengers", "competition_id": 1039})

        def stage(args, name):
            _run_stage(args, {**environment, "VGEC_STAGE_NAME": name,
                             "VGEC_ROOT_RUN_NAME": "quality-training"}, tracking,
                       timeout_seconds=stage_timeout(plan, time.monotonic() - started, name))

        try:
            stage(command + ["--preflight-only"], "quality-preflight")
            if preflight_only:
                write_json(output / "preflight.json", {"status": "passed", "control_verified": True})
                record(work, "preflight_passed", kind="quality-trial")
                return output
            stage(command + ["--defer-submission"], "quality-training")
            classifier = training / "run/classification"
            from model.run_reuse import validate_classifier_run
            validate_classifier_run(classifier, require_hints=False)
            corrector = package / "baseline/correction"
            gold = read(classifier / "validation.json")
            if gold != read(corrector / "validation.json"):
                raise ValueError("Classifier and preserved corrector development rows differ")
            from model.metrics import _load_predictions, experiment_report
            labels = {_id_key(r["id"]): r["predicted_class"] for r in read(classifier / "predictions.json")}
            predictions = [{**r, "label": labels[_id_key(r["id"])]}
                           for r in _load_predictions(corrector / "predictions.json")]
            metrics = experiment_report(gold, predictions)
            strategy = {"name": "greedy_reference", "generation": {}}
            selected = {"competition_id": 1039, "classifier_source": "challenger", "corrector_source": "quality_baseline",
                        "decision_rule": "argmax", "strategy": strategy,
                        "scope": "Unpromoted classifier experiment, not a proven private winner; quality_control is preserved."}
            selected.update(bundle_selected(output, classifier, corrector / "adapter", strategy))
            write_json(output / "selected.json", selected)
            write_json(output / "selected_metrics.json", metrics)
            write_json(output / "selected_validation_predictions.json", predictions)
            stage([sys.executable, "-m", "tools.focused_worker", "--classifier", str(classifier / "classifier"),
                   "--records", str(package / "private_test.json"), "--output", str(output / "private_classification.json"),
                   "--device", device], "candidate-private-classifier")
            export_frozen_corrector(private, read(output / "private_classification.json"), control, output)
            challenger_corrector = training / "run/correction"
            receipt = read(challenger_corrector / "run_manifest.json")
            if not receipt.get("saved_adapter_reload_verified") or gold != read(challenger_corrector / "validation.json"):
                raise ValueError("Challenger corrector must reload and share the development rows")
            new_predictions = [{**r, "label": labels[_id_key(r["id"])]}
                               for r in _load_predictions(challenger_corrector / "predictions.json")]
            new_metrics = experiment_report(gold, new_predictions)
            control_validation = read(package / control_info["validation_predictions"])
            control_labels = {_id_key(r["id"]): r["label"] for r in control_validation}
            correction_only_validation = [{**r, "label": control_labels[_id_key(r["id"])]}
                                          for r in _load_predictions(challenger_corrector / "predictions.json")]
            comparison = {
                "scope": "Same reused development rows; no automatic private winner promotion",
                "quality_control": experiment_report(gold, control_validation),
                "classifier_challenger": metrics,
                "corrector_challenger": experiment_report(gold, correction_only_validation),
                "both_challengers": new_metrics,
            }
            write_json(output / "comparison_metrics.json", comparison)
            both = output / "both_challengers"
            both.mkdir()
            both_selection = {**selected, "corrector_source": "challenger",
                              "scope": "Unpromoted20B/14B experiment; private quality unknown"}
            both_selection.update(bundle_selected(both, classifier, challenger_corrector / "adapter", strategy))
            write_json(both / "selected.json", both_selection)
            write_json(both / "selected_metrics.json", new_metrics)
            write_json(both / "selected_validation_predictions.json", new_predictions)
            write_json(output / "private_strategy.json", strategy)
            stage([sys.executable, "-m", "tools.decoding_sweep", "--adapter", str(challenger_corrector / "adapter"),
                   "--classifier-run", str(classifier), "--output-dir", str(output / "private_decoding"),
                   "--device", device, "--public-data", str(package / "private_test.json"),
                   "--strategy", str(output / "private_strategy.json")], "selected-private-corrector")
            corrections = read(output / "private_decoding/predictions.json")
            merge_export(private, read(output / "private_classification.json"), corrections, both)
            correction_only = output / "corrector_challenger"
            correction_only.mkdir()
            write_json(correction_only / "selected_metrics.json", comparison["corrector_challenger"])
            write_json(correction_only / "selected_validation_predictions.json", correction_only_validation)
            control_predictions = read_submission_json(control, private)
            merge_export(private, control_predictions, corrections, correction_only)
            write_json(output / "submission_candidates.json", {
                "quality_control": {"path": "quality_control/results.zip", "sha256": sha256(destination / "results.zip"),
                                    "scope": "Exact original quality-system output; report0.5806 was on1038, same-track1039 score unverified"},
                "classifier_challenger": {"path": "results.zip", "sha256": sha256(output / "results.zip"),
                                         "scope": "New classifier only, exact quality-corrector outputs; private quality unknown"},
                "corrector_challenger": {"path": "corrector_challenger/results.zip", "sha256": sha256(correction_only / "results.zip"),
                                         "scope": "Exact quality-system classifier labels plus new14B corrections; private quality unknown"},
                "both_challengers": {"path": "both_challengers/results.zip", "sha256": sha256(both / "results.zip"),
                                     "scope": "New20B classifier and new14B corrector; private quality unknown"}})
            write_json(output / "results.evaluation.json", {"evaluated": False, "status": "completed",
                       "count": len(private), "input_scope": "private_inference_only", "input_sha256": plan["inference_sha256"],
                       "correction_source": "frozen quality-system predictions on this exact input"})
            write_json(output / "pipeline_manifest.json", {"status": "completed", "competition_id": 1039,
                       "mode": "quality_trial", "experiment_kind": "quality-trial", "new_classifier_fits": 1, "new_corrector_fits": 1,
                       "selected_metrics": metrics, "candidates": "submission_candidates.json", "submission": {"status": "completed"}})
            tracking.complete("quality-trial", metrics)
            record(work, "completed", kind="quality-trial")
        except BaseException as error:
            record(work, "failed", kind="quality-trial", error_type=type(error).__name__)
            raise
    return output


def run_from_args(args, source_identity):
    wrapper = read(args.recipe)
    plan = read(PROJECT_ROOT / wrapper["plan"])
    kaggle = Path("/kaggle/working").is_dir()
    return run(plan, Path(args.input_root) if args.input_root else Path("/kaggle/input") if kaggle else PROJECT_ROOT / "dataset",
               Path(args.output_root) if args.output_root else Path("/kaggle/working/runs") if kaggle else PROJECT_ROOT / "runs",
               preflight_only=args.preflight_only, source_identity=source_identity)
