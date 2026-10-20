"""Fresh five-epoch fits and one private export, without controller deadlines."""
from contextlib import ExitStack
import json
from pathlib import Path
import shutil
import sys
import traceback

from model.runtime import PROJECT_ROOT, resolve_output, write_json
from tools.run_focused import materialize_package, merge_export


def read(path):
    return json.loads(Path(path).read_text())


def selected_classifier(output):
    """Publish selected-epoch results without a redundant saved-model evaluation."""
    receipt = read(output / "training_complete.json")
    history = read(output / "epoch_history.json")
    selected = next(row for row in history if row["epoch"] == receipt["best_epoch"])
    metrics = {"count": receipt["validation_count"], "classification": selected["classification"],
               "evaluation_status": "selected_epoch", "training": receipt}
    shutil.copy2(output / "selected_epoch_predictions.json", output / "predictions.json")
    write_json(output / "metrics.json", metrics)
    write_json(output / "run_manifest.json", {**receipt, "verification_status": "skipped_user_request"})
    return metrics


def run(plan, input_root, output_root, *, source_identity=None):
    from model.data import load_records
    from model.two_phase import _run_stage
    from model.reproducibility import process_environment
    from model.tracking import TrackingRun
    from model.metrics import _load_predictions, experiment_report
    from model.run_reuse import write_handoff
    from tools.run_experiment import prepare_run
    from tools.experiment_log import record

    recipe = read(PROJECT_ROOT / plan["training_recipe"])
    work = resolve_output({"run": {"name": plan["name"], "output_root": str(output_root)}})
    output = work / "run"
    output.mkdir(parents=True)
    print("VGEC_RUN_DIR:" + str(output), flush=True)
    proof = read(PROJECT_ROOT / plan["package_proof"])
    archive_name = recipe["data_archive"]
    previous_archive = proof.get("previous_archive_name")
    if (previous_archive and not any(Path(input_root).rglob(archive_name))
            and any(Path(input_root).rglob(previous_archive))):
        archive_name = previous_archive
    package, packet = materialize_package(
        input_root, work / "package", archive_name, proof["manifest_sha256"],
        legacy_manifest_sha256=proof.get("legacy_manifest_sha256"),
        legacy_control_json_sha256=proof.get("legacy_control_json_sha256"),
        additional_manifest_sha256=proof.get("compatible_manifest_sha256", []))
    if packet["data_sha256"] != recipe["data_sha256"]:
        raise ValueError("Training payload differs from the committed quality dataset")
    prepared, _ = prepare_run(PROJECT_ROOT / plan["training_recipe"], package,
                              work / "training", world_size=1, device="cuda")
    inputs = read(prepared / "experiment_manifest.json")["inputs"]
    private_path = Path(inputs["submission"]["path"])
    if inputs["submission"]["sha256"] != plan["inference_sha256"]:
        raise ValueError("Private inference input differs from committed input")
    private = load_records(private_path, require_labels=False)
    if any(row["label"] is not None or row["target"] is not None for row in private):
        raise ValueError("Private data must remain inference-only")
    classifier = output / "classification"
    corrector = output / "correction"
    config = read(prepared / "classification_config.json")
    environment = {**process_environment(42), "VGEC_CUDA_STAGE": "1", "WANDB_RUN_GROUP": work.name,
                   "VGEC_FORWARD_EVENTS": "1", "VGEC_PIPELINE_TRACKING": "1"}
    write_json(work / "experiment_manifest.json", {
        "plan": plan, "source": source_identity, "training": str(prepared),
        "controller_deadline": None, "verification": "skipped_user_request",
        "model_parameters": read(prepared / "experiment_manifest.json")["model_parameters"]})
    record(work, "started", kind="fast-training")
    with TrackingRun() as tracking, ExitStack() as resources:
        from model.weight_cache import temporary_cache_environment
        environment.update(temporary_cache_environment(resources))
        for item in inputs.values():
            tracking.protect_console_file(item["path"])
        tracking.start(config, output, "pipeline", group=work.name,
                       metadata={"competition_id": 1039, "kind": "fresh five-epoch quality-data retrain"})

        def stage(command, name):
            # No wall/stage timeout. Kaggle's external session limit still applies.
            message = "Starting stage: " + name
            print(message, flush=True)
            tracking.console_line(message)
            _run_stage(command, {**environment, "VGEC_STAGE_NAME": name}, tracking)
            message = "Finished stage: " + name
            print(message, flush=True)
            tracking.console_line(message)

        try:
            stage([sys.executable, "-m", "model.classification", "--config",
                   str(prepared / "classification_config.json"), "--output-dir", str(classifier),
                   "--device", "cuda", "--fit-only"], "classifier-training")
            classifier_metrics = selected_classifier(classifier)
            tracking.evaluation(classifier_metrics, "selected/classifier")
            # Classify now, before switching the owned base cache from20B to14B.
            stage([sys.executable, "-m", "tools.focused_worker", "--classifier", str(classifier / "classifier"),
                   "--records", str(private_path), "--output", str(output / "private_classification.json"),
                   "--batch-size", "8"], "private-classifier")
            stage([sys.executable, "-m", "model.correction", "--config",
                   str(prepared / "correction_config.json"), "--classifier-run", str(classifier),
                   "--output-dir", str(corrector), "--device", "cuda"], "corrector-training")
            write_handoff(output, correction=True)
            stage([sys.executable, "-m", "tools.fast_correct", "--adapter", str(corrector / "adapter"),
                   "--records", str(private_path), "--output-dir", str(output / "private_decoding"),
                   "--batch-size", "4"], "private-corrector")
            corrections = read(output / "private_decoding/predictions.json")
            merge_export(private, read(output / "private_classification.json"), corrections, output)
            # These are already-computed epoch results; no model replay is run.
            metrics = experiment_report(read(corrector / "validation.json"),
                                        _load_predictions(corrector / "predictions.json"))
            write_json(output / "selected_metrics.json", metrics)
            write_json(output / "results.evaluation.json", {
                "evaluated": False, "status": "completed", "count": len(private),
                "input_scope": "private_inference_only", "input_sha256": plan["inference_sha256"]})
            write_json(output / "pipeline_manifest.json", {
                "status": "completed", "mode": "both", "competition_id": 1039,
                "experiment_kind": "fast-training", "new_classifier_fits": 1, "new_corrector_fits": 1,
                "verification_status": "skipped_user_request", "controller_deadline": None,
                "classifier": "classification/classifier", "corrector": "correction/adapter",
                "selected_metrics": metrics, "submission": {"status": "completed", "path": "results.zip"}})
            message = "Submission ready: " + str(output / "results.zip")
            print(message, flush=True)
            tracking.console_line(message)
            tracking.complete("pipeline", metrics)
            record(work, "completed", kind="fast-training")
        except BaseException as error:
            tracking.console_line(traceback.format_exc())
            record(work, "failed", kind="fast-training", error_type=type(error).__name__)
            raise
    return output


def run_from_args(args, source_identity):
    if args.preflight_only:
        raise ValueError("This fresh-run recipe has no preflight mode")
    wrapper = read(args.recipe)
    plan = read(PROJECT_ROOT / wrapper["plan"])
    kaggle = Path("/kaggle/working").is_dir()
    return run(plan,
               Path(args.input_root) if args.input_root else Path("/kaggle/input") if kaggle else PROJECT_ROOT / "dataset",
               Path(args.output_root) if args.output_root else Path("/kaggle/working/runs") if kaggle else PROJECT_ROOT / "runs",
               source_identity=source_identity)
