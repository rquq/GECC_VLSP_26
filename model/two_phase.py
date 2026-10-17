"""Train Task1, Task2, or both, exporting portable completed-stage handoffs."""

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import queue
import signal
import subprocess
import sys
import threading
import time

from .runtime import (
    PROJECT_ROOT,
    load_config,
    resolve_data,
    resolve_submission_data,
    stage_output,
    write_json,
    split_seed,
)
from .run_reuse import (
    copy_classifier_run,
    resolve_adapter,
    resolve_run,
    validate_classifier_run,
    validate_reuse_data,
    write_handoff,
)
from .tracking import STAGE_EVENT, PROGRESS_EVENT, result_metrics, tracked_entrypoint

HEARTBEAT_SECONDS = 30


def _event(line, marker):
    position = line.find(marker)
    if position < 0:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(
            line[position + len(marker) :].lstrip()
        )
    except json.JSONDecodeError:
        return None
    return (
        value
        if isinstance(value, dict)
        and isinstance(value.get("metrics"), dict)
        and isinstance(value.get("stage"), str)
        else None
    )


def _stage_placement(config, task):
    import torch

    world_size = config["run"].get("world_size", 1)
    if type(world_size) is not int or world_size not in (1, 2):
        raise ValueError("run.world_size must be 1 or 2")
    device = config["run"]["device"]
    if device not in ("cuda", "cpu"):
        raise ValueError("run.device must be cuda or cpu")
    if device == "cuda" and (
        not torch.cuda.is_available() or torch.cuda.device_count() < world_size
    ):
        raise RuntimeError(f"{task} requires {world_size} visible CUDA GPUs")
    batch = config["training"]["args"]["per_device_train_batch_size"] * world_size
    effective = config["training"]["effective_batch_size"]
    if effective < batch or effective % batch:
        raise ValueError(
            "effective_batch_size must be a positive multiple of microbatch * world_size"
        )
    if task == "classification":
        from .classification import (
            placement_preflight,
            _load_tokenizer,
            _configure_padding,
        )

        tokenizer = _load_tokenizer(
            config["model"]["name"], revision=config["model"]["revision"]
        )
        _configure_padding(tokenizer)
        del tokenizer

        return [
            placement_preflight(
                config["model"], f"cuda:{rank}" if device == "cuda" else "cpu"
            )
            for rank in range(world_size)
        ]
    if world_size > 1 and config["model"].get("device_map", "single") != "single":
        raise ValueError(
            "DDP correction requires single-device replicas, not balanced model splitting"
        )
    from .correction import placement_preflight

    return [
        placement_preflight(config["model"], device, local_rank=rank)
        for rank in range(world_size)
    ]


def preflight(
    classification,
    correction=None,
    original=None,
    bundle=None,
    *,
    mode="both",
    classifier_run=None,
    initial_adapter=None,
    check_placement=True,
    resume_root=None,
):
    """Check data/reuse first, then placement for only the stages being trained."""
    if mode not in ("task1", "task2", "both"):
        raise ValueError("mode must be task1, task2, or both")
    run_corrector = mode != "task1"
    fresh_classifier = classifier_run is None
    predicted_hints = (
        run_corrector
        and correction.get("inference", {}).get("hint_mode", "predicted") == "predicted"
    )
    if mode == "task2" and fresh_classifier:
        raise ValueError(
            "Task2 requires --classifier-run or --prior-run with a completed Task1"
        )
    if mode == "task1" and initial_adapter is not None:
        raise ValueError("--initial-adapter continues Task2; use task2 or both")
    report = {"mode": mode, "classifier_reused": not fresh_classifier}
    resumed = {}
    if fresh_classifier:
        from .classification import (
            _connected_folds,
            _correction_split,
            _organizer_partition,
        )
        from .data import load_classifier_split, load_records, split_records

        seed, fraction = (
            split_seed(classification),
            classification["data"]["validation_fraction"],
        )
        mixed_config = copy.deepcopy(classification)
        mixed_config["data"].update(
            path=classification["data"].get("augmentation_path"),
            filename=classification["data"]["augmentation_filename"],
        )
        mixed_path = resolve_data(mixed_config)
        classifier_train, classifier_info = load_classifier_split(
            original, mixed_path, fraction, seed
        )
        original_train, validation = _organizer_partition(
            original, fraction, seed, classifier_info
        )
        if resume_root is not None:
            from .classifier_resume import plan

            resumed = plan(resume_root, classification, original, mixed_path)
            report["classifier_reused_members"] = list(resumed)
        oof = classification.get("oof", {}).get("enabled", False)
        if predicted_hints and not oof:
            raise ValueError(
                "Fresh two-stage runs require oof.enabled=true for leakage-safe correction hints"
            )
        report.update(
            seed=seed,
            classification_train=len(classifier_train),
            validation=len(validation),
            original_data=str(original),
            classification_data=str(mixed_path),
            hints_enabled=bool(oof),
        )
        if fraction == 0 and oof:
            raise ValueError("Final fits cannot use OOF hints")
        if oof:
            resolved, train, holdout, provenance = _correction_split(
                classification, Path(original), original_train, validation, bundle
            )
            epochs = classification["oof"]["epochs"]
            if type(epochs) is not int or epochs < 1:
                raise ValueError("oof.epochs must be a positive integer")
            grouping = {
                "train_rows": provenance["train_rows"]
                + classifier_info["provenance"]["train_rows"]
            }
            _connected_folds(
                original_train,
                train,
                grouping,
                classification["oof"]["folds"],
                seed,
                classifier_train,
            )
            report.update(correction_train=len(train), correction_data=str(resolved))
        if run_corrector:
            if (
                seed != split_seed(correction)
                or fraction != correction["data"]["validation_fraction"]
            ):
                raise ValueError(
                    "Both task configs must use the same seed and validation_fraction"
                )
            if not oof:
                from .data import load_training_split

                train, holdout, _ = load_training_split(bundle, fraction, seed)
                if holdout != validation:
                    raise ValueError(
                        "Classifier and corrector validation splits differ"
                    )
                report.update(
                    correction_train=len(train),
                    correction_data=str(bundle),
                    hint_mode="none",
                )
    else:
        saved = validate_classifier_run(classifier_run, require_hints=predicted_hints)
        if run_corrector:
            report.update(
                validate_reuse_data(
                    classifier_run, correction, bundle, original_path=original
                )
            )
        else:
            report["validation"] = len(
                json.loads((Path(classifier_run) / "validation.json").read_text())
            )
        report["classifier_run"] = str(classifier_run)
        report["classification_train"] = json.loads(
            (Path(classifier_run) / "run_manifest.json").read_text()
        )["train_count"]
        if run_corrector and (
            saved["run"]["seed"] != correction["run"]["seed"]
            or saved["data"]["validation_fraction"]
            != correction["data"]["validation_fraction"]
        ):
            raise ValueError(
                "Saved Task1 and correction configs must use the same seed and validation_fraction"
            )
    if initial_adapter is not None:
        from .correction import validate_initial_adapter

        validate_initial_adapter(initial_adapter, correction)
        report["initial_adapter"] = str(initial_adapter)
    if fresh_classifier and check_placement:
        entries = classification.get("ensemble", {}).get("members", [])
        if entries:
            from .runtime import classifier_member_config
            report["classification_models"] = {
                entry["name"]: _stage_placement(
                    classifier_member_config(classification, entry), "classification"
                )
                for entry in entries
                if entry["name"] not in resumed
            }
        else:
            report["classification_replicas"] = _stage_placement(
                classification, "classification"
            )
    if run_corrector and check_placement:
        report["correction_replicas"] = _stage_placement(correction, "correction")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def stage_command(module, config):
    world_size = config["run"].get("world_size", 1)
    if world_size > 1:
        return [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            str(world_size),
            "-m",
            module,
        ]
    return [sys.executable, "-m", module]


def _run_stage(command, env, tracking, timeout_seconds=None):
    """Reap each process group; keep the root alive even during silent loading."""
    from .reproducibility import process_environment

    env = process_environment(int(env.get("VGEC_MODEL_SEED", "42")), env)
    deadline = (
        time.monotonic() + timeout_seconds if timeout_seconds is not None else None
    )
    stage = env.get("VGEC_STAGE_NAME", "starting")
    gpu = env.get("VGEC_CUDA_STAGE") == "1"
    before = None
    if gpu:
        from .gpu_lifecycle import snapshot

        before = snapshot()
    tracking.heartbeat(stage=stage)
    lines = queue.Queue()

    def read_lines(stream):
        try:
            for line in stream:
                lines.put(line)
        finally:
            lines.put(None)

    with subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    ) as process:
        reader = threading.Thread(
            target=read_lines, args=(process.stdout,), daemon=True
        )
        reader.start()
        next_heartbeat = time.monotonic() + HEARTBEAT_SECONDS
        try:
            while True:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Stage {stage} exceeded its committed time budget"
                    )
                remaining = max(0.01, next_heartbeat - time.monotonic())
                if deadline is not None:
                    remaining = min(remaining, max(0.01, deadline - time.monotonic()))
                try:
                    line = lines.get(timeout=remaining)
                except queue.Empty:
                    line = ""
                if time.monotonic() >= next_heartbeat:
                    tracking.heartbeat()
                    next_heartbeat = time.monotonic() + HEARTBEAT_SECONDS
                if line is None:
                    break
                result, progress = _event(line, STAGE_EVENT), _event(
                    line, PROGRESS_EVENT
                )
                if result is not None:
                    event = result
                    tracking.stage_result(event)
                    if env.get("VGEC_FORWARD_EVENTS") == "1":
                        print(STAGE_EVENT + json.dumps(event), flush=True)
                    print(
                        f"Completed {event['stage']}: {json.dumps(event['metrics'])}",
                        flush=True,
                    )
                elif progress is not None:
                    tracking.progress(progress)
                    if env.get("VGEC_FORWARD_EVENTS") == "1":
                        print(PROGRESS_EVENT + json.dumps(progress), flush=True)
                else:
                    print(line, end="", flush=True)
            returncode = process.wait()
        except BaseException:
            from tools.process_groups import descendant_groups, signal_groups

            groups = descendant_groups(process.pid) | {process.pid}
            try:
                signal_groups(groups, signal.SIGTERM)
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                signal_groups(groups, signal.SIGKILL)
                process.wait()
            except ProcessLookupError:
                process.wait()
            finally:
                signal_groups(groups - {process.pid}, signal.SIGKILL)
            raise
        finally:
            reader.join(timeout=2)
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)
    if before is not None:
        from .gpu_lifecycle import wait_for_release

        try:
            after = wait_for_release(before)
        except RuntimeError:
            write_json(
                tracking.output / "gpu_transition_failure.json",
                {"stage": stage, "before": before},
            )
            raise
        with (tracking.output / "gpu_transitions.jsonl").open("a") as log:
            log.write(
                json.dumps(
                    {
                        "stage": stage,
                        "before": before,
                        "after": after,
                        "child_exited": True,
                    }
                )
                + "\n"
            )
    tracking.heartbeat()


@tracked_entrypoint
def main(*, tracking):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("task1", "task2", "both"), default="both")
    parser.add_argument("--classification-config", default="configs/baseline.json")
    parser.add_argument("--correction-config", default="configs/correction.json")
    parser.add_argument("--classification-data")
    parser.add_argument("--correction-data")
    reuse = parser.add_mutually_exclusive_group()
    reuse.add_argument(
        "--classifier-run",
        help="Completed classifier stage or enclosing portable/legacy saved output",
    )
    reuse.add_argument(
        "--prior-run",
        help="Extracted previous-version Kaggle output or unique enclosing input mount",
    )
    parser.add_argument(
        "--initial-adapter",
        help="Compatible saved correction adapter (or enclosing saved output) for continued fine-tuning",
    )
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--submission-data",
        help="Raw organizer ID/Input JSON or auto to discover public_test.json; otherwise export untouched validation",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--defer-submission",
        action="store_true",
        help="Internal tuning coordinator: save completed models/metrics; parent exports after selection",
    )
    args = parser.parse_args()
    prior = args.classifier_run or args.prior_run
    if args.mode == "task2" and prior is None:
        parser.error(
            "--mode task2 requires --classifier-run or --prior-run; Task1 is never trained in task2 mode"
        )
    if args.mode == "task1" and args.initial_adapter:
        parser.error("--initial-adapter continues Task2; use task2 or both")
    if args.mode == "task1" and args.submission_data:
        parser.error(
            "--submission-data requires Task2; Task1 has no corrector and cannot produce a full GEC submission"
        )
    classifier_run = resolve_run(prior) if prior else None
    corrector = (
        load_config(args.correction_config, "correction")
        if args.mode != "task1"
        else None
    )
    predicted_hints = (
        corrector is not None
        and corrector.get("inference", {}).get("hint_mode", "predicted") == "predicted"
    )
    classifier = (
        validate_classifier_run(classifier_run, require_hints=predicted_hints)
        if classifier_run
        else load_config(args.classification_config, "classification")
    )
    submission_path = resolve_submission_data(
        args.submission_data,
        input_root=Path((corrector or classifier)["data"]["input_root"]),
    )
    submission_records = None
    if submission_path is not None:
        from .data import load_records

        submission_records = load_records(submission_path, require_labels=False)
        if not submission_records:
            raise ValueError("Submission input must contain at least one record")
    original = (
        resolve_data(classifier, args.classification_data)
        if classifier_run is None
        else (
            resolve_data(classifier, args.classification_data)
            if args.classification_data
            else None
        )
    )
    needs_bundle = corrector is not None or (
        classifier_run is None and classifier.get("oof", {}).get("enabled", False)
    )
    if corrector is not None:
        bundle = resolve_data(corrector, args.correction_data)
    elif needs_bundle:
        hints_config = copy.deepcopy(classifier)
        hints_config["data"].update(
            path=classifier["data"].get("correction_path"),
            filename=classifier["data"].get(
                "correction_filename", "VGEC_balanced.json"
            ),
        )
        bundle = resolve_data(hints_config, args.correction_data)
    else:
        bundle = None
    initial_adapter = (
        resolve_adapter(args.initial_adapter) if args.initial_adapter else None
    )
    preflight(
        classifier,
        corrector,
        original,
        bundle,
        mode=args.mode,
        classifier_run=classifier_run,
        initial_adapter=initial_adapter,
        check_placement=args.preflight_only,
        resume_root=args.output_dir,
    )
    if args.preflight_only:
        return
    output_config = corrector or classifier
    from .classifier_resume import MANIFEST, resume_output

    if args.output_dir and (Path(args.output_dir) / MANIFEST).is_file():
        output = resume_output(args.output_dir)
    else:
        output = stage_output(
            {
                "run": {
                    "name": f"sailor-7b-{args.mode}",
                    "output_root": output_config["run"]["output_root"],
                }
            },
            args.output_dir,
        )
    input_roots = [
        source for source in (classifier_run, initial_adapter) if source is not None
    ]
    for argument in (prior, args.initial_adapter):
        if argument:
            source = Path(argument).expanduser().resolve()
            input_roots.append(source.parent if source.is_file() else source)
    for source in input_roots:
        if output == source or output.is_relative_to(source):
            raise ValueError(
                "Output directory must not be inside read-only reused input artifacts"
            )
    output.mkdir(parents=True, exist_ok=True)
    group = output.parent.name if output.name == "run" else output.name
    tracking.start(
        output_config,
        output,
        "pipeline",
        group=group,
        metadata={
            "mode": args.mode,
            "classifier_reused": classifier_run is not None,
            "initial_adapter_reused": initial_adapter is not None,
        },
    )
    child_env = os.environ.copy()
    child_env["WANDB_RUN_GROUP"] = tracking.identity["group"]
    child_env["VGEC_PIPELINE_TRACKING"] = "1"
    child_env["VGEC_CUDA_STAGE"] = (
        "1" if output_config["run"]["device"] == "cuda" else "0"
    )
    if classifier_run is not None:
        tracking.use_model(classifier_run / "classifier", "classifier")
    if initial_adapter is not None:
        tracking.use_model(initial_adapter, "corrector")
    from tools.runtime import source_identity

    write_json(output / "source_manifest.json", source_identity(PROJECT_ROOT))
    write_json(output / "classification_config.json", classifier)
    if corrector is not None:
        write_json(output / "correction_config.json", corrector)
    packaged_classifier = output / "classification"
    from .classifier_resume import plan

    resumed = (
        plan(output, classifier, original, classifier["data"].get("augmentation_path"))
        if classifier_run is None
        else {}
    )
    from .classifier_resume import pending_training

    pending = pending_training(output) if resumed else {}
    if classifier_run is None:
        entries = classifier.get("ensemble", {}).get(
            "members", [{"name": "single", "model": classifier["model"]}]
        )
        for entry in entries:
            from .runtime import classifier_member_config
            member_config = classifier_member_config(classifier, entry)
            name = entry["name"]
            target = (
                packaged_classifier / "members" / name
                if len(entries) > 1
                else packaged_classifier
            )
            if name in resumed:
                receipt = pending.get(name)
                if receipt is not None:
                    write_json(target / "training_complete.json", receipt)
                    print(
                        f"Verifying saved classifier-{name}; training skipped",
                        flush=True,
                    )
                    _run_stage(
                        [
                            sys.executable,
                            "-m",
                            "model.classifier_finalize",
                            "--output-dir",
                            str(target),
                            "--device",
                            member_config["run"]["device"],
                        ],
                        {**child_env, "VGEC_STAGE_NAME": f"classifier-{name}"},
                        tracking,
                    )
                    validate_classifier_run(target, require_hints=False)
                old_identity = target / "wandb_run.json"
                identity = (
                    json.loads(old_identity.read_text())
                    if old_identity.is_file()
                    else {}
                )
                tracking.use_model(target / "classifier", "classifier")
                tracking.stage_result(
                    {
                        "stage": f"classifier-{name}",
                        "run_id": identity.get("id"),
                        "metrics": result_metrics(
                            json.loads((target / "metrics.json").read_text())
                        ),
                    }
                )
                print(
                    f"Reusing completed classifier-{name}; training skipped", flush=True
                )
                continue
            config_path = output / f"classification_{name}.json"
            write_json(config_path, member_config)
            command = stage_command("model.classification", member_config) + [
                "--config",
                str(config_path),
                "--data",
                str(original),
                "--output-dir",
                str(target),
            ]
            if bundle is not None:
                command += ["--correction-data", str(bundle)]
            env = {
                **child_env,
                "VGEC_MODEL_SEED": "42",
                "VGEC_STAGE_NAME": (
                    f"classifier-{name}" if len(entries) > 1 else "classifier"
                ),
            }
            separate_verification = (
                not member_config.get("oof", {}).get("enabled", False)
                and member_config["run"].get("world_size", 1) == 1
            )
            _run_stage(
                command + (["--fit-only"] if separate_verification else []),
                env,
                tracking,
            )
            if separate_verification:
                _run_stage(
                    [
                        sys.executable,
                        "-m",
                        "model.classifier_finalize",
                        "--output-dir",
                        str(target),
                        "--device",
                        member_config["run"]["device"],
                    ],
                    env,
                    tracking,
                )
        if len(entries) > 1:
            from .classifier_ensemble import build

            metrics = build(packaged_classifier, classifier)
            tracking.stage_result(
                {"stage": "classifier", "metrics": result_metrics(metrics)}
            )
    else:
        copy_classifier_run(classifier_run, packaged_classifier)
        identity_path = packaged_classifier / "wandb_run.json"
        identity = (
            json.loads(identity_path.read_text()) if identity_path.is_file() else {}
        )
        tracking.stage_result(
            {
                "stage": "classifier",
                "metrics": result_metrics(
                    json.loads((packaged_classifier / "metrics.json").read_text())
                ),
                "run_id": identity.get("id"),
            }
        )
        oof_path = packaged_classifier / "oof_metrics.json"
        if oof_path.is_file():
            tracking.stage_result(
                {
                    "stage": "oof",
                    "metrics": result_metrics(json.loads(oof_path.read_text())),
                }
            )
    # The first completed stage is already reusable if a later stage fails.
    write_handoff(output)
    if corrector is not None:
        validate_classifier_run(packaged_classifier, require_hints=predicted_hints)
        validate_reuse_data(packaged_classifier, corrector, bundle)
        command = stage_command("model.correction", corrector) + [
            "--config",
            str(Path(args.correction_config).resolve()),
            "--data",
            str(bundle),
            "--classifier-run",
            str(packaged_classifier),
            "--output-dir",
            str(output / "correction"),
        ]
        if initial_adapter is not None:
            command += ["--initial-adapter", str(initial_adapter)]
        _run_stage(
            command,
            {
                **child_env,
                "VGEC_MODEL_SEED": str(corrector["run"]["seed"]),
                "VGEC_STAGE_NAME": "corrector",
            },
            tracking,
        )
        write_handoff(output, correction=True)
    manifest = {
        "mode": args.mode,
        "classifier": "classification/classifier",
        "classifier_training_seeds": [42]
        * len(classifier.get("ensemble", {}).get("members", [None])),
        "classifier_models": [
            entry["model"]["name"]
            for entry in classifier.get("ensemble", {}).get(
                "members", [{"model": classifier["model"]}]
            )
        ],
        "split_seed": split_seed(classifier),
        "seed": classifier["run"]["seed"],
        "classifier_reused": classifier_run is not None,
        "classifier_reused_members": list(resumed),
        "classification_metrics": json.loads(
            (packaged_classifier / "metrics.json").read_text()
        ),
    }
    if corrector is not None:
        manifest.update(
            corrector="correction/adapter",
            correction_metrics=json.loads(
                (output / "correction" / "metrics.json").read_text()
            ),
        )
        manifest["combined_score"] = manifest["correction_metrics"]["combined_score"]
        manifest["combined_score_metadata"] = manifest["correction_metrics"][
            "combined_score_metadata"
        ]
    if corrector is not None and not args.defer_submission:
        validation_path = packaged_classifier / "validation.json"
        records = (
            submission_records
            if submission_records is not None
            else json.loads(validation_path.read_text(encoding="utf-8"))
        )
        evaluated = all(
            row["label"] is not None and row["target"] is not None for row in records
        )
        inference_path = submission_path
        if inference_path is None:
            inference_path = output / "validation_inference.json"
            write_json(
                inference_path,
                [
                    {
                        "ID": r["id"],
                        "Input": r["source"],
                        "Output": {
                            "Error_Type_Id": r["label"],
                            "Fixed Sentence": r["target"],
                        },
                    }
                    for r in records
                ],
            )
        try:
            inference_command = [
                sys.executable,
                "-m",
                "model.pipeline_predict",
                "--classifier",
                str(packaged_classifier / "classifier"),
                "--adapter",
                str(output / "correction/adapter"),
                "--data",
                str(inference_path),
                "--output",
                str(output / "results.json"),
                "--device",
                corrector["run"]["device"],
                "--batch-size",
                str(classifier.get("evaluation", {}).get("batch_size", 1)),
                "--input-scope",
                (
                    "explicit_submission"
                    if submission_records is not None
                    else "validation"
                ),
            ]
            if evaluated:
                inference_command.append("--evaluate")
            _run_stage(
                inference_command,
                {**child_env, "VGEC_MODEL_SEED": "42", "VGEC_STAGE_NAME": "inference"},
                tracking,
            )
            submission = json.loads((output / "results.evaluation.json").read_text())
        except BaseException:
            manifest["submission"] = {
                "status": "failed",
                "diagnostics": "results.evaluation.json",
            }
            write_json(output / "pipeline_manifest.json", manifest)
            raise
        tracking.stage_result(
            {
                "stage": "inference",
                "metrics": (
                    result_metrics(
                        json.loads((output / "results.metrics.json").read_text())
                    )
                    if evaluated
                    else {}
                ),
            }
        )
        # Keep archived artifact references portable; provenance remains in evaluation metadata.
        submission["artifacts"] = {
            name: Path(path).relative_to(output).as_posix()
            for name, path in submission["artifacts"].items()
        }
        manifest["submission"] = submission
        if submission["evaluated"]:
            tracking.evaluation(
                json.loads((output / "results.metrics.json").read_text()), "submission"
            )
    elif corrector is not None:
        manifest["submission"] = {
            "status": "deferred",
            "reason": "Parent tuning coordinator exports after model and decoder selection",
        }
    else:
        manifest["submission"] = {
            "status": "unavailable",
            "reason": "Task1-only has no corrector; no full GEC submission was fabricated",
        }
        print(manifest["submission"]["reason"], flush=True)
    write_json(output / "pipeline_manifest.json", manifest)
    if classifier_run is None:
        tracking.use_model(packaged_classifier / "classifier", "classifier")
    if corrector is not None:
        tracking.use_model(output / "correction" / "adapter", "corrector")
    if corrector is not None:
        print(
            json.dumps(
                {
                    "combined_score": manifest["combined_score"],
                    "combined_score_metadata": manifest["combined_score_metadata"],
                }
            ),
            flush=True,
        )
    if output_config["run"].get("archive", True):
        tracking.heartbeat(stage="archive")
        archive = shutil.make_archive(
            str(output) + "-artifacts", "zip", output.parent, output.name
        )
        print(f"Artifacts: {archive}", flush=True)
    print(f"Completed {args.mode}: {output}", flush=True)


if __name__ == "__main__":

    def terminated(signum, frame):
        raise KeyboardInterrupt("Pipeline termination requested")

    signal.signal(signal.SIGTERM, terminated)
    main()
