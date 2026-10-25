"""Committed Kaggle comparison: reuse completed models, screen decoding, fit four challengers."""

import json
import os
from pathlib import Path
import sys
import time
from contextlib import contextmanager

from model.runtime import PROJECT_ROOT, write_json, resolve_output
from model.classifier_resume import sha256
from model.tuning import classifier_search, calibrated_vote


@contextmanager
def _attempt(work, preflight_only):
    from tools.experiment_log import record

    record(work, "started", kind="tuning")
    try:
        yield
    except BaseException as error:
        record(work, "failed", kind="tuning", error_type=type(error).__name__)
        raise
    else:
        record(
            work, "preflight_passed" if preflight_only else "completed", kind="tuning"
        )


def discover_source(input_root, proof):
    candidates = [
        p.parent
        for p in Path(input_root).rglob("pipeline_manifest.json")
        if all((p.parent / name).is_file() for name in proof["file_sha256"])
    ]
    if len(candidates) != 1:
        raise ValueError("Attach exactly one completed October16 notebook output")
    root = candidates[0]
    for name, digest in proof["file_sha256"].items():
        p = root / name
        if not p.resolve().is_relative_to(root.resolve()) or sha256(p) != digest:
            raise ValueError("Completed source hash differs: " + name)
    return root


def validate_plan(plan, project=PROJECT_ROOT, device="cuda"):
    if plan.get("format") != "vgec-tuning-v1":
        raise ValueError("Unknown tuning plan")
    if not 1 <= plan["screen_per_class"] <= 4 or not 1 <= plan["finalists"] <= 4:
        raise ValueError("Unbounded decoder screening budget")
    recipe = json.loads((Path(project) / plan["training_recipe"]).read_text())
    from model.runtime import load_config

    for key in ["classification_config", "correction_config"]:
        c = load_config(Path(project) / recipe[key])
        models = [e["model"] for e in c.get("ensemble", {}).get("members", [])] or [
            c["model"]
        ]
        if any(
            m.get("load_in_4bit")
            or m.get("device_map") != ("balanced" if device == "cuda" else "single")
            for m in models
        ):
            raise ValueError("Kaggle challengers require standard LoRA on two GPUs")
        if (
            c["run"]["seed"] != 42
            or c["data"]["split_seed"] != 42
            or c["data"]["validation_fraction"] != 0.2
        ):
            raise ValueError("Tuning must preserve the shared seed42 grouped split")
    if recipe["data_sha256"] != plan["data_sha256"]:
        raise ValueError("Tuning data hashes differ")
    return recipe


def reuse_public_predictions(strategy, source_tag, proof, public_path):
    return (
        strategy == "greedy_reference"
        and source_tag == "baseline"
        and proof.get("public_data_sha256") == sha256(public_path)
    )


def stage_timeout(plan, name, elapsed):
    remaining = plan["max_wall_seconds"] - elapsed
    if remaining <= 0:
        raise RuntimeError(
            "Tuning wall budget exhausted; saved stage artifacts are retained"
        )
    # This coordinator contains four model fits plus fresh reload verification.
    # It is bounded by the whole experiment, not one individual-stage allowance.
    return (
        remaining
        if name == "retuned-models"
        else min(plan["max_stage_seconds"], remaining)
    )


def validation_vote(classifier, calibration):
    predictions = []
    for name in ["qwen7b", "sailor7b", "mistral7b"]:
        rows = json.loads(
            (Path(classifier) / f"members/{name}/predictions.json").read_text()
        )
        predictions.append(
            [
                {
                    "id": p["id"],
                    "label": p["predicted_class"],
                    "probabilities": p["probabilities"],
                }
                for p in rows
            ]
        )
    return calibrated_vote(
        predictions,
        calibration["weights"],
        calibration["temperature"],
        calibration["phrase_scale"],
    )


def run(
    plan,
    input_root,
    output_root,
    *,
    device="cuda",
    preflight_only=False,
    source_identity=None,
):
    from model.two_phase import _run_stage
    from model.run_reuse import validate_classifier_run
    from model.tracking import TrackingRun
    from tools.run_experiment import prepare_run

    recipe = validate_plan(plan, device=device)
    proof = json.loads((PROJECT_ROOT / plan["source_proof"]).read_text())
    source = discover_source(input_root, proof)
    validate_classifier_run(source / "classification", require_hints=False)
    if not json.loads((source / "correction/run_manifest.json").read_text()).get(
        "saved_adapter_reload_verified"
    ):
        raise ValueError("Source corrector has no successful saved-adapter reload")
    if json.loads(
        (source / "classification/validation.json").read_text()
    ) != json.loads((source / "correction/validation.json").read_text()):
        raise ValueError("Source stage validation rows differ")
    work = resolve_output(
        {"run": {"name": plan["name"], "output_root": str(output_root)}}
    )
    output = work / "run"
    output.mkdir(parents=True)
    print("VGEC_RUN_DIR:" + str(output), flush=True)
    with _attempt(work, preflight_only):
        write_json(
            work / "experiment_manifest.json",
            {
                "plan": plan,
                "source": source_identity,
                "source_output": str(source),
                "source_proof_sha256": sha256(PROJECT_ROOT / plan["source_proof"]),
            },
        )
        # Prepare/validate the fresh challenger data and architecture metadata before
        # consuming GPU hours; this also verifies exact canonical dataset hashes.
        training_work, command = prepare_run(
            PROJECT_ROOT / plan["training_recipe"],
            input_root,
            work / "training",
            world_size=1,
            device=device,
        )
        from model.data import load_classifier_split, load_training_split
        from model.classification import _organizer_partition

        requested = json.loads(
            (training_work / "classification_config.json").read_text()
        )
        train, info = load_classifier_split(
            requested["data"]["path"], requested["data"]["augmentation_path"], 0.2, 42
        )
        _, validation = _organizer_partition(
            Path(requested["data"]["path"]), 0.2, 42, info
        )
        correction = json.loads((training_work / "correction_config.json").read_text())
        correction_train, correction_validation, _ = load_training_split(
            correction["data"]["path"], 0.2, 42
        )
        if (
            train != json.loads((source / "classification/train.json").read_text())
            or correction_train
            != json.loads((source / "correction/train.json").read_text())
            or validation
            != json.loads((source / "classification/validation.json").read_text())
            or correction_validation != validation
        ):
            raise ValueError(
                "Tuning population differs from completed baseline; comparison would be invalid"
            )
        if preflight_only:
            write_json(
                output / "preflight.json",
                {
                    "status": "passed",
                    "source": str(source),
                    "training": str(training_work),
                },
            )
            return output
        started = time.monotonic()
        config = json.loads((source / "classification/config.json").read_text())
        environment = {
            **os.environ,
            "VGEC_CUDA_STAGE": "1" if device == "cuda" else "0",
            "WANDB_RUN_GROUP": work.name,
            "VGEC_MODEL_SEED": "42",
            "VGEC_FORWARD_EVENTS": "1",
        }
        with TrackingRun() as tracking:
            tracking.start(
                config,
                output,
                "pipeline",
                group=work.name,
                metadata={"kind": "bounded tuning", "source": source.name},
            )

            def stage(args, name):
                _run_stage(
                    args,
                    {
                        **environment,
                        "VGEC_STAGE_NAME": name,
                        "VGEC_ROOT_RUN_NAME": "retuned-model-training",
                    },
                    tracking,
                    timeout_seconds=stage_timeout(
                        plan, name, time.monotonic() - started
                    ),
                )

            stage(command + ["--preflight-only"], "challenger-preflight")
            baseline_class = classifier_search(
                source / "classification", output / "baseline_classifier_search.json"
            )
            baseline_strategy_args = []
            if plan.get("decoder_strategies"):
                write_json(
                    output / "baseline_strategies.json", plan["decoder_strategies"]
                )
                baseline_strategy_args = [
                    "--strategies",
                    str(output / "baseline_strategies.json"),
                ]
            stage(
                [
                    sys.executable,
                    "-m",
                    "tools.decoding_sweep",
                    "--adapter",
                    str(source / "correction/adapter"),
                    "--classifier-run",
                    str(source / "classification"),
                    "--output-dir",
                    str(output / "baseline_decoding"),
                    "--device",
                    device,
                    "--per-class",
                    str(plan["screen_per_class"]),
                    "--finalists",
                    str(plan["finalists"]),
                ]
                + baseline_strategy_args,
                "baseline-decoders",
            )
            # Select models/decoding before spending GPU time on public inference.
            stage(command + ["--defer-submission"], "retuned-models")
            challenger = training_work / "run"
            challenger_class = classifier_search(
                challenger / "classification",
                output / "challenger_classifier_search.json",
            )
            # The saved training run already evaluates greedy inference. Reuse the
            # baseline panel's best non-reference strategies for the new adapter.
            baseline_decode = json.loads(
                (output / "baseline_decoding/summary.json").read_text()
            )
            chosen = [
                r["strategy"]
                for r in baseline_decode["full"]
                if r["status"] == "completed"
            ]
            write_json(output / "challenger_strategies.json", chosen)
            stage(
                [
                    sys.executable,
                    "-m",
                    "tools.decoding_sweep",
                    "--adapter",
                    str(challenger / "correction/adapter"),
                    "--classifier-run",
                    str(challenger / "classification"),
                    "--output-dir",
                    str(output / "challenger_decoding"),
                    "--device",
                    device,
                    "--per-class",
                    str(plan["screen_per_class"]),
                    "--finalists",
                    str(plan["challenger_finalists"]),
                    "--strategies",
                    str(output / "challenger_strategies.json"),
                ],
                "challenger-decoders",
            )
            challenger_decode = json.loads(
                (output / "challenger_decoding/summary.json").read_text()
            )
            bestclass = max(
                [
                    ("baseline", source, baseline_class["best"]),
                    ("challenger", challenger, challenger_class["best"]),
                ],
                key=lambda x: x[2]["metrics"]["macro_f1"],
            )
            corrections = []
            for label, src, decoded in [
                ("baseline", source, baseline_decode),
                ("challenger", challenger, challenger_decode),
            ]:
                for result in decoded["full"]:
                    if result["status"] == "completed":
                        corrections.append((label, src, result))
            bestcorrection = max(
                corrections,
                key=lambda x: x[2]["metrics"]["corrections"]["all"]["gleu_plus"],
            )
            class_label, class_source, calibration = bestclass
            correction_label, correction_source, decode_report = bestcorrection
            selected = {
                "classifier_source": class_label,
                "classifier_path": str(class_source / "classification"),
                "calibration": calibration,
                "corrector_source": correction_label,
                "adapter": str(correction_source / "correction/adapter"),
                "strategy": decode_report["strategy"],
                "scope": "selected on reused78-row development set; official score unknown",
            }
            write_json(output / "selected.json", selected)
            decoder_folder = output / (
                "baseline_decoding"
                if correction_label == "baseline"
                else "challenger_decoding"
            )
            predictions = json.loads(
                (
                    decoder_folder
                    / decode_report["strategy"]["name"]
                    / "predictions.json"
                ).read_text()
            )
            from model.data import _id_key, load_records
            from model.metrics import experiment_report

            class_predictions = {
                _id_key(p["id"]): p
                for p in validation_vote(class_source / "classification", calibration)
            }
            gold = json.loads((source / "classification/validation.json").read_text())
            target = {_id_key(p["id"]): p for p in predictions}
            final_predictions = [
                {
                    **target[_id_key(r["id"])],
                    "label": class_predictions[_id_key(r["id"])]["label"],
                }
                for r in gold
            ]
            metrics = experiment_report(gold, final_predictions)
            write_json(output / "selected_metrics.json", metrics)
            # Export the selected pipeline, not an untested combination of flags.
            runtime = json.loads(
                (training_work / "experiment_manifest.json").read_text()
            )
            public = load_records(
                runtime["inputs"]["submission"]["path"], require_labels=False
            )
            if reuse_public_predictions(
                decode_report["strategy"]["name"],
                correction_label,
                proof,
                runtime["inputs"]["submission"]["path"],
            ):
                raw = json.loads((correction_source / "results.json").read_text())
                targets = {
                    _id_key(r["ID"]): r["Output"]["Fixed Sentence"][0] for r in raw
                }
            else:
                write_json(output / "public_strategy.json", decode_report["strategy"])
                stage(
                    [
                        sys.executable,
                        "-m",
                        "tools.decoding_sweep",
                        "--adapter",
                        str(correction_source / "correction/adapter"),
                        "--classifier-run",
                        str(class_source / "classification"),
                        "--output-dir",
                        str(output / "public_decoding"),
                        "--device",
                        device,
                        "--public-data",
                        runtime["inputs"]["submission"]["path"],
                        "--strategy",
                        str(output / "public_strategy.json"),
                    ],
                    "selected-public-corrector",
                )
                targets = {
                    _id_key(r["id"]): r["target"]
                    for r in json.loads(
                        (output / "public_decoding/predictions.json").read_text()
                    )
                }
            from model.pipeline_predict import _classify_isolated

            member_predictions = []
            for name in ["qwen7b", "sailor7b", "mistral7b"]:
                tracking.heartbeat(stage="selected-public-classifier-" + name)
                member_predictions.append(
                    _classify_isolated(
                        class_source / f"classification/members/{name}/classifier",
                        public,
                        device,
                        1,
                    )
                )
            voted = calibrated_vote(
                member_predictions,
                calibration["weights"],
                calibration["temperature"],
                calibration["phrase_scale"],
            )
            from model.prediction_export import write_submission, submission_rows

            final = [{**p, "target": targets[_id_key(p["id"])]} for p in voted]
            write_submission(output / "results.json", submission_rows(public, final))
            summary = {
                "baseline_classification": json.loads(
                    (source / "classification/metrics.json").read_text()
                )["classification"],
                "baseline_decoding": baseline_decode["best"],
                "challenger_classification": challenger_class["best"],
                "challenger_decoding": challenger_decode["best"],
                "selected": selected,
                "selected_metrics": metrics,
                "native_precision": True,
                "source": str(source),
                "training_run": str(challenger),
            }
            write_json(output / "tuning_summary.json", summary)
            write_json(
                output / "pipeline_manifest.json",
                {
                    "mode": "tuning",
                    "status": "completed",
                    "summary": "tuning_summary.json",
                    "submission": {"status": "completed"},
                    "selected_metrics": metrics,
                },
            )
            tracking.complete("tuning", metrics)
        return output


def run_from_args(args, source_identity):
    wrapper = json.loads(Path(args.recipe).read_text())
    plan = json.loads((PROJECT_ROOT / wrapper["plan"]).read_text())
    kaggle = Path("/kaggle/working").is_dir()
    inputs = (
        Path(args.input_root)
        if args.input_root
        else Path("/kaggle/input") if kaggle else PROJECT_ROOT / "dataset"
    )
    outputs = (
        Path(args.output_root)
        if args.output_root
        else Path("/kaggle/working/runs") if kaggle else PROJECT_ROOT / "runs"
    )
    return run(
        plan,
        inputs,
        outputs,
        preflight_only=args.preflight_only,
        source_identity=source_identity,
    )
