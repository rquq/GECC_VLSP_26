"""Two fresh fits versus preserved baselines; bounded decoding and private export."""

import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import stat
import sys
import time
import zipfile

from model.data import _id_key
from model.runtime import PROJECT_ROOT, write_json, resolve_output
from tools.package_focused_training import MANIFEST, sha256


def _json(path):
    return json.loads(Path(path).read_text())


def _safe(name):
    path = Path(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError("Unsafe focused package path")
    return path


def _migrate_legacy_control(root, destination, manifest, json_sha256):
    """Accept the known old upload after Kaggle expands its nested control ZIP."""
    old = "baseline/quality_control.zip"
    new = "baseline/quality_control/results.json"
    if old not in manifest["file_sha256"] or not json_sha256:
        raise ValueError("Legacy control JSON proof is missing")
    candidates = []
    for control in (root / "baseline").rglob("results.json"):
        if not control.resolve().is_relative_to(root.resolve()):
            raise ValueError("Legacy control JSON path escapes package")
        if control.is_file() and sha256(control) == json_sha256:
            candidates.append(control)
    if len(candidates) != 1:
        raise ValueError("Legacy control JSON payload hash needs exactly one matching expanded results.json; found "
                         + str(len(candidates)))
    raw = candidates[0].read_bytes()
    # All non-archive files retain their original manifest verification.
    for name, digest in manifest["file_sha256"].items():
        if name == old:
            continue
        path = root / _safe(name)
        if (not path.resolve().is_relative_to(root.resolve()) or not path.is_file()
                or sha256(path) != digest):
            raise ValueError("Focused package file hash differs: " + name)
    source_manifest_sha256 = sha256(root / MANIFEST)
    if destination != root:
        destination.mkdir(parents=True, exist_ok=False)
        for name in manifest["file_sha256"]:
            if name == old:
                continue
            path = destination / _safe(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / name, path)
    path = destination / new
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    migrated = json.loads(json.dumps(manifest))
    del migrated["file_sha256"][old]
    migrated["file_sha256"][new] = json_sha256
    migrated.setdefault("control_submission", {})["path"] = new
    migrated["upload_migration"] = {"original_manifest_sha256": source_manifest_sha256,
                                    "control_json_sha256": json_sha256, "nested_archive_hash_checked": False}
    write_json(destination / MANIFEST, migrated)
    print("Using verified plain control JSON from the existing Kaggle upload", flush=True)
    return destination, migrated


def materialize_package(inputs, destination, archive_name, manifest_sha256, *,
                        legacy_manifest_sha256=None, legacy_control_json_sha256=None):
    """Support one ZIP or arbitrary Kaggle extraction prefixes, with pinned hashes."""
    inputs, destination = Path(inputs), Path(destination)
    accepted_hashes = {manifest_sha256, legacy_manifest_sha256} - {None}
    archives = list(inputs.rglob(archive_name))
    if len(archives) > 1:
        raise ValueError("Attach exactly one focused package archive")
    if archives:
        with zipfile.ZipFile(archives[0]) as z:
            candidates = [
                i
                for i in z.infolist()
                if Path(i.filename).name == MANIFEST and not i.is_dir()
            ]
            if len(candidates) != 1:
                raise ValueError("Expected exactly one focused package manifest")
            entry = candidates[0]
            raw = z.read(entry)
            raw_sha256 = hashlib.sha256(raw).hexdigest()
            if raw_sha256 not in accepted_hashes:
                raise ValueError("Focused package manifest hash differs")
            m = json.loads(raw)
            prefix = Path(entry.filename).parent
            if m.get("format") != "vgec-focused-package-v1":
                raise ValueError("Unknown focused package format")
            destination.mkdir(parents=True, exist_ok=False)
            (destination / MANIFEST).write_bytes(raw)
            names = z.namelist()
            for name in m["file_sha256"]:
                tail = _safe(name)
                member = (prefix / tail).as_posix()
                if names.count(member) != 1:
                    raise ValueError("Missing/duplicate focused package file: " + name)
                info = z.getinfo(member)
                if stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError("Symlink package path is forbidden")
                if raw_sha256 == legacy_manifest_sha256 and name == "baseline/quality_control.zip":
                    # Backward compatibility only: unwrap JSON, never hash the old ZIP container.
                    with zipfile.ZipFile(io.BytesIO(z.read(info))) as control:
                        if control.namelist() != ["results.json"]:
                            raise ValueError("Legacy control archive must contain only results.json")
                        path = destination / "baseline/quality_control/results.json"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(control.read("results.json"))
                    continue
                path = destination / tail
                path.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as source, path.open("wb") as out:
                    shutil.copyfileobj(source, out, 1024 * 1024)
        root = destination
        if raw_sha256 == legacy_manifest_sha256:
            root, m = _migrate_legacy_control(root, destination, m, legacy_control_json_sha256)
    else:
        candidates = list(inputs.rglob(MANIFEST))
        if len(candidates) != 1:
            raise ValueError(
                "Attach exactly one focused package with its bundled baselines"
            )
        actual_manifest_sha256 = sha256(candidates[0])
        if actual_manifest_sha256 not in accepted_hashes:
            raise ValueError("Focused package manifest hash differs")
        root = candidates[0].parent
        m = _json(candidates[0])
        if actual_manifest_sha256 == legacy_manifest_sha256:
            root, m = _migrate_legacy_control(root, destination, m, legacy_control_json_sha256)
    if m.get("format") != "vgec-focused-package-v1":
        raise ValueError("Unknown focused package format")
    for name, digest in m["file_sha256"].items():
        path = root / _safe(name)
        if (
            not path.resolve().is_relative_to(root.resolve())
            or not path.is_file()
            or sha256(path) != digest
        ):
            raise ValueError("Focused package file hash differs: " + name)
    return root, m


def decision_diagnostics(gold, comparisons):
    from model.metrics import score

    indexed = {_id_key(x["id"]): x for x in comparisons}
    if len(indexed) != len(comparisons) or set(indexed) != {
        _id_key(g["id"]) for g in gold
    }:
        raise ValueError("Classifier IDs differ from development IDs")
    normal = []
    gated = []
    for g in gold:
        row = indexed[_id_key(g["id"])]
        p = row["probabilities"]
        if (
            len(p) != 5
            or any(not math.isfinite(x) or not 0 <= x <= 1 for x in p)
            or not math.isclose(sum(p), 1, abs_tol=1e-5)
        ):
            raise ValueError("Invalid classifier probabilities")
        if type(row["predicted_class"]) is not int or row["predicted_class"] != max(
            range(5), key=lambda i: p[i]
        ):
            raise ValueError(
                "Saved classifier decision differs from probability argmax"
            )
        normal.append(
            {"id": g["id"], "label": row["predicted_class"], "target": g["source"]}
        )
        label = 0 if 1 - p[0] < 0.5 else max(range(1, 5), key=lambda i: p[i])
        gated.append({"id": g["id"], "label": label, "target": g["source"]})
    return {
        "argmax": score(gold, normal)["classification"],
        "binary_first_05": score(gold, gated)["classification"],
        "export_rule": "argmax",
        "scope": "Fixed diagnostic only; no threshold fitting",
    }


def select_corrector(reports):
    candidates = []
    for source, report in reports.items():
        full = report["full"]
        if not any(
            r["status"] == "completed" and r["strategy"]["name"] == "greedy_reference"
            for r in full
        ):
            raise ValueError("Native greedy must complete for " + source)
        candidates.extend((source, r) for r in full if r["status"] == "completed")

    def key(pair):
        m = pair[1]["metrics"]
        return (
            m["corrections"]["all"]["gleu_plus"],
            m["corrections"]["error_only"]["gleu_plus"],
            m["clean_preservation"]["exact_source_preservation"],
        )

    return max(candidates, key=key)


def stage_timeout(plan, elapsed, stage=None):
    remaining = plan["max_wall_seconds"] - elapsed
    if remaining <= 0:
        raise TimeoutError(
            "Focused experiment wall budget exhausted; completed artifacts retained"
        )
    # Final2034-row decoding can dominate fitting, especially with beam search.
    # Give it the remaining whole-run budget instead of a shorter fit-stage cap.
    return (
        remaining
        if stage == "selected-private-corrector"
        else min(remaining, plan["max_stage_seconds"])
    )


def bundle_selected(output, classifier, adapter, strategy):
    """Keep selected weights inside the downloadable run, with relative paths."""
    from model.run_reuse import copy_classifier_run, validate_classifier_run

    output = Path(output)
    root = output / "selected_models"
    copy_classifier_run(classifier, root / "classification")
    validate_classifier_run(root / "classification", require_hints=False)
    shutil.copytree(adapter, root / "correction/adapter")
    files = {
        str(p.relative_to(root)): sha256(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }
    write_json(
        root / "file_manifest.json",
        {
            "format": "vgec-selected-models-v1",
            "strategy": strategy,
            "file_sha256": files,
        },
    )
    return {
        "classifier_path": "selected_models/classification/classifier",
        "adapter": "selected_models/correction/adapter",
        "file_manifest": "selected_models/file_manifest.json",
        "file_manifest_sha256": sha256(root / "file_manifest.json"),
        "replay_entrypoint": "python -m tools.replay_focused --selection <run>/selected.json --data <input.json> --output <new-directory>",
    }


def resolve_selected(selection_path):
    """Verify portable dependencies and the exact recorded decoder before reload."""
    selection_path = Path(selection_path).resolve()
    root = selection_path.parent
    s = _json(selection_path)
    if (
        s.get("decision_rule") != "argmax"
        or s.get("classifier_path") != "selected_models/classification/classifier"
        or s.get("adapter") != "selected_models/correction/adapter"
        or s.get("file_manifest") != "selected_models/file_manifest.json"
    ):
        raise ValueError("Selected model paths/decision protocol differ")
    manifest = root / _safe(s["file_manifest"])
    if (
        not manifest.resolve().is_relative_to(root)
        or sha256(manifest) != s["file_manifest_sha256"]
    ):
        raise ValueError("Selected model manifest hash differs")
    m = _json(manifest)
    if m.get("format") != "vgec-selected-models-v1":
        raise ValueError("Unknown selected model contract")
    if m.get("strategy") != s.get("strategy"):
        raise ValueError("Selected decoder strategy differs from its manifest")
    for name, digest in m["file_sha256"].items():
        file = manifest.parent / _safe(name)
        if (
            not file.resolve().is_relative_to(manifest.parent.resolve())
            or sha256(file) != digest
        ):
            raise ValueError("Selected dependency hash differs: " + name)
    paths = {k: (root / _safe(s[k])).resolve() for k in ["classifier_path", "adapter"]}
    if any(not p.is_relative_to(root) for p in paths.values()):
        raise ValueError("Selected model path escapes run")
    if (
        not (paths["classifier_path"] / "classifier_config.json").is_file()
        or not (paths["adapter"] / "inference.json").is_file()
    ):
        raise ValueError("Selected model dependencies missing")
    return s, paths


def merge_export(records, classes, corrections, output):
    from model.prediction_export import (
        submission_rows,
        write_submission,
        comparison_rows,
        _indexed_predictions,
    )

    expected = {_id_key(r["id"]) for r in records}
    indexed = {_id_key(r["id"]): r["label"] for r in classes}
    if len(indexed) != len(classes) or set(indexed) != expected:
        raise ValueError("Classifier output IDs differ from inference IDs")
    _indexed_predictions(records, corrections)
    predictions = [{**r, "label": indexed[_id_key(r["id"])]} for r in corrections]
    write_submission(
        Path(output) / "results.json", submission_rows(records, predictions)
    )
    write_json(
        Path(output) / "results.comparisons.json", comparison_rows(records, predictions)
    )


def validate_plan(plan):
    if plan.get("format") != "vgec-focused-v1":
        raise ValueError("Unknown focused plan")
    if not 1 <= plan["screen_per_class"] <= 4 or not 1 <= plan["finalists"] <= 4:
        raise ValueError("Unbounded focused decoding budget")
    if not 60 <= plan["max_stage_seconds"] <= plan["max_wall_seconds"] <= 43200:
        raise ValueError("Invalid focused wall budget")
    strategies = plan["decoder_strategies"]
    if (
        len(strategies) > 4
        or len({s["name"] for s in strategies}) != len(strategies)
        or "greedy_reference" not in [s["name"] for s in strategies]
    ):
        raise ValueError(
            "Focused plan requires native greedy and at most four unique decoders"
        )
    if any(
        s["generation"].get("do_sample", False)
        or s["generation"].get("num_beams", 1) > 4
        for s in strategies
    ):
        raise ValueError("Focused decoders must be deterministic and beam<=4")
    from model.runtime import load_config

    recipe = _json(PROJECT_ROOT / plan["training_recipe"])
    tracks = set()
    for field in ["classification_config", "correction_config"]:
        c = load_config(PROJECT_ROOT / recipe[field])
        track = c.get("competition_id", 1038)
        tracks.add(track)
        if (
            c.get("ensemble")
            or (track == 1038 and c["model"].get("load_in_4bit"))
            or c["run"]["seed"] != 42
            or c["data"]["split_seed"] != 42
            or c["data"]["validation_fraction"] != 0.2
        ):
            raise ValueError(
                "Focused trial requires single models, shared split42 and standard LoRA for track1038"
            )
        if (
            c["training"]["args"]["num_train_epochs"] < 5
            or c["training"]["args"].get("max_steps", -1) != -1
        ):
            raise ValueError("Fresh fits require at least five full epochs")
    if len(tracks) != 1:
        raise ValueError("Both focused stages must use the same competition_id")
    return recipe


def run(
    plan,
    input_root,
    output_root,
    *,
    device="cuda",
    preflight_only=False,
    source_identity=None
):
    from model.two_phase import _run_stage
    from model.run_reuse import validate_classifier_run
    from model.tracking import TrackingRun
    from model.reproducibility import process_environment
    from tools.run_experiment import prepare_run

    recipe = validate_plan(plan)
    work = resolve_output(
        {"run": {"name": plan["name"], "output_root": str(output_root)}}
    )
    output = work / "run"
    output.mkdir(parents=True)
    print("VGEC_RUN_DIR:" + str(output), flush=True)
    proof = _json(PROJECT_ROOT / plan["package_proof"])
    package, packet = materialize_package(
        input_root, work / "package", recipe["data_archive"], proof["manifest_sha256"]
    )
    if (
        packet["data_sha256"] != recipe["data_sha256"]
        or packet["submission_sha256"] != recipe["submission_sha256"]
    ):
        raise ValueError("Focused package data hashes differ from recipe")
    baseline_class = package / "baseline/classification"
    baseline_corrector = package / "baseline/correction"
    validate_classifier_run(baseline_class, require_hints=False)
    if not _json(baseline_corrector / "run_manifest.json").get(
        "saved_adapter_reload_verified"
    ):
        raise ValueError("Baseline corrector reload is unverified")
    training, command = prepare_run(
        PROJECT_ROOT / plan["training_recipe"],
        package,
        work / "training",
        world_size=1,
        device=device,
    )
    from model.data import load_training_split

    gold = load_training_split(package / "VGEC_balanced.json", 0.2, 42)[1]
    if gold != _json(baseline_class / "validation.json") or gold != _json(
        baseline_corrector / "validation.json"
    ):
        raise ValueError("Baseline and challenger development rows differ")
    write_json(
        work / "experiment_manifest.json",
        {
            "plan": plan,
            "source": source_identity,
            "package_manifest_sha256": proof["manifest_sha256"],
            "package": str(package),
            "training": str(training),
            "model_parameters": _json(training / "experiment_manifest.json")[
                "model_parameters"
            ],
        },
    )
    from tools.experiment_log import record

    record(
        work, "started", kind="focused", commit=(source_identity or {}).get("commit")
    )
    started = time.monotonic()
    environment = {
        **process_environment(42),
        "VGEC_CUDA_STAGE": "1" if device == "cuda" else "0",
        "WANDB_RUN_GROUP": work.name,
        "VGEC_FORWARD_EVENTS": "1",
        "VGEC_PIPELINE_TRACKING": "1",
    }
    config = _json(training / "classification_config.json")
    from contextlib import ExitStack

    with TrackingRun() as tracking, ExitStack() as resources:
        if device == "cuda" and config.get("competition_id", 1038) == 1039:
            from model.weight_cache import temporary_cache_environment
            environment.update(temporary_cache_environment(resources))
        tracking.start(
            config,
            output,
            "pipeline",
            group=work.name,
            metadata={"kind": "focused data challenger and saved baselines",
                      "competition_id": config.get("competition_id", 1038)},
        )

        def stage(args, name):
            stage_environment = {
                **environment,
                "VGEC_STAGE_NAME": name,
                "VGEC_ROOT_RUN_NAME": "challenger-training",
            }
            if name in ("baseline-decoders", "challenger-decoders"):
                stage_environment["VGEC_DECODER_GROUP"] = name
            _run_stage(
                args,
                stage_environment,
                tracking,
                timeout_seconds=stage_timeout(plan, time.monotonic() - started, name),
            )

        try:
            stage(command + ["--preflight-only"], "challenger-preflight")
            if preflight_only:
                write_json(
                    output / "preflight.json",
                    {
                        "status": "passed",
                        "validation": len(gold),
                        "package_hashes_verified": True,
                    },
                )
                record(work, "preflight_passed", kind="focused")
                return output
            stage(
                [
                    sys.executable,
                    "-m",
                    "tools.focused_worker",
                    "--classifier",
                    str(baseline_class / "classifier"),
                    "--records",
                    str(baseline_class / "validation.json"),
                    "--output",
                    str(output / "baseline_reload.json"),
                    "--expected-metrics",
                    str(baseline_class / "metrics.json"),
                    "--device",
                    device,
                ],
                "baseline-classifier-reload",
            )
            stage(command + ["--defer-submission"], "fresh-sailor-and-qwen")
            challenger = training / "run"
            classifier_paths = {
                "baseline": baseline_class,
                "challenger": challenger / "classification",
            }
            diagnostics = {}
            for tag, path in classifier_paths.items():
                validate_classifier_run(path, require_hints=False)
                if _json(path / "validation.json") != gold:
                    raise ValueError("Challenger validation differs")
                diagnostics[tag] = decision_diagnostics(
                    gold, _json(path / "predictions.json")
                )
            class_tag = max(
                diagnostics, key=lambda k: diagnostics[k]["argmax"]["macro_f1"]
            )
            classifier = classifier_paths[class_tag]
            write_json(output / "classifier_comparison.json", diagnostics)
            write_json(output / "strategies.json", plan["decoder_strategies"])
            correctors = {
                "baseline": baseline_corrector / "adapter",
                "challenger": challenger / "correction/adapter",
            }
            decoded = {}
            for tag, adapter in correctors.items():
                stage(
                    [
                        sys.executable,
                        "-m",
                        "tools.decoding_sweep",
                        "--adapter",
                        str(adapter),
                        "--classifier-run",
                        str(classifier),
                        "--output-dir",
                        str(output / (tag + "_decoding")),
                        "--device",
                        device,
                        "--per-class",
                        str(plan["screen_per_class"]),
                        "--finalists",
                        str(plan["finalists"]),
                        "--strategies",
                        str(output / "strategies.json"),
                    ],
                    tag + "-decoders",
                )
                decoded[tag] = _json(output / (tag + "_decoding") / "summary.json")
            correction_tag, decode = select_corrector(decoded)
            # Re-evaluate the exact native baseline, not merely its newly optimized decoder.
            baseline_greedy = next(
                r
                for r in decoded["baseline"]["full"]
                if r["strategy"]["name"] == "greedy_reference"
            )
            expected = _json(baseline_corrector / "metrics.json")["corrections"]["all"][
                "gleu_plus"
            ]
            if not math.isclose(
                expected,
                baseline_greedy["metrics"]["corrections"]["all"]["gleu_plus"],
                abs_tol=1e-8,
            ):
                raise ValueError(
                    "Reloaded native baseline greedy score differs from saved checkpoint"
                )
            predictions = _json(
                output
                / (correction_tag + "_decoding")
                / decode["strategy"]["name"]
                / "predictions.json"
            )
            from model.metrics import experiment_report

            labels = {
                _id_key(r["id"]): r["predicted_class"]
                for r in _json(classifier / "predictions.json")
            }
            selected_predictions = [
                {**r, "label": labels[_id_key(r["id"])]} for r in predictions
            ]
            metrics = experiment_report(gold, selected_predictions)
            selected = {
                "competition_id": config.get("competition_id", 1038),
                "classifier_source": class_tag,
                "decision_rule": "argmax",
                "corrector_source": correction_tag,
                "strategy": decode["strategy"],
                "scope": "selected on reused78-row development set; official/private score unknown",
            }
            selected.update(
                bundle_selected(
                    output, classifier, correctors[correction_tag], decode["strategy"]
                )
            )
            write_json(output / "selected.json", selected)
            write_json(output / "selected_metrics.json", metrics)
            _, selected_paths = resolve_selected(output / "selected.json")
            write_json(
                output / "selected_validation_predictions.json", selected_predictions
            )
            write_json(output / "public_strategy.json", decode["strategy"])
            public = package / "private_test.json"
            stage(
                [
                    sys.executable,
                    "-m",
                    "tools.focused_worker",
                    "--classifier",
                    str(selected_paths["classifier_path"]),
                    "--records",
                    str(public),
                    "--output",
                    str(output / "private_classification.json"),
                    "--device",
                    device,
                ],
                "selected-private-classifier",
            )
            stage(
                [
                    sys.executable,
                    "-m",
                    "tools.decoding_sweep",
                    "--adapter",
                    str(selected_paths["adapter"]),
                    "--classifier-run",
                    str(selected_paths["classifier_path"].parent),
                    "--output-dir",
                    str(output / "private_decoding"),
                    "--device",
                    device,
                    "--public-data",
                    str(public),
                    "--strategy",
                    str(output / "public_strategy.json"),
                ],
                "selected-private-corrector",
            )
            from model.data import load_records

            records = load_records(public, require_labels=False)
            merge_export(
                records,
                _json(output / "private_classification.json"),
                _json(output / "private_decoding/predictions.json"),
                output,
            )
            write_json(
                output / "results.evaluation.json",
                {
                    "evaluated": False,
                    "input_scope": "private_inference_only",
                    "input_sha256": sha256(public),
                    "count": len(records),
                    "status": "completed",
                },
            )
            summary = {
                "competition_id": config.get("competition_id", 1038),
                "selected": selected,
                "selected_metrics": metrics,
                "classification": diagnostics,
                "decoding": decoded,
                "training_run": str(challenger),
                "package": str(package),
                "native_precision": True,
            }
            write_json(output / "tuning_summary.json", summary)
            write_json(
                output / "pipeline_manifest.json",
                {
                    "competition_id": config.get("competition_id", 1038),
                    "mode": "tuning",
                    "experiment_kind": "focused",
                    "status": "completed",
                    "summary": "tuning_summary.json",
                    "submission": {"status": "completed"},
                    "selected_metrics": metrics,
                },
            )
            tracking.complete("focused", metrics)
            record(work, "completed", kind="focused")
        except BaseException as error:
            record(work, "failed", kind="focused", error_type=type(error).__name__)
            raise
    return output


def run_from_args(args, source_identity):
    wrapper = _json(args.recipe)
    plan = _json(PROJECT_ROOT / wrapper["plan"])
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
