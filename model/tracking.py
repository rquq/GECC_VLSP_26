"""Rank-zero W&B outcomes and explicit corpus-safe progress logs."""

from __future__ import annotations

from collections import Counter
from functools import wraps
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import threading
import warnings
import uuid

from .runtime import write_json

DEFAULTS = {
    "enabled": True,
    "project": "vlsp26-gec",
    "mode": "online",
    "log_models": True,
    "log_text": False,
}
_PRIVATE_KEYS = {
    "source",
    "target",
    "reference",
    "prediction",
    "predictions",
    "text",
    "prompt",
    "content",
    "train_rows",
    "authored_overrides",
    "data_provenance",
    "provenance",
    "classifier_provenance",
    "reference_audit",
    "annotation_like_audit",
    "annotation_audit",
    "hint_descriptions",
    "input",
    "output",
    "fixed sentence",
    "raw_generation",
    "target_correction",
    "predicted_correction",
    "candidates",
    "original_snapshot",
    "repairs",
    "before",
    "after",
    "review_evidence",
    "reason",
}


def safe_value(value, *, allow_prompt=False):
    """Strip corpus-bearing fields and credentials even from nested manifests."""
    if isinstance(value, dict):
        return {
            str(key): safe_value(item, allow_prompt=allow_prompt)
            for key, item in value.items()
            if (
                str(key).lower() not in _PRIVATE_KEYS
                or (
                    allow_prompt and str(key).lower() in {"prompt", "hint_descriptions"}
                )
            )
            and not any(
                part in str(key).lower()
                for part in (
                    "api_key",
                    "password",
                    "secret",
                    "access_token",
                    "auth_token",
                )
            )
            and not str(key).lower().endswith("_token")
            and str(key).lower() not in {"env", "environment", "token"}
        }
    if isinstance(value, (list, tuple)):
        return [safe_value(item, allow_prompt=allow_prompt) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def records_identity(records):
    """Hash data incrementally; never serialize source/reference text to W&B."""
    digest = hashlib.sha256()
    counts = Counter()
    for row in records:
        digest.update(
            json.dumps(
                row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        )
        digest.update(b"\n")
        counts[str(row.get("label"))] += 1
    return {
        "count": len(records),
        "sha256": digest.hexdigest(),
        "label_counts": dict(counts),
    }


STAGE_EVENT = "VGEC_STAGE_RESULT "
PROGRESS_EVENT = "VGEC_STAGE_PROGRESS "
ROOT_HEARTBEAT_SECONDS = 30
PROGRESS_METRICS = {
    "loss",
    "grad_norm",
    "epoch",
    "training_loss",
    "learning_rate",
    "macro_f1",
    "micro_f1",
    "gleu_plus",
    "exact_match",
    "rouge_l",
    "error_gleu_plus",
    "error_exact_match",
    "error_rouge_l",
    "clean_preservation",
    "combined_score",
    "invalid_json_rate",
    "invalid_prediction_rate",
    "best_epoch",
}


def publish_progress(stage, prefix, metrics, optimizer_step=None):
    if int(os.environ.get("RANK", "0")) != 0 or not os.environ.get(
        "VGEC_PIPELINE_TRACKING"
    ):
        return
    event = {
        "stage": os.environ.get("VGEC_STAGE_NAME") or stage,
        "prefix": prefix,
        "metrics": {
            k: v for k, v in metrics.items() if k in PROGRESS_METRICS and _number(v)
        },
    }
    if _number(optimizer_step):
        event["optimizer_step"] = optimizer_step
    print(PROGRESS_EVENT + json.dumps(event, allow_nan=False), flush=True)


def _number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def result_metrics(report):
    """Only outcomes belong in W&B; full reports and configurations stay local."""
    classification = report.get("classification", report) or {}
    metrics = {
        key: classification[key]
        for key in ("macro_f1", "micro_f1")
        if _number(classification.get(key))
    }
    for subset, prefix in (("all", ""), ("error_only", "error_")):
        correction = report.get("corrections", {}).get(subset, {})
        for key in ("gleu_plus", "exact_match", "rouge_l"):
            if _number(correction.get(key)):
                metrics[prefix + key] = correction[key]
    preservation = report.get("clean_preservation", {}).get("exact_source_preservation")
    if _number(preservation):
        metrics["clean_preservation"] = preservation
    for key in ("combined_score", "invalid_json_rate", "invalid_prediction_rate"):
        if _number(report.get(key)):
            metrics[key] = report[key]
    return metrics


def publish_stage_result(stage, report, *, best_epoch=None, run_id=None):
    """Send an outcome-only event to the supervising pipeline process."""
    if int(os.environ.get("RANK", "0")) != 0 or not os.environ.get(
        "VGEC_PIPELINE_TRACKING"
    ):
        return
    event = {
        "stage": (
            (os.environ.get("VGEC_STAGE_NAME") or stage)
            if stage in ("classifier", "corrector")
            else stage
        ),
        "metrics": result_metrics(report),
    }
    if _number(best_epoch):
        event["metrics"]["best_epoch"] = best_epoch
    if run_id:
        event["run_id"] = run_id
    print(STAGE_EVENT + json.dumps(event, allow_nan=False), flush=True)


class TrackingRun:
    """Own a single SDK Run; finish failures without replacing the training error."""

    def __init__(self):
        self.run = None
        self.sdk = None
        self.settings = dict(DEFAULTS)
        self.output = None
        self.identity = None
        self.completed_stages = {}
        self.started_at = time.monotonic()
        self.current_stage = "starting"
        self.task = None
        self._heartbeat_lock = threading.RLock()
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread = None

    def __enter__(self):
        return self

    def start(self, config, output, task, *, group=None, metadata=None):
        self.task = task
        self.output = Path(output)
        self.settings = {**DEFAULTS, **config.get("tracking", {})}
        mode = os.environ.get("WANDB_MODE", self.settings["mode"])
        if mode not in ("online", "offline", "disabled"):
            raise ValueError(
                "tracking.mode / WANDB_MODE must be online, offline, or disabled"
            )
        if self.settings["log_text"]:
            raise ValueError(
                "Corpus text uploads are prohibited by the VLSP corpus agreement; tracking.log_text must be false"
            )
        if int(os.environ.get("RANK", "0")) != 0:
            return self
        self.output.mkdir(parents=True, exist_ok=True)
        self.identity = {
            "id": uuid.uuid4().hex,
            "url": None,
            "task": task,
            "group": (
                (group or self.output.name)
                if task == "pipeline"
                else (os.environ.get("WANDB_RUN_GROUP") or group or self.output.name)
            ),
            "mode": mode,
            "status": "running",
            "artifacts": {},
            "metadata": safe_value(metadata or {}),
        }
        if not self.settings["enabled"] or mode == "disabled":
            self.identity.update(mode="disabled", status="disabled")
            self._write_identity()
            return self
        import wandb

        self.sdk = wandb
        stage = {"classification": "classifier", "correction": "corrector"}.get(
            task, task
        )
        if task == "classification-oof":
            stage = f"fold-{int(self.output.name.rsplit('-', 1)[1]) + 1}"
        name = (
            os.environ.get("VGEC_ROOT_RUN_NAME", "run")
            if task == "pipeline"
            else (
                os.environ.get("VGEC_STAGE_NAME")
                or f"classifier-seed-{config['run']['seed']}"
                if task == "classification"
                else stage
            )
        )
        self.identity["name"] = name
        self.run = wandb.init(
            project=self.settings["project"],
            entity=self.settings.get("entity"),
            mode=mode,
            id=self.identity["id"],
            resume="never",
            reinit="create_new",
            group=self.identity["group"],
            job_type=task,
            name=name,
            dir=str(self.output),
            config={},
            settings=wandb.Settings(
                disable_git=True,
                save_code=False,
                console="off",
                quiet=True,
                x_disable_stats=False,
                x_stats_sampling_interval=15,
                x_stats_pid=os.getpid(),
                x_stats_track_process_tree=True,
                x_stats_disk_paths=("/", str(self.output.resolve())),
            ),
        )
        self.identity.update(id=self.run.id, url=self.run.url)
        self._write_identity()
        self.run.define_metric("optimizer_step", hidden=True, summary="none")
        self.run.define_metric("train/*", step_metric="optimizer_step", summary="none")
        self.run.define_metric(
            "validation/*", step_metric="optimizer_step", summary="none"
        )
        self.run.define_metric("train/epoch", hidden=True, summary="none")
        self.run.define_metric("validation/epoch", hidden=True, summary="none")
        self.run.define_metric(
            "validation/classification/epoch", hidden=True, summary="none"
        )
        if task == "pipeline":
            self.run.define_metric("pipeline/elapsed_seconds", summary="none")
            for pattern in (
                "classifier/*",
                "classifier-seed-42/*",
                "classifier-seed-43/*",
                "classifier-seed-44/*",
                "classifier-qwen7b/*",
                "classifier-sailor7b/*",
                "classifier-mistral7b/*",
                "corrector/*",
                "inference/*",
                "pipeline/*",
            ):
                self.run.define_metric(
                    pattern, step_metric="pipeline/elapsed_seconds", summary="none"
                )
        self.message("Started " + stage)
        if task == "pipeline":

            def keep_alive():
                while not self._heartbeat_stop.wait(ROOT_HEARTBEAT_SECONDS):
                    try:
                        self.heartbeat()
                    except Exception as error:
                        warnings.warn(
                            f"Root heartbeat failed ({type(error).__name__})",
                            RuntimeWarning,
                        )

            self._heartbeat_thread = threading.Thread(target=keep_alive, daemon=True)
            self._heartbeat_thread.start()
        return self

    def _write_identity(self):
        if self.identity is not None:
            write_json(self.output / "wandb_run.json", self.identity)

    def message(self, text):
        """Write only internally constructed, text-free progress to the Logs tab."""
        if self.run is not None:
            self.run.write_logs(text)

    def log(self, values, prefix="train", *, optimizer_step=None):
        publish_progress(
            "classifier" if self.task == "classification" else "corrector",
            prefix,
            values,
            optimizer_step,
        )
        if self.run is None:
            return
        allowed = ("loss", "grad_norm", "epoch", "training_loss", "learning_rate")
        scalars = {
            f"{prefix}/{key}": value
            for key, value in values.items()
            if key in allowed and _number(value)
        }
        if not scalars:
            return
        if optimizer_step is not None:
            scalars["optimizer_step"] = optimizer_step
        self.run.log(scalars)
        progress = " ".join(
            f"{key.rsplit('/', 1)[-1]}={value:.6g}" for key, value in scalars.items()
        )
        self.message(f"{prefix}: {progress}")

    def evaluation(self, report, prefix, *, optimizer_step=None, epoch=None):
        metrics = result_metrics(report)
        publish_progress(
            "classifier" if self.task == "classification" else "corrector",
            prefix,
            {**metrics, **({"epoch": epoch} if epoch is not None else {})},
            optimizer_step,
        )
        if self.run is None:
            return
        if prefix.split("/", 1)[0] == "baseline":
            if metrics:
                self.run.log(
                    {
                        "baseline": self.sdk.Table(
                            columns=["metric", "value"],
                            data=[[key, value] for key, value in metrics.items()],
                        )
                    }
                )
            self.message(
                "baseline: "
                + " ".join(f"{key}={value:.6g}" for key, value in metrics.items())
            )
            return
        values = {f"{prefix}/{key}": value for key, value in metrics.items()}
        if epoch is not None:
            values[f"{prefix}/epoch"] = epoch
        if optimizer_step is not None:
            values["optimizer_step"] = optimizer_step
        if values:
            self.run.log(values)
        self.message(
            prefix
            + ": "
            + " ".join(f"{key}={value:.6g}" for key, value in metrics.items())
        )
        classification = report.get("classification", report) or {}
        if prefix == "final" and "per_class" in classification:
            self.run.log(
                {
                    "final/per_class": self.sdk.Table(
                        columns=["label", "precision", "recall", "f1", "support"],
                        data=[
                            [
                                label,
                                *(
                                    row[field]
                                    for field in (
                                        "precision",
                                        "recall",
                                        "f1",
                                        "support",
                                    )
                                ),
                            ]
                            for label, row in classification["per_class"].items()
                        ],
                    )
                }
            )

    def summary(self, values):
        if self.run is not None:
            self.run.summary.update(safe_value(values))

    def complete(self, stage, report, *, best_epoch=None):
        self.evaluation(report, "final")
        if _number(best_epoch):
            self.summary({"best_epoch": best_epoch})
        self.__exit__(None, None, None)
        publish_stage_result(
            stage,
            report,
            best_epoch=best_epoch,
            run_id=(
                self.identity["id"]
                if self.identity and self.identity["mode"] != "disabled"
                else None
            ),
        )

    def stage_result(self, event):
        """Refresh root results immediately; repeat events never duplicate stages."""
        stage = event["stage"]
        if stage in self.completed_stages:
            return
        metrics = {
            key: value
            for key, value in event["metrics"].items()
            if key
            in {
                "macro_f1",
                "micro_f1",
                "gleu_plus",
                "exact_match",
                "rouge_l",
                "error_gleu_plus",
                "error_exact_match",
                "error_rouge_l",
                "clean_preservation",
                "combined_score",
                "invalid_json_rate",
                "invalid_prediction_rate",
                "best_epoch",
            }
            and _number(value)
        }
        result = {"stage": stage, "run_id": event.get("run_id"), **metrics}
        self.completed_stages[stage] = result
        write_json(
            self.output / "stage_results.json", list(self.completed_stages.values())
        )
        if self.run is not None:
            self.run.log({f"{stage}/{key}": value for key, value in metrics.items()})
            self.run.summary.update(
                {f"{stage}/{key}": value for key, value in metrics.items()}
            )
            columns = [
                "stage",
                "run_id",
                "macro_f1",
                "micro_f1",
                "gleu_plus",
                "exact_match",
                "rouge_l",
                "error_gleu_plus",
                "error_exact_match",
                "clean_preservation",
                "combined_score",
                "best_epoch",
            ]
            self.run.log(
                {
                    "results": self.sdk.Table(
                        columns=columns,
                        data=[
                            [row.get(column) for column in columns]
                            for row in self.completed_stages.values()
                        ],
                    )
                }
            )
        self.message(
            stage
            + " completed: "
            + " ".join(f"{key}={value:.6g}" for key, value in metrics.items())
        )

    def heartbeat(self, stage=None):
        with self._heartbeat_lock:
            if stage is not None:
                self.current_stage = stage
            elapsed = time.monotonic() - self.started_at
            status = {
                "current_stage": self.current_stage,
                "elapsed_seconds": elapsed,
                "completed_stages": list(self.completed_stages),
            }
            if self.output is not None:
                write_json(self.output / "pipeline_progress.json", status)
            if self.run is not None:
                self.run.log(
                    {
                        "pipeline/elapsed_seconds": elapsed,
                        "pipeline/stages_completed": len(self.completed_stages),
                    }
                )
                self.run.summary.update(
                    {
                        "current_stage": self.current_stage,
                        "elapsed_seconds": elapsed,
                        "completed_stages": list(self.completed_stages),
                    }
                )

    def progress(self, event):
        stage, prefix = event.get("stage", ""), event.get("prefix", "")
        if not re.fullmatch(
            r"classifier(?:-seed-(?:42|43|44)|-(?:qwen7b|sailor7b|mistral7b))?|corrector|inference|oof",
            stage,
        ) or not re.fullmatch(
            r"(?:train|training|validation|baseline|final)(?:/classification)?", prefix
        ):
            return
        self.current_stage = stage
        values = {
            f"{stage}/{prefix}/{k}": v
            for k, v in event.get("metrics", {}).items()
            if k in PROGRESS_METRICS and _number(v)
        }
        if _number(event.get("optimizer_step")):
            values[f"{stage}/optimizer_step"] = event["optimizer_step"]
        if values and self.run is not None:
            self.run.log(
                {
                    **values,
                    "pipeline/elapsed_seconds": time.monotonic() - self.started_at,
                }
            )

    def _name(self, suffix):
        return re.sub(r"[^a-zA-Z0-9_.-]", "-", f"{self.identity['id']}-{suffix}")

    def model(self, path, name, *, metadata=None, aliases=None):
        if self.run is None or not self.settings["log_models"]:
            return
        path = Path(path)
        artifact = self.sdk.Artifact(
            self._name(name), type="model", metadata=safe_value(metadata or {})
        )
        # Only files generated by save_pretrained plus the saved inference schema.
        allowed = {
            "adapter_config.json",
            "classifier_config.json",
            "preprocessing.json",
            "loading.json",
            "inference.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "vocab.json",
            "merges.txt",
            "tokenizer.model",
            "spiece.model",
            "adapter_model.safetensors",
            "adapter_model.bin",
            "adapter_model.safetensors.index.json",
        }
        for file in sorted(path.iterdir()):
            if file.is_file() and (
                file.name in allowed
                or re.fullmatch(r"adapter_model-\d+-of-\d+\.safetensors", file.name)
            ):
                if file.name in {
                    "classifier_config.json",
                    "loading.json",
                    "inference.json",
                    "adapter_config.json",
                    "preprocessing.json",
                }:
                    safe_path = self.output / f"wandb-{name}-{file.name}"
                    write_json(
                        safe_path,
                        safe_value(
                            json.loads(file.read_text(encoding="utf-8")),
                            allow_prompt=file.name == "inference.json",
                        ),
                    )
                    artifact.add_file(str(safe_path), name=file.name)
                else:
                    artifact.add_file(str(file), name=file.name)
        self.run.log_artifact(artifact, aliases=aliases or ["latest"])
        entity = self.run.entity or self.settings.get("entity")
        self.identity["artifacts"][
            name
        ] = f"{entity}/{self.run.project}/{artifact.name}:latest"
        self._write_identity()

    def use_model(self, path, name):
        if self.run is None or not self.settings["log_models"]:
            return
        path = Path(path)
        if (path / "ensemble.json").is_file():
            from .classifier_ensemble import members

            for member in members(path):
                self.use_model(member, "classifier")
            return
        identity_path = path.parent / "wandb_run.json"
        identity = (
            json.loads(identity_path.read_text(encoding="utf-8"))
            if identity_path.is_file()
            else {}
        )
        reference = identity.get("artifacts", {}).get(name)
        if (
            reference
            and identity.get("mode") == "online"
            and self.identity["mode"] == "online"
        ):
            self.run.use_artifact(reference, type="model")
        else:
            # Portable/offline/legacy reuse has no server artifact to resolve.
            # Content hashes bind the consumed adapter to an input artifact.
            files = {
                file.name: hashlib.sha256(file.read_bytes()).hexdigest()
                for file in path.iterdir()
                if file.is_file() and file.name.startswith("adapter_model")
            }
            artifact = self.sdk.Artifact(
                self._name(f"input-{name}"),
                type="model",
                metadata={"sha256": files, "origin_run_id": identity.get("id")},
            )
            for filename in files:
                artifact.add_file(str(path / filename), name=filename)
            if self.identity["mode"] == "offline":
                # Offline SDK cannot resolve remote input artifacts. Persist the
                # input snapshot and provenance locally for subsequent wandb sync.
                self.run.log_artifact(artifact)
                # The input artifact already records hashes; do not duplicate them in the run summary.
            else:
                self.run.use_artifact(artifact)

    def callback(self):
        from transformers import TrainerCallback

        tracking = self

        class LiveMetrics(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if state.is_world_process_zero and logs:
                    values = {
                        key: value
                        for key, value in logs.items()
                        if key in {"loss", "grad_norm", "learning_rate"}
                    }
                    if "train_loss" in logs:
                        values["training_loss"] = logs["train_loss"]
                    tracking.log(
                        {**values, "epoch": state.epoch},
                        "train",
                        optimizer_step=state.global_step,
                    )
                return control

        return LiveMetrics()

    def __exit__(self, error_type, error, traceback):
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=2)
        if self.run is None and (
            self.identity is None or self.identity["status"] != "running"
        ):
            return False
        finishing_error = None
        try:
            if self.run is not None:
                try:
                    self.message(
                        "Failed " + error_type.__name__ if error_type else "Finished"
                    )
                    self.run.summary["status"] = "failed" if error_type else "finished"
                    if error_type:
                        # Exception messages may embed source text or secrets.
                        self.run.summary["failure_type"] = error_type.__name__
                finally:
                    self.run.finish(exit_code=1 if error_type else 0)
        except BaseException as failure:
            finishing_error = failure
        finally:
            if self.identity is not None:
                self.identity["status"] = (
                    "failed"
                    if error_type or finishing_error
                    else (
                        "disabled"
                        if self.identity["mode"] == "disabled"
                        else "finished"
                    )
                )
                try:
                    self._write_identity()
                    if self.task == "pipeline":
                        write_json(
                            self.output / "pipeline_progress.json",
                            {
                                "status": self.identity["status"],
                                "current_stage": self.current_stage,
                                "elapsed_seconds": time.monotonic() - self.started_at,
                                "completed_stages": list(self.completed_stages),
                            },
                        )
                except BaseException as failure:
                    finishing_error = finishing_error or failure
            self.run = None
        if finishing_error is not None:
            if error_type is None:
                raise finishing_error
            warnings.warn(
                f"W&B finalization failed ({type(finishing_error).__name__}); preserving original error",
                RuntimeWarning,
            )
        return False


def tracked_entrypoint(function):
    """Keep CLI bodies unindented while making every post-start exit reliable."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        with TrackingRun() as tracking:
            return function(*args, **kwargs, tracking=tracking)

    return wrapped
