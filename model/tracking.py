"""Explicit, rank-zero W&B runs with corpus-private reports and adapter lineage.

No SDK-global run, model watching, code uploads, or corpus artifacts are used.
"""
from __future__ import annotations

from collections import Counter
from functools import wraps
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import warnings
import uuid

from .runtime import PROJECT_ROOT, write_json

DEFAULTS = {"enabled": True, "project": "vlsp26-gec", "mode": "online",
            "log_models": True, "log_text": False}
_PRIVATE_KEYS = {"source", "target", "reference", "prediction", "predictions", "text",
                 "prompt", "content", "train_rows", "authored_overrides", "data_provenance",
                 "provenance", "classifier_provenance", "reference_audit", "annotation_like_audit",
                 "annotation_audit", "hint_descriptions", "input", "output", "fixed sentence",
                 "raw_generation", "target_correction", "predicted_correction", "candidates"}


def safe_value(value, *, allow_prompt=False):
    """Strip corpus-bearing fields and credentials even from nested manifests."""
    if isinstance(value, dict):
        return {str(key): safe_value(item, allow_prompt=allow_prompt) for key, item in value.items()
                if (str(key).lower() not in _PRIVATE_KEYS or
                    (allow_prompt and str(key).lower() in {"prompt", "hint_descriptions"}))
                and not any(part in str(key).lower() for part in ("api_key", "password", "secret", "access_token", "auth_token"))
                and not str(key).lower().endswith("_token")
                and str(key).lower() not in {"env", "environment", "token"}}
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
        digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
        counts[str(row.get("label"))] += 1
    return {"count": len(records), "sha256": digest.hexdigest(), "label_counts": dict(counts)}


def git_identity():
    def git(*args):
        result = subprocess.run(["git", "-C", str(PROJECT_ROOT), *args],
                                capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def _scalars(value, prefix):
    result = {}
    if isinstance(value, dict):
        for key, item in value.items():
            result.update(_scalars(item, f"{prefix}/{key}"))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(value):
            result[prefix] = value
    return result


class TrackingRun:
    """Own a single SDK Run; finish failures without replacing the training error."""

    def __init__(self):
        self.run = None
        self.sdk = None
        self.settings = dict(DEFAULTS)
        self.output = None
        self.identity = None

    def __enter__(self):
        return self

    def start(self, config, output, task, *, group=None, metadata=None):
        self.output = Path(output)
        self.settings = {**DEFAULTS, **config.get("tracking", {})}
        mode = os.environ.get("WANDB_MODE", self.settings["mode"])
        if mode not in ("online", "offline", "disabled"):
            raise ValueError("tracking.mode / WANDB_MODE must be online, offline, or disabled")
        if self.settings["log_text"]:
            raise ValueError("Corpus text uploads are prohibited by the VLSP corpus agreement; tracking.log_text must be false")
        if int(os.environ.get("RANK", "0")) != 0:
            return self
        self.output.mkdir(parents=True, exist_ok=True)
        self.identity = {"id": uuid.uuid4().hex, "url": None, "task": task,
                         "group": os.environ.get("WANDB_RUN_GROUP") or group or self.output.name,
                         "mode": mode, "status": "running", "artifacts": {}}
        if not self.settings["enabled"] or mode == "disabled":
            self.identity.update(mode="disabled", status="disabled")
            self._write_identity()
            return self
        import wandb
        self.sdk = wandb
        self.run = wandb.init(
            project=self.settings["project"], entity=self.settings.get("entity"),
            mode=mode, id=self.identity["id"], resume="never", reinit="create_new",
            group=self.identity["group"], job_type=task, name=f"{self.output.name}-{task}",
            dir=str(self.output), config=safe_value({**config, "git": git_identity(),
                                                    "tracking_metadata": metadata or {}}),
            settings=wandb.Settings(disable_git=True, save_code=False, console="off"))
        self.identity.update(id=self.run.id, url=self.run.url)
        self._write_identity()
        self.run.define_metric("optimizer_step")
        self.run.define_metric("train/*", step_metric="optimizer_step")
        self.run.define_metric("validation/*", step_metric="optimizer_step")
        return self

    def _write_identity(self):
        if self.identity is not None:
            write_json(self.output / "wandb_run.json", self.identity)

    def log(self, values, prefix="", *, optimizer_step=None):
        if self.run is None:
            return
        scalars = _scalars(safe_value(values), prefix.rstrip("/"))
        scalars = {key.lstrip("/"): value for key, value in scalars.items()}
        if optimizer_step is not None:
            scalars["optimizer_step"] = optimizer_step
        if scalars:
            self.run.log(scalars)

    def evaluation(self, report, prefix, *, optimizer_step=None, epoch=None):
        self.log(report, prefix, optimizer_step=optimizer_step)
        if self.run is None:
            return
        classification = report.get("classification", report)
        tables = {}
        if "per_class" in classification:
            tables[f"{prefix}/per_class"] = self.sdk.Table(
                columns=["label", "precision", "recall", "f1", "support"],
                data=[[label, *(row[field] for field in ("precision", "recall", "f1", "support"))]
                      for label, row in classification["per_class"].items()])
        if "confusion_matrix" in classification:
            columns = classification.get("prediction_columns", [0, 1, 2, 3, 4, "invalid"])
            tables[f"{prefix}/confusion_matrix"] = self.sdk.Table(
                columns=["gold", "predicted", "count"],
                data=[[gold, str(predicted), count]
                      for gold, row in zip(classification.get("labels", range(5)), classification["confusion_matrix"])
                      for predicted, count in zip(columns, row)])
        if epoch is not None:
            tables[f"{prefix}/epoch"] = epoch
        if optimizer_step is not None:
            tables["optimizer_step"] = optimizer_step
        if tables:
            self.run.log(tables)

    def summary(self, values):
        if self.run is not None:
            self.run.summary.update(safe_value(values))

    def report(self, name, value):
        if self.run is None:
            return
        safe = safe_value(value)
        path = self.output / f"wandb-{name}.json"
        write_json(path, safe)
        artifact = self.sdk.Artifact(self._name(name), type="evaluation", metadata={"corpus_text": False})
        artifact.add_file(str(path), name=f"{name}.json")
        self.run.log_artifact(artifact)


    def _name(self, suffix):
        return re.sub(r"[^a-zA-Z0-9_.-]", "-", f"{self.identity['id']}-{suffix}")

    def model(self, path, name, *, metadata=None, aliases=None):
        if self.run is None or not self.settings["log_models"]:
            return
        path = Path(path)
        artifact = self.sdk.Artifact(self._name(name), type="model", metadata=safe_value(metadata or {}))
        # Only files generated by save_pretrained plus the saved inference schema.
        allowed = {"adapter_config.json", "classifier_config.json", "preprocessing.json",
                   "loading.json", "inference.json", "tokenizer.json", "tokenizer_config.json",
                   "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt",
                   "tokenizer.model", "spiece.model", "adapter_model.safetensors", "adapter_model.bin",
                   "adapter_model.safetensors.index.json"}
        for file in sorted(path.iterdir()):
            if file.is_file() and (file.name in allowed or re.fullmatch(r"adapter_model-\d+-of-\d+\.safetensors", file.name)):
                if file.name in {"classifier_config.json", "loading.json", "inference.json", "adapter_config.json", "preprocessing.json"}:
                    safe_path = self.output / f"wandb-{name}-{file.name}"
                    write_json(safe_path, safe_value(json.loads(file.read_text(encoding="utf-8")),
                                                     allow_prompt=file.name == "inference.json"))
                    artifact.add_file(str(safe_path), name=file.name)
                else:
                    artifact.add_file(str(file), name=file.name)
        self.run.log_artifact(artifact, aliases=aliases or ["latest"])
        entity = self.run.entity or self.settings.get("entity")
        self.identity["artifacts"][name] = f"{entity}/{self.run.project}/{artifact.name}:latest"
        self._write_identity()

    def use_model(self, path, name):
        if self.run is None or not self.settings["log_models"]:
            return
        path = Path(path)
        identity_path = path.parent / "wandb_run.json"
        identity = json.loads(identity_path.read_text(encoding="utf-8")) if identity_path.is_file() else {}
        reference = identity.get("artifacts", {}).get(name)
        if reference and identity.get("mode") == "online" and self.identity["mode"] == "online":
            self.run.use_artifact(reference, type="model")
        else:
            # Portable/offline/legacy reuse has no server artifact to resolve.
            # Content hashes bind the consumed adapter to an input artifact.
            files = {file.name: hashlib.sha256(file.read_bytes()).hexdigest()
                     for file in path.iterdir() if file.is_file()
                     and file.name.startswith("adapter_model")}
            artifact = self.sdk.Artifact(self._name(f"input-{name}"), type="model",
                                         metadata={"sha256": files, "origin_run_id": identity.get("id")})
            for filename in files:
                artifact.add_file(str(path / filename), name=filename)
            if self.identity["mode"] == "offline":
                # Offline SDK cannot resolve remote input artifacts. Persist the
                # input snapshot and provenance locally for subsequent wandb sync.
                self.run.log_artifact(artifact)
                self.run.summary[f"inputs/{name}"] = {"sha256": files, "origin_run_id": identity.get("id"),
                                                    "artifact": artifact.name}
            else:
                self.run.use_artifact(artifact)

    def callback(self):
        from transformers import TrainerCallback
        tracking = self

        class LiveMetrics(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if state.is_world_process_zero and logs:
                    tracking.log({**logs, "epoch": state.epoch}, "train", optimizer_step=state.global_step)
                return control

        return LiveMetrics()

    def __exit__(self, error_type, error, traceback):
        finishing_error = None
        try:
            if self.run is not None:
                try:
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
                self.identity["status"] = "failed" if error_type or finishing_error else (
                    "disabled" if self.identity["mode"] == "disabled" else "finished")
                try:
                    self._write_identity()
                except BaseException as failure:
                    finishing_error = finishing_error or failure
            self.run = None
        if finishing_error is not None:
            if error_type is None:
                raise finishing_error
            warnings.warn(f"W&B finalization failed ({type(finishing_error).__name__}); preserving original error", RuntimeWarning)
        return False


def tracked_entrypoint(function):
    """Keep CLI bodies unindented while making every post-start exit reliable."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        with TrackingRun() as tracking:
            return function(*args, **kwargs, tracking=tracking)
    return wrapped
