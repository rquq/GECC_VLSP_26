"""Two bounded BamiBERT screens; select internally, evaluate frozen holdout once."""

from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time

from model.data import (
    _id_key,
    _source_key,
    load_classifier_split,
    load_records,
    split_records,
)
from model.runtime import write_json
from tools.experiment_log import record

MODEL = "Qualcomm-AI-Research/BamiBERT"
REVISION = "57bc1340debbe4e348ec549047a763caebe4a977"


def screening_split(originals, bundles):
    """One shared original-row validation; link parents, texts and template families."""
    parent = {_id_key(r["id"]): _id_key(r["id"]) for r in originals}

    def find(key):
        if parent[key] != key:
            parent[key] = find(parent[key])
        return parent[key]

    def union(a, b):
        parent[find(a)] = find(b)

    texts, families = {}, {}
    forced_train = set()
    for rows, provenance in bundles:
        original_ids = set(parent)
        meta = {_id_key(r["id"]): r for r in provenance["train_rows"]}
        forced_train.update(_id_key(k) for k in provenance.get("quarantined_ids", []))
        for repair in provenance.get("repairs", []):
            forced_train.add(_id_key(repair["before"]["ID"]))
        for row in rows:
            key = _id_key(row["id"])
            m = meta[key]
            parents = (
                [key]
                if key in original_ids
                else [_id_key(p) for p in m.get("parent_ids", [])]
            )
            if not parents:
                raise ValueError("Screening requires synthetic parent provenance")
            for p in parents[1:]:
                union(parents[0], p)
            family = m.get("family") or m.get("template_rule")
            if family:
                if family in families:
                    union(parents[0], families[family])
                families[family] = parents[0]
            for field in ("source", "target"):
                text = _source_key(row[field])
                if text in texts:
                    union(parents[0], texts[text])
                texts[text] = parents[0]
    groups = defaultdict(list)
    for r in originals:
        groups[find(_id_key(r["id"]))].append(r)
    locked = {find(k) for k in forced_train}
    candidates = [(k, v) for k, v in groups.items() if k not in locked]
    random.Random(42).shuffle(candidates)
    total = Counter(r["label"] for r in originals)
    target = {c: max(1, round(n * 0.2)) for c, n in total.items()}
    selected, counts = set(), Counter()
    for key, group in candidates:
        add = Counter(r["label"] for r in group)
        if any(counts[c] < target[c] for c in add) and all(
            counts[c] + n <= target[c] + 1 and counts[c] + n < total[c]
            for c, n in add.items()
        ):
            selected.add(key)
            counts.update(add)
    validation = [r for r in originals if find(_id_key(r["id"])) in selected]
    if set(counts) != set(range(5)):
        raise ValueError("Grouped screening could not retain every validation subtype")
    fit_sets = []
    for rows, provenance in bundles:
        meta = {_id_key(r["id"]): r for r in provenance["train_rows"]}
        fit = []
        for r in rows:
            key = _id_key(r["id"])
            ps = (
                [key]
                if key in parent
                else [_id_key(p) for p in meta[key]["parent_ids"]]
            )
            if not any(find(p) in selected for p in ps):
                fit.append(r)
        if set(r["label"] for r in fit) != set(range(5)):
            raise ValueError("Grouped screen lost a training subtype")
        fit_sets.append(fit)
    return (
        fit_sets,
        validation,
        dict(
            validation_counts=dict(counts),
            component_sizes=sorted(map(len, groups.values()), reverse=True),
            repair_components_forced_training=len(locked),
        ),
    )


def fit(train, validation, output, deadline):
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        TrainingArguments,
        TrainerCallback,
        set_seed,
    )
    from model.classification import (
        _architecture,
        _class_weights,
        _classification_metrics,
        _encoded,
        _hierarchical_trainer_class,
        _predict,
        _preprocessing,
    )
    from model.encoder import classify

    set_seed(42)
    output.mkdir()
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, revision=REVISION, num_labels=5, attn_implementation="sdpa"
    )
    model.config.classifier_compute_dtype = "float16"
    parameters = sum(p.numel() for p in model.parameters())
    if parameters != 102955013 or parameters > 8_000_000_000:
        raise ValueError("Bami architecture parameter audit differs")
    model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    config = dict(
        task="classification",
        run=dict(seed=42, device="cuda", world_size=1),
        data=dict(validation_fraction=0.2),
        model=dict(name=MODEL, revision=REVISION, backend="encoder"),
        training=dict(max_length=512, effective_batch_size=8, class_weight_power=0.5),
    )
    preprocessing = _preprocessing(config)
    weight_metadata = _class_weights(train, 0.5)
    weights = weight_metadata["weights"]

    class Deadline(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            if time.monotonic() >= deadline:
                raise TimeoutError("Approved total Bami GPU budget exhausted")

    # Labels returned by Trainer are original taxonomy IDs; five head logits are joint-decoded.
    def metrics(evaluation):
        from model.classification import _joint_log_probabilities

        probs = (
            _joint_log_probabilities(torch.tensor(evaluation.predictions))
            .exp()
            .tolist()
        )
        predictions = [
            dict(id=r["id"], label=max(range(5), key=p.__getitem__), probabilities=p)
            for r, p in zip(validation, probs)
        ]
        return {
            "macro_f1": _classification_metrics(validation, predictions)["macro_f1"]
        }

    arguments = TrainingArguments(
        output_dir=str(output / "epochs"),
        num_train_epochs=5,
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        report_to=[],
        fp16=True,
        seed=42,
        data_seed=42,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        disable_tqdm=True,
    )
    trainer = _hierarchical_trainer_class()(
        model=model,
        args=arguments,
        train_dataset=_encoded(tokenizer, train, preprocessing),
        eval_dataset=_encoded(tokenizer, validation, preprocessing),
        data_collator=DataCollatorWithPadding(tokenizer),
        processing_class=tokenizer,
        compute_metrics=metrics,
        callbacks=[Deadline()],
        loss_weights=torch.tensor(weights, device="cuda"),
    )
    began = time.monotonic()
    result = trainer.train()
    checkpoint = output / "classifier"
    trainer.save_model(str(checkpoint))
    tokenizer.save_pretrained(checkpoint)
    write_json(checkpoint / "preprocessing.json", preprocessing)
    write_json(
        checkpoint / "classifier_config.json",
        dict(
            backend="encoder",
            model=config["model"],
            architecture=_architecture(),
            labels=list(range(5)),
            num_labels=5,
            parameters=parameters,
            pad_token_id=tokenizer.pad_token_id,
            tokenizer_length=len(tokenizer),
        ),
    )
    before = _predict(model, tokenizer, validation, preprocessing, "cuda", 2)
    summary = dict(
        training_loss=result.training_loss,
        optimizer_steps=result.global_step,
        best_checkpoint=Path(trainer.state.best_model_checkpoint).name,
        epoch_metrics=trainer.state.log_history,
        parameters=parameters,
        weighting=weight_metadata,
        seconds=time.monotonic() - began,
    )
    trainer.model = trainer.model_wrapped = None
    trainer.callback_handler.model = trainer.callback_handler.optimizer = None
    trainer.callback_handler.lr_scheduler = None
    trainer.optimizer = None
    trainer.accelerator.free_memory()
    del trainer, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    after = classify(checkpoint, validation, "cuda", 2)
    if any(
        a["label"] != b["label"]
        or max(abs(x - y) for x, y in zip(a["probabilities"], b["probabilities"]))
        > 1e-5
        for a, b in zip(before, after)
    ):
        raise ValueError("Saved Bami reload predictions differ")
    summary["classification"] = _classification_metrics(validation, after)
    summary["reload_verified"] = True
    write_json(output / "internal_predictions.json", after)
    write_json(output / "internal_metrics.json", summary)
    write_json(output / "config.json", config)
    write_json(output / "train.json", train)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse verified completed fits; never rerun a completed screen",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=args.resume)
    bundles = [
        load_classifier_split(args.original, p, 0.2, 42)
        for p in (args.control, args.candidate)
    ]
    originals, holdout = split_records(load_records(args.original), 0.2, 42)
    fits, validation, grouping = screening_split(
        originals, [(rows, info["provenance"]) for rows, info in bundles]
    )
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    manifest = dict(
        source_commit=source,
        model=MODEL,
        revision=REVISION,
        seed=42,
        original_sha256=hashlib.sha256(args.original.read_bytes()).hexdigest(),
        grouping=grouping,
        screen_validation_ids=[r["id"] for r in validation],
        fit_counts=list(map(len, fits)),
        bundle_sha256=[
            hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (args.control, args.candidate)
        ],
    )
    manifest_path = args.output / "screen_manifest.json"
    if args.resume:
        previous = json.loads(manifest_path.read_text())
        if any(
            previous[k] != manifest[k]
            for k in (
                "model",
                "revision",
                "original_sha256",
                "bundle_sha256",
                "screen_validation_ids",
                "fit_counts",
            )
        ):
            raise ValueError("Resume screen model/data/split differs")
        write_json(args.output / "resume_manifest.json", manifest)
    else:
        write_json(manifest_path, manifest)
    record(args.output, "started", experiment="approved-two-bami-screens", **manifest)
    consumed = 0.0
    events_path = args.output / "experiment_events.jsonl"
    if args.resume and events_path.exists():
        consumed = max(
            (
                float(e.get("seconds", 0))
                for e in map(json.loads, events_path.read_text().splitlines())
            ),
            default=0,
        )
    began = time.monotonic() - consumed
    summaries = []
    try:
        for name, rows in zip(("control", "candidate"), fits):
            completed = args.output / name / "internal_metrics.json"
            if args.resume and completed.exists():
                summary = json.loads(completed.read_text())
                if summary.get("reload_verified") is not True:
                    raise ValueError("Cannot resume an unverified fit")
                summaries.append(summary)
                continue
            print(
                json.dumps(
                    dict(
                        stage=name,
                        train_rows=len(rows),
                        internal_validation=len(validation),
                    )
                ),
                flush=True,
            )
            summaries.append(fit(rows, validation, args.output / name, began + 30 * 60))
        selected = max(
            range(2), key=lambda i: summaries[i]["classification"]["macro_f1"]
        )
        name = ("control", "candidate")[selected]
        from model.encoder import classify
        from model.classification import _classification_metrics

        predictions = classify(args.output / name / "classifier", holdout, "cuda", 2)
        metrics = _classification_metrics(holdout, predictions)
        write_json(args.output / name / "validation.json", holdout)
        write_json(args.output / name / "metrics.json", dict(classification=metrics))
        write_json(
            args.output / name / "predictions.json",
            [
                dict(
                    id=r["id"],
                    source=r["source"],
                    predicted_class=p["label"],
                    probabilities=p["probabilities"],
                )
                for r, p in zip(holdout, predictions)
            ],
        )
        write_json(
            args.output / name / "run_manifest.json",
            dict(
                task="classification",
                seed=42,
                validation_fraction=0.2,
                data_sha256=manifest["original_sha256"],
                classifier_data_sha256=manifest["bundle_sha256"][selected],
                train_count=len(fits[selected]),
                validation_count=len(holdout),
                hints_enabled=False,
                saved_classifier_reloaded_for_evaluation=True,
                backend="encoder",
            ),
        )
        result = dict(
            selected=name,
            selection_scope="fixed grouped training-only internal validation",
            holdout=metrics,
            internal=[s["classification"] for s in summaries],
            seconds=time.monotonic() - began,
        )
        write_json(args.output / "result.json", result)
        record(
            args.output, "completed", experiment="approved-two-bami-screens", **result
        )
        print(json.dumps(result), flush=True)
    except Exception as exc:
        record(
            args.output,
            "failed",
            experiment="approved-two-bami-screens",
            error_type=type(exc).__name__,
            seconds=time.monotonic() - began,
        )
        raise


if __name__ == "__main__":
    main()
