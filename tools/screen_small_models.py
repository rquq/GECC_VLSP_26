"""Committed local model screen; private data/artifacts, training-only selection.

Run with USE_TF=0 USE_FLAX=0 PYTHONPATH=. and the project Python environment.
Each trial is a separate process. No Kaggle notebook settings are introduced.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from model.data import load_classifier_split, load_records, load_training_split, split_records
from model.metrics import experiment_report
from model.runtime import CompleteEpochMixin, write_json
from tools.local_classifier import screening_split

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs/local-small-models-20261014"
DATA = ROOT / "dataset/kaggle-qwen-ready-20261013"
CONFIG = ROOT / "configs/local_screen.json"


def weighted_classification_loss(logits, labels, weights):
    """Preserve weights when accumulating single-row microbatches.

    PyTorch's weighted mean divides by the batch's weight sum, cancelling
    class weighting entirely for microbatch one. Average weighted row losses.
    """
    import torch.nn.functional as F
    return F.cross_entropy(logits.float(), labels, weight=weights.to(labels.device), reduction="none").mean()


def prepare():
    OUTPUT.mkdir(exist_ok=True)
    original = DATA / "VGEC 2026_v1.json"
    originals, holdout = split_records(load_records(original), .2, 42)
    classifier, info = load_classifier_split(original, DATA / "VGEC_classifier.json", .2, 42)
    corrector, validation, cinfo = load_training_split(DATA / "VGEC_balanced.json", .2, 42)
    if validation != holdout:
        raise ValueError("Correction holdout differs")
    manifest = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "seed": 42, "holdout_count": len(holdout), "tasks": {},
                "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
                "data_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (original, DATA / "VGEC_classifier.json", DATA / "VGEC_balanced.json")}}
    for task, rows, provenance in (("classification", classifier, info["provenance"]),
                                   ("correction", corrector, cinfo["provenance"])):
        fits, internal, grouping = screening_split(originals, [(rows, provenance)])
        write_json(OUTPUT / f"{task}_split.json", {"fit": fits[0], "internal": internal,
                                                   "full_train": rows, "holdout": holdout})
        manifest["tasks"][task] = {"fit_count": len(fits[0]), "internal_count": len(internal),
                                   "full_train_count": len(rows), "grouping": grouping,
                                   "internal_ids": [r["id"] for r in internal]}
    write_json(OUTPUT / "screen_manifest.json", manifest)
    print(json.dumps({"prepared": manifest["tasks"]}), flush=True)


def text_sources(rows, candidate):
    if not candidate.get("segmentation"):
        return [r["source"] for r in rows]
    runtime = json.loads((OUTPUT / "runtime/runtime.json").read_text())
    os.environ["JAVA_HOME"] = str(Path(runtime["java"]).parents[1])
    os.environ["PATH"] = str(Path(runtime["java"]).parent) + os.pathsep + os.environ["PATH"]
    import py_vncorenlp
    segmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=runtime["segmenter"],
                                      max_heap_size="-Xmx512m")
    return [" ".join(segmenter.word_segment(r["source"])) for r in rows]


def train(candidate, final=False, epochs=None):
    import gc
    import numpy as np
    import torch
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM,
                              Trainer, Seq2SeqTrainer, TrainingArguments, Seq2SeqTrainingArguments,
                              DataCollatorWithPadding, DataCollatorForSeq2Seq, TrainerCallback, set_seed)
    from model.classification import _classification_metrics

    cfg = json.loads(CONFIG.read_text())
    source_commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    code_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    task = candidate["task"]
    split = json.loads((OUTPUT / f"{task}_split.json").read_text())
    train_rows, validation = (split["full_train"], split["holdout"]) if final else (split["fit"], split["internal"])
    # Final fits do not access holdout until the predetermined number of epochs is finished.
    epochs = epochs or cfg["epochs"][task]
    output = OUTPUT / (candidate["name"] + ("-final" if final else ""))
    output.mkdir(exist_ok=False)
    began = time.monotonic()
    set_seed(42)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    tokenizer = AutoTokenizer.from_pretrained(candidate["model"], revision=candidate["revision"])
    sources = text_sources(train_rows + validation, candidate)
    # py_vncorenlp changes cwd; all subsequent paths are absolute.
    tokens = tokenizer(sources, truncation=True, max_length=cfg["max_length"])
    if task == "classification":
        model = AutoModelForSequenceClassification.from_pretrained(candidate["model"], revision=candidate["revision"],
                                                                  num_labels=5, attn_implementation="sdpa")
        labels = [r["label"] for r in train_rows + validation]
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(candidate["model"], revision=candidate["revision"],
                                                   torch_dtype=torch.bfloat16 if candidate.get("lora") else torch.float32)
        labels = tokenizer(text_target=[r["target"] for r in train_rows + validation],
                           truncation=True, max_length=cfg["max_length"])["input_ids"]
    base_parameters = sum(p.numel() for p in model.parameters())
    if candidate.get("lora"):
        from peft import get_peft_model, LoraConfig, TaskType
        model = get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM,
                               r=16, lora_alpha=32, lora_dropout=.05, target_modules=["q", "v"]))
    parameters = sum(p.numel() for p in model.parameters())
    if parameters > 8_000_000_000:
        raise ValueError("Competition size cap exceeded")
    if base_parameters > 200_000_000:
        model.gradient_checkpointing_enable()
        if candidate.get("lora"):
            model.enable_input_require_grads()
    model.config.use_cache = False
    model.to("cuda")
    encoded = [{**{k: values[i] for k, values in tokens.items()}, "labels": label}
               for i, label in enumerate(labels)]

    def predictions(model, rows, row_sources):
        model.eval()
        results = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for start in range(0, len(rows), 2):
                batch = tokenizer(row_sources[start:start+2], truncation=True, max_length=cfg["max_length"],
                                  padding=True, return_tensors="pt").to("cuda")
                if task == "classification":
                    probabilities = model(**batch).logits.float().softmax(-1)
                    if not torch.isfinite(probabilities).all():
                        raise ValueError("Nonfinite classification output")
                    for row, probs in zip(rows[start:start+2], probabilities.cpu().tolist()):
                        results.append({"id": row["id"], "label": int(np.argmax(probs)),
                                        "probabilities": probs, "target": row["source"]})
                else:
                    values = model.generate(**batch, max_new_tokens=cfg["max_length"], num_beams=1,
                                            do_sample=False, use_cache=True)
                    for row, value in zip(rows[start:start+2], tokenizer.batch_decode(values, skip_special_tokens=True)):
                        results.append({"id": row["id"], "label": row["label"], "target": value.strip()})
        return results

    def metrics(evaluation):
        if task == "classification":
            probs = torch.tensor(evaluation.predictions).float().softmax(-1).tolist()
            return {"macro_f1": _classification_metrics(validation,
                    [{"id": r["id"], "label": int(np.argmax(p))} for r, p in zip(validation, probs)])["macro_f1"]}
        decoded = tokenizer.batch_decode(np.where(evaluation.predictions != -100,
                                                  evaluation.predictions, tokenizer.pad_token_id), skip_special_tokens=True)
        report = experiment_report(validation, [{"id": r["id"], "label": r["label"], "target": p.strip()}
                                               for r, p in zip(validation, decoded)])
        return {"gleu_plus": report["corrections"]["all"]["gleu_plus"],
                "error_gleu_plus": report["corrections"]["error_only"]["gleu_plus"],
                "clean_preservation": report["clean_preservation"]["exact_source_preservation"]}

    class Deadline(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            if time.monotonic() - began > cfg["max_trial_seconds"]:
                raise TimeoutError("Local trial budget exhausted")

    weights = torch.tensor([len(train_rows)/(5*Counter(r["label"] for r in train_rows)[i]) for i in range(5)]).sqrt()
    class WeightedTrainer(CompleteEpochMixin, Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = weighted_classification_loss(outputs.logits, labels, weights)
            return (loss, outputs) if return_outputs else loss

    args = dict(output_dir=str(output / "epochs"), num_train_epochs=epochs,
                learning_rate=candidate["learning_rate"], per_device_train_batch_size=1,
                per_device_eval_batch_size=2, gradient_accumulation_steps=8, bf16=True,
                optim="adamw_bnb_8bit" if base_parameters > 200_000_000 else "adamw_torch",
                weight_decay=.01, warmup_ratio=.1, seed=42, data_seed=42,
                report_to=[], logging_steps=20, disable_tqdm=True, save_total_limit=1,
                eval_strategy="no" if final else "epoch", save_strategy="no" if final else "epoch",
                load_best_model_at_end=not final, metric_for_best_model="macro_f1" if task == "classification" else "gleu_plus",
                greater_is_better=True)
    if task == "classification":
        arguments, trainer_cls, collator = TrainingArguments(**args), WeightedTrainer, DataCollatorWithPadding(tokenizer)
    else:
        arguments = Seq2SeqTrainingArguments(**args, predict_with_generate=True, generation_max_length=cfg["max_length"],
                                             generation_num_beams=1)
        class CompleteSeq2SeqTrainer(CompleteEpochMixin, Seq2SeqTrainer):
            pass
        trainer_cls, collator = CompleteSeq2SeqTrainer, DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = trainer_cls(model=model, args=arguments, train_dataset=encoded[:len(train_rows)],
                          eval_dataset=None if final else encoded[len(train_rows):],
                          data_collator=collator, processing_class=tokenizer, compute_metrics=metrics,
                          callbacks=[Deadline()])
    result = trainer.train()
    checkpoint = output / "checkpoint"
    trainer.save_model(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    before = predictions(model, validation, sources[len(train_rows):])
    selected_epoch = epochs if final else int(Path(trainer.state.best_model_checkpoint).name.split("-")[-1])
    if not final:
        selected_epoch = next(round(e["epoch"]) for e in trainer.state.log_history
                              if e.get("step") == selected_epoch and ("eval_macro_f1" in e or "eval_gleu_plus" in e))
    summary = {"candidate": candidate, "source_commit": source_commit, "code_sha256": code_sha256,
               "task": task, "scope": "frozen_holdout" if final else "training_only_internal",
               "train_count": len(train_rows), "validation_count": len(validation),
               "epochs": epochs, "selected_epoch": selected_epoch, "optimizer_steps": result.global_step,
               "training_loss": result.training_loss, "base_parameters": base_parameters,
               "parameters_with_adapters": parameters, "epoch_history": trainer.state.log_history,
               "seconds": time.monotonic() - began, "peak_vram_bytes": torch.cuda.max_memory_allocated(),
               "preprocessing": "VnCoreNLP RDR word segmentation" if candidate.get("segmentation") else "raw syllables",
               "classification": _classification_metrics(validation, before) if task == "classification" else None}
    if task == "correction":
        report = experiment_report(validation, before)
        summary.update(corrections=report["corrections"], clean_preservation=report["clean_preservation"])
        summary["classification"] = None  # Gold labels here are alignment only, never a Task1 result.
    write_json(output / "predictions.json", before)
    trainer.model = trainer.model_wrapped = None
    trainer.callback_handler.model = trainer.callback_handler.optimizer = None
    trainer.callback_handler.lr_scheduler = None
    trainer.optimizer = None
    trainer.accelerator.free_memory()
    del trainer, model, collator
    gc.collect(); torch.cuda.empty_cache()
    if candidate.get("lora"):
        from peft import PeftModel
        model = AutoModelForSeq2SeqLM.from_pretrained(candidate["model"], revision=candidate["revision"], torch_dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(model, checkpoint)
    else:
        cls = AutoModelForSequenceClassification if task == "classification" else AutoModelForSeq2SeqLM
        model = cls.from_pretrained(checkpoint, local_files_only=True)
    model.to("cuda")
    after = predictions(model, validation, sources[len(train_rows):])
    if any(a["label"] != b["label"] or a["target"] != b["target"] for a, b in zip(before, after)):
        raise ValueError("Saved checkpoint reload predictions differ")
    summary["reload_verified"] = True
    summary["seconds"] = time.monotonic()-began
    write_json(output / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "epoch_history"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", nargs="?")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    if args.prepare:
        prepare(); return
    candidate = next(c for c in json.loads(CONFIG.read_text())["candidates"] if c["name"] == args.candidate)
    train(candidate, args.final, args.epochs)


if __name__ == "__main__":
    main()
