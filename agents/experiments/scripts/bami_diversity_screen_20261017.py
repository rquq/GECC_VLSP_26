"""Private reproducible local data probe; run from repository root with PYTHONPATH=."""


def main():
    import os

    os.environ.update(USE_TF="0", USE_FLAX="0", TOKENIZERS_PARALLELISM="false")
    import json, math, hashlib, sys, time
    from pathlib import Path
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        DataCollatorWithPadding,
    )
    from model.reproducibility import seed_everything
    from model.data import load_classifier_split
    from model.classification import _classification_metrics
    from tools.screen_small_models import weighted_classification_loss
    from model.runtime import write_json

    arm = sys.argv[1]
    root = Path("runs/diversity-expansion-20261017/bami-comparison") / arm
    root.mkdir(parents=True, exist_ok=False)
    seed_everything(42)
    torch.set_num_threads(2)
    began = time.monotonic()
    p = Path("dataset") / (
        "kaggle-quality-private-20261017"
        if arm == "baseline"
        else (
            "kaggle-diverse-v2-private-20261017"
            if arm.startswith("diverse")
            else "kaggle-diverse-v3-private-20261017"
        )
    )
    rows, info = load_classifier_split(
        p / "VGEC 2026_v1.json", p / "VGEC_classifier.json", 0.2, 42
    )
    validation = info["validation_records"]
    name = "Qualcomm-AI-Research/BamiBERT"
    revision = "57bc1340debbe4e348ec549047a763caebe4a977"
    tokenizer = AutoTokenizer.from_pretrained(
        name, revision=revision, local_files_only=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        name,
        revision=revision,
        num_labels=5,
        local_files_only=True,
        attn_implementation="eager",
    ).cuda()
    assert sum(x.numel() for x in model.parameters()) < 8_000_000_000
    collate = DataCollatorWithPadding(tokenizer)

    def encode(rs):
        t = tokenizer([r["source"] for r in rs], truncation=True, max_length=256)
        return [
            {**{k: v[i] for k, v in t.items()}, "labels": r["label"]}
            for i, r in enumerate(rs)
        ]

    encoded = encode(rows)
    eval_encoded = encode(validation)
    max_steps = math.ceil(len(rows) / 8) * 3 if arm.endswith("_passes") else 420
    from collections import Counter

    counts = Counter(r["label"] for r in rows)
    weights = torch.tensor(
        [len(rows) / (5 * counts[i]) for i in range(5)], device="cuda"
    ).sqrt()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    def evaluate():
        model.eval()
        pred = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(validation), 4):
                batch = {
                    k: v.cuda() for k, v in collate(eval_encoded[i : i + 4]).items()
                }
                batch.pop("labels")
                ps = model(**batch).logits.float().softmax(-1).cpu().tolist()
                pred += [
                    {
                        "id": r["id"],
                        "label": max(range(5), key=p.__getitem__),
                        "probabilities": p,
                    }
                    for r, p in zip(validation[i : i + 4], ps)
                ]
        return _classification_metrics(validation, pred), pred

    rng = torch.Generator().manual_seed(42)
    order = []
    position = 0
    history = []
    best = -1
    for step in range(1, max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        for micro in range(4):
            if position + 2 > len(order):
                order = torch.randperm(len(encoded), generator=rng).tolist()
                position = 0
            selected = [encoded[j] for j in order[position : position + 2]]
            position += 2
            batch = {k: v.cuda() for k, v in collate(selected).items()}
            labels = batch.pop("labels")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = (
                    weighted_classification_loss(model(**batch).logits, labels, weights)
                    / 4
                )
            loss.backward()
            loss_total += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        factor = step / 40 if step < 40 else (max_steps - step) / (max_steps - 40)
        for group in optimizer.param_groups:
            group["lr"] = 2e-5 * factor
        optimizer.step()
        if step % 60 == 0 or step == max_steps:
            metrics, pred = evaluate()
            entry = {
                "step": step,
                "loss": loss_total,
                "macro_f1": metrics["macro_f1"],
                "micro_f1": metrics["micro_f1"],
                "elapsed_seconds": time.monotonic() - began,
            }
            history.append(entry)
            write_json(root / "history.json", history)
            print(json.dumps({"arm": arm, **entry}), flush=True)
            if metrics["macro_f1"] > best:
                best = metrics["macro_f1"]
                write_json(root / "metrics.json", metrics)
                write_json(root / "predictions.json", pred)
                torch.save(model.state_dict(), root / "selected.pt")
        if time.monotonic() - began > 1800:
            raise TimeoutError("Bounded comparison timed out")
    manifest = {
        "arm": arm,
        "model": name,
        "revision": revision,
        "seed": 42,
        "optimizer_steps": max_steps,
        "effective_batch": 8,
        "microbatch": 2,
        "lr": 2e-5,
        "precision": "full encoder finetune with BF16 autocast, no LoRA or quantization",
        "max_length": 256,
        "validation_ids": [r["id"] for r in validation],
        "train_rows": len(rows),
        "dataset_sha256": hashlib.sha256(
            (p / "VGEC_classifier.json").read_bytes()
        ).hexdigest(),
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "elapsed_seconds": time.monotonic() - began,
        "best_macro_f1": best,
        "scope": "Exploratory same78 repeatedly used development set; encoder/data screening, not a7B or independent quality gain",
    }
    write_json(root / "manifest.json", manifest)
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
