"""Private reproducible local data probe; run from repository root with PYTHONPATH=."""


def main():
    import os

    os.environ.update(USE_TF="0", USE_FLAX="0", TOKENIZERS_PARALLELISM="false")
    import json, sys
    from pathlib import Path
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from model.data import load_classifier_split
    from model.reproducibility import seed_everything
    from model.classification import _classification_metrics
    from model.runtime import write_json

    arm = sys.argv[1]
    root = Path("runs/diversity-expansion-20261017/bami-comparison") / arm
    m = json.loads((root / "manifest.json").read_text())
    p = Path("dataset") / (
        "kaggle-quality-private-20261017"
        if arm == "baseline"
        else (
            "kaggle-diverse-v2-private-20261017"
            if arm.startswith("diverse")
            else "kaggle-diverse-v3-private-20261017"
        )
    )
    _, info = load_classifier_split(
        p / "VGEC 2026_v1.json", p / "VGEC_classifier.json", 0.2, 42
    )
    rows = info["validation_records"]
    seed_everything(42)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(
        m["model"], revision=m["revision"], local_files_only=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        m["model"],
        revision=m["revision"],
        num_labels=5,
        local_files_only=True,
        attn_implementation="eager",
    )
    model.load_state_dict(
        torch.load(root / "selected.pt", map_location="cpu", weights_only=True),
        strict=True,
    )
    model.cuda().eval()
    pred = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(rows), 4):
            b = tokenizer(
                [r["source"] for r in rows[i : i + 4]],
                truncation=True,
                max_length=256,
                padding=True,
                return_tensors="pt",
            ).to("cuda")
            ps = model(**b).logits.float().softmax(-1).cpu().tolist()
            pred += [
                {
                    "id": r["id"],
                    "label": max(range(5), key=p.__getitem__),
                    "probabilities": p,
                }
                for r, p in zip(rows[i : i + 4], ps)
            ]
    metrics = _classification_metrics(rows, pred)
    assert metrics == json.loads((root / "metrics.json").read_text())
    assert pred == json.loads((root / "predictions.json").read_text())
    write_json(
        root / "reload_verification.json",
        {
            "status": "passed",
            "fresh_process": True,
            "macro_f1": metrics["macro_f1"],
            "exact_saved_predictions_match": True,
        },
    )
    print(arm, "reload passed", metrics["macro_f1"])


if __name__ == "__main__":
    main()
