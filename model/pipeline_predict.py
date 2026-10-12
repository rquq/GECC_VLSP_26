"""Predict with a saved QLoRA classifier and an independently fitted correction adapter."""
import argparse
import gc
import json
from pathlib import Path

from .data import load_records
from .runtime import write_json
from .prediction_export import comparison_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--hint-mode", choices=["predicted", "none"], help="Override saved hint conditioning")
    parser.add_argument("--rerank", action="store_true", help="Score beam candidates and unchanged source")
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    records = load_records(args.data, require_labels=args.evaluate)
    if not records:
        raise ValueError("Input must contain at least one record")
    from .classification import classify
    hints = classify(args.classifier, records, device=args.device, batch_size=args.batch_size)
    # Stage 1 releases its weights before loading the independently fitted corrector.
    gc.collect()
    import torch
    if args.device == "cuda":
        torch.cuda.empty_cache()
    from .correction import load_corrector, correct
    model, tokenizer, inference = load_corrector(args.adapter, device=args.device)
    if args.hint_mode:
        inference["hint_mode"] = args.hint_mode
    if args.rerank:
        inference["reranking"] = {**inference.get("reranking", {}), "enabled": True}
    predictions = correct(model, tokenizer, records, hints, inference)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, comparison_rows(records, predictions))
    if args.evaluate:
        from .metrics import experiment_report
        metrics = experiment_report(records, predictions)
        write_json(output.with_suffix(".metrics.json"), metrics)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Predictions: {output}")


if __name__ == "__main__":
    main()
