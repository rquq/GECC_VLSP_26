"""Batched private correction with append-only progress and OOM batch reduction."""
import argparse
import json
from pathlib import Path

from model.runtime import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    import torch
    from model.correction import load_corrector, correct
    from model.data import load_records, _id_key
    from model.tracking import publish_progress

    records = load_records(args.records, require_labels=False)
    if any(row["label"] is not None or row["target"] is not None for row in records):
        raise ValueError("Private worker accepts only unlabeled inference rows")
    model, tokenizer, inference = load_corrector(args.adapter)
    if inference.get("hint_mode") != "none":
        raise ValueError("This private worker requires independent no-hint correction")
    # Similar-length batches reduce padding without altering final typed-ID order.
    ordered = sorted(records, key=lambda row: len(row["source"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cursor, batch_size, predictions = 0, args.batch_size, []
    with (args.output_dir / "predictions.jsonl").open("x", encoding="utf-8") as journal:
        while cursor < len(ordered):
            batch = ordered[cursor:cursor + batch_size]
            hints = [{"id": row["id"], "label": 0, "probabilities": [1, 0, 0, 0, 0]} for row in batch]
            try:
                result = correct(model, tokenizer, batch, hints, {**inference, "batch_size": batch_size})
            except torch.cuda.OutOfMemoryError:
                if batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                torch.cuda.empty_cache()
                print(json.dumps({"status": "reduce_inference_batch", "batch_size": batch_size}), flush=True)
                continue
            predictions.extend(result)
            for row in result:
                journal.write(json.dumps(row, ensure_ascii=False) + "\n")
            journal.flush()
            cursor += len(batch)
            write_json(args.output_dir / "progress.json", {"completed": cursor, "total": len(records),
                                                          "batch_size": batch_size})
            if cursor % 16 < len(batch) or cursor == len(records):
                report = {"completed_rows": cursor, "total_rows": len(records), "batch_size": batch_size}
                print(json.dumps({"private_correction": report}), flush=True)
                publish_progress("private-corrector", "decoding",
                                 {"completed_rows": cursor, "requested_rows": len(records)})
    indexed = {_id_key(row["id"]): row for row in predictions}
    write_json(args.output_dir / "predictions.json", [indexed[_id_key(row["id"])] for row in records])


if __name__ == "__main__":
    main()
