"""Reload a saved adapter and emit proposed VLSP predictions, not an official submission."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    args = parser.parse_args()
    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .data import load_records
    from .metrics import score
    from .modeling import generate_predictions
    from .train import write_json

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; pass --device cpu for CPU inference")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    records = load_records(args.data, require_labels=args.evaluate)
    if args.limit:
        records = records[:args.limit]
    config = PeftConfig.from_pretrained(args.adapter)
    base = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path, revision=config.revision,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
        attn_implementation="sdpa",
    ).to(args.device)
    model = PeftModel.from_pretrained(base, args.adapter)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    predictions = generate_predictions(model, tokenizer, records, args.max_new_tokens)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, [{"ID": r["id"], "Output": {"Error_Type_Id": r["label"],
                         "Fixed Sentence": r["target"]}} for r in predictions])
    write_json(output.with_suffix(".diagnostics.json"), predictions)
    if args.evaluate:
        write_json(output.with_suffix(".metrics.json"), score(records, predictions))
    print(json.dumps({"predictions": len(predictions), "invalid_generations": sum(r["parse_error"] is not None for r in predictions),
                      "output": str(output)}))


if __name__ == "__main__":
    main()
