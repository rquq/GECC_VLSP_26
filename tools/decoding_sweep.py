"""Compare bounded decoding strategies on saved adapters; optional local NF4 inference proxy."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

from model.data import _id_key
from model.runtime import write_json
from model.tuning import decoding_strategies, inference_rows, panel


def cache_identity(adapter, records, proxy, device="cuda"):
    from model.classifier_resume import sha256

    adapter = Path(adapter)
    import importlib.metadata

    versions = {
        name: importlib.metadata.version(name)
        for name in (
            "torch",
            "transformers",
            "peft",
            "tokenizers",
            "bitsandbytes",
            "lm-format-enforcer",
        )
    }
    return {
        "adapter_sha256": sha256(adapter / "adapter_model.safetensors"),
        "decoder_implementation_sha256": sha256(Path(__file__)),
        "versions": versions,
        "device": device,
        "adapter_metadata_sha256": {
            p.name: sha256(p)
            for p in sorted(adapter.iterdir())
            if p.is_file() and p.suffix in {".json", ".txt"}
        },
        "validation_sha256": hashlib.sha256(
            json.dumps(records, ensure_ascii=False).encode()
        ).hexdigest(),
        "proxy": proxy,
        "precision": (
            "NF4 inference proxy" if proxy else "saved native standard-LoRA inference"
        ),
    }


def _load(adapter, device, proxy):
    import torch
    from model.reproducibility import seed_everything

    seed_everything(42)
    if not proxy:
        from model.correction import load_corrector

        return load_corrector(adapter, device)
    if device != "cuda":
        raise ValueError("The local NF4 proxy requires CUDA")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    settings = json.loads((adapter / "loading.json").read_text())
    base = AutoModelForCausalLM.from_pretrained(
        settings["name"],
        revision=settings["revision"],
        torch_dtype=torch.float16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        local_files_only=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        ),
    )
    # This is inference with an existing standard-LoRA adapter. k-bit training
    # preparation would unnecessarily promote the large vocabulary tensors.
    model = PeftModel.from_pretrained(base, adapter, is_trainable=False)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True)
    return model, tokenizer, json.loads((adapter / "inference.json").read_text())


def _schema(tokenizer):
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.integrations.transformers import (
        build_transformers_prefix_allowed_tokens_fn,
    )

    schema = {
        "type": "object",
        "properties": {"corrected_sentence": {"type": "string", "minLength": 1}},
        "required": ["corrected_sentence"],
        "additionalProperties": False,
    }
    return build_transformers_prefix_allowed_tokens_fn(
        tokenizer, JsonSchemaParser(schema)
    )


def _candidate_score(model, tokenizer, prefix, target):
    import torch

    answer = tokenizer.encode(
        json.dumps({"corrected_sentence": target}, ensure_ascii=False),
        add_special_tokens=False,
    )
    answer.append(tokenizer.eos_token_id)
    device = model.get_input_embeddings().weight.device
    ids = torch.tensor([prefix + answer], device=device)
    logits = model(
        input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False
    ).logits
    labels = torch.tensor(answer, device=logits.device)
    total = 0.0
    # Avoid a large FP32 sequence*vocabulary allocation on the local8GB GPU.
    for start in range(0, len(answer), 8):
        end = min(start + 8, len(answer))
        values = logits[0, len(prefix) - 1 + start : len(prefix) - 1 + end].float()
        total -= torch.nn.functional.cross_entropy(
            values, labels[start:end], reduction="sum"
        ).item()
    return total / len(answer)


def decode(model, tokenizer, record, inference, strategy):
    import torch
    from model.correction import _prompt_ids, _parse_correction
    from model.reproducibility import seed_everything

    seed_everything(42)
    prefix = _prompt_ids(tokenizer, record["source"], 0, inference)
    if len(prefix) > inference["max_length"]:
        raise ValueError("Prompt exceeds saved context limit")
    kwargs = {
        **inference["generation"],
        **strategy["generation"],
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": model.generation_config.eos_token_id or tokenizer.eos_token_id,
    }
    if strategy.get("schema"):
        kwargs["prefix_allowed_tokens_fn"] = _schema(tokenizer)
    ids = torch.tensor([prefix], device=model.get_input_embeddings().weight.device)

    def generate(options):
        output = model.generate(
            input_ids=ids, attention_mask=torch.ones_like(ids), **options
        )
        sequences = output.sequences if hasattr(output, "sequences") else output
        return [
            tokenizer.decode(seq[len(prefix) :], skip_special_tokens=True).strip()
            for seq in sequences
        ]

    with torch.inference_mode():
        raw = generate(kwargs)
        candidates = []
        for value in raw:
            target, error = _parse_correction(value)
            if error is None:
                candidates.append({"target": target, "raw_generation": value})
        if not candidates:
            retry = {
                **kwargs,
                "do_sample": False,
                "num_beams": 4,
                "num_beam_groups": 1,
                "diversity_penalty": 0.0,
                "num_return_sequences": 1,
                "length_penalty": 0.0,
                "early_stopping": True,
                "max_new_tokens": max(
                    kwargs["max_new_tokens"], inference["max_length"] + 64
                ),
                "prefix_allowed_tokens_fn": _schema(tokenizer),
            }
            retry.pop("penalty_alpha", None)
            retry.pop("low_memory", None)
            value = generate(retry)[0]
            target, error = _parse_correction(value)
            if error is not None:
                raise ValueError("Schema recovery did not produce a valid correction")
            candidates = [
                {"target": target, "raw_generation": value, "schema_retry": True}
            ]
        if strategy.get("edit_rerank"):
            from model.correction_reranking import edit_cost

            candidates.append(
                {
                    "target": record["source"],
                    "raw_generation": None,
                    "origin": "unchanged_source",
                }
            )
            for candidate in candidates:
                ll = _candidate_score(model, tokenizer, prefix, candidate["target"])
                cost = edit_cost(record["source"], candidate["target"]) / max(
                    len(record["source"]), 1
                )
                candidate["rank_score"] = ll - strategy["edit_penalty"] * cost
            selected = max(candidates, key=lambda c: c["rank_score"])
        else:
            selected = candidates[0]
    return {
        "id": record["id"],
        "target": selected["target"],
        "raw_generation": selected["raw_generation"],
        "parse_error": None,
        "schema_retry": selected.get("schema_retry", False),
        **({"candidates": candidates} if strategy.get("edit_rerank") else {}),
    }


def measured(records, predictions, classifier):
    from model.metrics import experiment_report

    indexed = {_id_key(p["id"]): p for p in predictions}
    rows = [
        {**indexed[_id_key(r["id"])], "label": classifier[_id_key(r["id"])]}
        for r in records
    ]
    return experiment_report(records, rows)


def sweep(
    adapter,
    classifier_run,
    output,
    *,
    device="cuda",
    proxy=False,
    per_class=2,
    finalists=3,
    strategies=None
):
    import torch
    from model.tracking import publish_stage_result

    adapter, classifier_run, output = Path(adapter), Path(classifier_run), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = json.loads((classifier_run / "validation.json").read_text())
    classifier = {
        _id_key(r["id"]): r["predicted_class"]
        for r in json.loads((classifier_run / "predictions.json").read_text())
    }
    provenance = cache_identity(adapter, records, proxy, device)
    old = output / "provenance.json"
    if old.exists() and json.loads(old.read_text()) != provenance:
        raise ValueError("Sweep cache provenance changed")
    write_json(old, provenance)
    model, tokenizer, inference = _load(adapter, device, proxy)
    matrix = decoding_strategies() if strategies is None else strategies
    reports = {}

    def execute(strategy, selection, phase):
        name = strategy["name"]
        folder = output / name
        folder.mkdir(exist_ok=True)
        path = folder / "predictions.json"
        bound = folder / "strategy.json"
        if bound.exists() and json.loads(bound.read_text()) != strategy:
            raise ValueError("Strategy cache differs")
        write_json(bound, strategy)
        predictions = json.loads(path.read_text()) if path.exists() else []
        indexed = {_id_key(p["id"]): p for p in predictions}
        start = time.monotonic()
        try:
            for row in inference_rows(selection):
                if _id_key(row["id"]) in indexed:
                    continue
                pred = decode(model, tokenizer, row, inference, strategy)
                predictions.append(pred)
                indexed[_id_key(row["id"])] = pred
                write_json(path, predictions)
                from model.tracking import publish_progress

                publish_progress(
                    "decode-" + name,
                    "decoding",
                    {
                        "completed_rows": len(predictions),
                        "requested_rows": len(selection),
                    },
                )
                write_json(
                    output / "progress.json",
                    {
                        "phase": phase,
                        "strategy": name,
                        "completed": len(predictions),
                        "requested": len(selection),
                        "elapsed_seconds": time.monotonic() - start,
                        "allocated_gib": (
                            torch.cuda.memory_allocated() / 1024**3
                            if device == "cuda"
                            else None
                        ),
                    },
                )
            metrics = measured(
                selection, [indexed[_id_key(r["id"])] for r in selection], classifier
            )
            report = {
                "status": "completed",
                "phase": phase,
                "strategy": strategy,
                "metrics": metrics,
                "seconds_this_phase": time.monotonic() - start,
                "proxy": proxy,
                "peak_allocated_gib": (
                    torch.cuda.max_memory_allocated() / 1024**3
                    if device == "cuda"
                    else None
                ),
            }
            publish_stage_result("decode-" + name, metrics)
        except (
            torch.cuda.OutOfMemoryError,
            RuntimeError,
            ValueError,
            TypeError,
        ) as error:
            report = {
                "status": "failed",
                "phase": phase,
                "strategy": strategy,
                "error_type": type(error).__name__,
                "error": str(error),
                "completed_rows": len(predictions),
                "proxy": proxy,
            }
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
        write_json(folder / (phase + ".json"), report)
        write_json(output / "latest_result.json", report)
        print(
            json.dumps({k: v for k, v in report.items() if k != "metrics"}), flush=True
        )
        return report

    screening = panel(records, per_class)
    for strategy in matrix:
        reports[strategy["name"]] = execute(strategy, screening, "screen")

    def key(report):
        m = report["metrics"]
        return (
            m["corrections"]["error_only"]["gleu_plus"],
            m["combined_score"],
            m["clean_preservation"]["exact_source_preservation"],
        )

    successful = sorted(
        [r for r in reports.values() if r["status"] == "completed"],
        key=key,
        reverse=True,
    )
    if not successful:
        raise RuntimeError("No decoding strategy completed the screening panel")
    names = (
        ["greedy_reference"]
        if reports.get("greedy_reference", {}).get("status") == "completed"
        else []
    )
    for report in successful:
        if len(names) >= finalists:
            break
        if report["strategy"]["name"] not in names:
            names.append(report["strategy"]["name"])
    full = [
        execute(next(s for s in matrix if s["name"] == name), records, "full")
        for name in names
    ]
    completed = [r for r in full if r["status"] == "completed"]
    if not completed:
        raise RuntimeError("No strategy completed full validation")
    best = max(completed, key=key)
    report = {
        "status": "completed",
        "provenance": provenance,
        "screen_rows": len(screening),
        "full_rows": len(records),
        "screen": list(reports.values()),
        "full": full,
        "best": best,
        "scope": "development-set decoder selection, not an unseen score; proxy runs cannot select native deployment",
    }
    write_json(output / "summary.json", report)
    del model, tokenizer
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--classifier-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--proxy", action="store_true")
    parser.add_argument("--per-class", type=int, default=2)
    parser.add_argument("--finalists", type=int, default=3)
    parser.add_argument("--strategies")
    parser.add_argument("--public-data")
    parser.add_argument("--strategy")
    args = parser.parse_args()
    if args.public_data:
        if args.proxy:
            raise ValueError("Quantized local proxies cannot export native submissions")
        from model.data import load_records

        records = load_records(args.public_data, require_labels=False)
        model, tokenizer, inference = _load(Path(args.adapter), args.device, False)
        strategy = json.loads(Path(args.strategy).read_text())
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        predictions = []
        for row in inference_rows(records):
            predictions.append(decode(model, tokenizer, row, inference, strategy))
            write_json(output / "predictions.json", predictions)
            write_json(
                output / "progress.json",
                {"completed": len(predictions), "total": len(records)},
            )
        return
    strategies = (
        json.loads(Path(args.strategies).read_text()) if args.strategies else None
    )
    sweep(
        args.adapter,
        args.classifier_run,
        args.output_dir,
        device=args.device,
        proxy=args.proxy,
        per_class=args.per_class,
        finalists=args.finalists,
        strategies=strategies,
    )


if __name__ == "__main__":
    main()
