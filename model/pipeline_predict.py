"""Export organizer submissions using a saved classifier and correction adapter.

Comparison diagnostics and local (not official) evaluation remain separate files.
"""
import argparse
import gc
import json
from pathlib import Path

from .data import _validate_records, load_records
from .runtime import resolve_output, resolve_submission_data, write_json
from .prediction_export import comparison_rows, submission_rows, write_submission


def _classify_isolated(classifier, records, device, batch_size):
    """Exit the classifier process before the corrector allocates CUDA weights."""
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    from .classification import classify

    with ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn")) as worker:
        return worker.submit(classify, classifier, records, device, batch_size).result()


def run_inference(classifier, adapter, records, output, *, device="cuda", batch_size=1,
                  evaluate=False, input_scope="explicit_submission", input_path=None,
                  hint_mode=None, rerank=False):
    """Reload both saved stages and write submission, diagnostics, and metadata.

    Gold labels/targets are retained only for comparison/evaluation, never passed
    to either inference stage. Invalid generations fail export without fallback.
    """
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch-size must be positive")
    _validate_records(records, require_labels=evaluate)
    if not records:
        raise ValueError("Input must contain at least one record")
    output = Path(output).expanduser().resolve()
    if output.name != "results.json":
        raise ValueError("Organizer prediction file must be named results.json; choose a new directory for another export")
    paths = {"submission": str(output),
             "zip": str(output.with_suffix(".zip")),
             "comparisons": str(output.with_suffix(".comparisons.json")),
             "classification": str(output.with_suffix(".classification.json")),
             "evaluation": str(output.with_suffix(".evaluation.json"))}
    if evaluate:
        paths["metrics"] = str(output.with_suffix(".metrics.json"))
    if len(set(paths.values())) != len(paths):
        raise ValueError("Submission output must not collide with companion artifact paths")
    if any(Path(path).exists() for path in paths.values()):
        raise ValueError("Inference output artifacts must not already exist; choose a new output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    inference_records = [{"id": row["id"], "source": row["source"], "label": None,
                          "target": None} for row in records]
    from .classification import classify
    hints = (_classify_isolated(classifier, inference_records, device, batch_size)
             if device == "cuda" else
             classify(classifier, inference_records, device=device, batch_size=batch_size))
    write_json(paths["classification"], hints)
    # Stage 1 releases its weights before loading the independently fitted corrector.
    gc.collect()
    import torch
    if device == "cuda":
        torch.cuda.empty_cache()
    from .correction import load_corrector, correct
    model, tokenizer, inference = load_corrector(adapter, device=device)
    if hint_mode:
        inference["hint_mode"] = hint_mode
    if rerank:
        inference["reranking"] = {**inference.get("reranking", {}), "enabled": True}
    try:
        predictions = correct(model, tokenizer, inference_records, hints, inference)
    finally:
        del model, tokenizer
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    write_json(paths["comparisons"], comparison_rows(records, predictions))
    metadata = {"input_scope": input_scope, "input_path": str(input_path) if input_path else None,
                "count": len(records), "evaluated": evaluate,
                "classifier": str(classifier), "adapter": str(adapter), "artifacts": paths,
                "submission_schema": "ID/Output: Error_Type_Id (integer 0..4), Fixed Sentence (singleton list of nonblank text)"}
    export_error = None
    try:
        submission = submission_rows(records, predictions)
    except ValueError as error:
        export_error = error
    if evaluate and all(isinstance(row.get("target"), str) for row in predictions):
        from .metrics import experiment_report
        metrics = experiment_report(records, predictions)
        write_json(paths["metrics"], metrics)
        metadata.update(combined_score=metrics["combined_score"],
                        combined_score_metadata=metrics["combined_score_metadata"])
        from .tracking import result_metrics
        print(json.dumps(result_metrics(metrics), ensure_ascii=False), flush=True)
    elif evaluate:
        paths.pop("metrics")
        metadata.update(evaluated=False, evaluation_failure="Invalid non-string predicted correction; cannot score")
    if export_error is not None:
        paths.pop("submission")
        paths.pop("zip")
        metadata.update(status="failed", failure=str(export_error))
        write_json(paths["evaluation"], metadata)
        raise export_error
    write_submission(output, submission)
    metadata["status"] = "completed"
    write_json(paths["evaluation"], metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    print(f"Submission ({input_scope}): {output}", flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_mutually_exclusive_group(required=True)
    stages.add_argument("--run", help="Completed two-stage saved run or unique enclosing extracted output")
    stages.add_argument("--classifier", help="Saved classifier checkpoint; requires --adapter")
    parser.add_argument("--adapter", help="Saved correction adapter, when using --classifier")
    parser.add_argument("--data", required=True, help="Raw organizer ID/Input JSON array, or auto to discover public_test.json")
    parser.add_argument("--output", help="Prediction file named results.json; defaults to a fresh runs/submission-*/results.json")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--evaluate", action="store_true", help="Require gold labels and write local evaluation metrics")
    parser.add_argument("--hint-mode", choices=["predicted", "none"], help="Override saved hint conditioning")
    parser.add_argument("--rerank", action="store_true", help="Score beam candidates and unchanged source")
    args = parser.parse_args()
    if args.run and args.adapter:
        parser.error("--adapter cannot be used with --run")
    if args.classifier and not args.adapter:
        parser.error("--classifier requires --adapter; Task1 alone cannot produce a full GEC submission")
    if args.run:
        from .run_reuse import resolve_run
        classifier = resolve_run(args.run, "classification") / "classifier"
        adapter = resolve_run(args.run, "correction") / "adapter"
    else:
        classifier, adapter = Path(args.classifier), Path(args.adapter)
    data = resolve_submission_data(args.data)
    if data is None:
        parser.error("No public_test.json found; attach it or pass --data with its path")
    records = load_records(data, require_labels=args.evaluate)
    output = (Path(args.output) if args.output else
              resolve_output({"run": {"name": "submission", "output_root": "runs"}}) / "results.json")
    run_inference(classifier, adapter, records, output, device=args.device,
                  batch_size=args.batch_size, evaluate=args.evaluate, input_path=data,
                  hint_mode=args.hint_mode, rerank=args.rerank)


if __name__ == "__main__":
    main()
