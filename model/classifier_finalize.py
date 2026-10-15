"""Verify a trained classifier in a fresh process, without fitting any weights."""

import argparse
import json
import math
from pathlib import Path

from .runtime import write_json


def finalize(output, device):
    from .classification import classify, _classification_metrics
    from .prediction_export import comparison_rows

    output = Path(output)
    receipt = json.loads((output / "training_complete.json").read_text())
    train = json.loads((output / "train.json").read_text())
    validation = json.loads((output / "validation.json").read_text())
    final_fit = receipt["final_fit"]
    batch_size = receipt["config"]["training"]["args"].get(
        "per_device_eval_batch_size", 8
    )
    predictions = classify(
        output / "classifier",
        train[:1] if final_fit else validation,
        device,
        batch_size,
    )
    classification = (
        None if final_fit else _classification_metrics(validation, predictions)
    )
    if classification is not None and not math.isclose(
        classification["macro_f1"], receipt["selection_macro_f1"], abs_tol=1e-8
    ):
        raise ValueError(
            "Reloaded classifier score differs from the selected epoch; checkpoint not accepted"
        )
    metrics = {
        "count": len(validation),
        "evaluation_status": "unavailable_full_training" if final_fit else "evaluated",
        "classification": classification,
        "training": receipt,
    }
    write_json(
        output / "predictions.json",
        comparison_rows(validation, [] if final_fit else predictions),
    )
    write_json(output / "metrics.json", metrics)
    write_json(
        output / "run_manifest.json",
        {
            **receipt,
            "output_dir": str(output),
            "classifier_path": str(output / "classifier"),
            "saved_classifier_reloaded_for_evaluation": not final_fit,
            "saved_classifier_reload_verified": True,
        },
    )
    return metrics


def main():
    from .tracking import TrackingRun

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], required=True)
    args = parser.parse_args()
    root = Path(args.output_dir)
    receipt = json.loads((root / "training_complete.json").read_text())
    with TrackingRun() as tracking:
        tracking.start(
            receipt["config"],
            root / "verification",
            "classification-verification",
            group=receipt["config"].get("_tracking_group"),
        )
        metrics = finalize(root, args.device)
        tracking.complete("classifier", metrics, best_epoch=receipt["best_epoch"])


if __name__ == "__main__":
    main()
