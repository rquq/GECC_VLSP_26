"""Reload a relocated focused selection with its exact greedy/beam/reranking strategy."""

import argparse
import json
import os
from pathlib import Path
import sys

from model.runtime import write_json
from tools.run_focused import resolve_selected, merge_export


def replay(selection, data, output, device="cuda"):
    selected, paths = resolve_selected(selection)
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    from model.data import load_records
    from model.reproducibility import process_environment
    from model.two_phase import _run_stage
    from model.tracking import TrackingRun

    config = json.loads((paths["classifier_path"].parent / "config.json").read_text())
    if not os.environ.get("WANDB_API_KEY"):
        config["tracking"]["mode"] = "offline"
    environment = {
        **process_environment(42),
        "VGEC_CUDA_STAGE": "1" if device == "cuda" else "0",
        "VGEC_PIPELINE_TRACKING": "1",
    }
    with TrackingRun() as tracking:
        tracking.start(
            config, output, "pipeline", metadata={"kind": "portable selected inference"}
        )

        def stage(command, name):
            _run_stage(
                command,
                {**environment, "VGEC_STAGE_NAME": name},
                tracking,
                timeout_seconds=18000,
            )

        stage(
            [
                sys.executable,
                "-m",
                "tools.focused_worker",
                "--classifier",
                str(paths["classifier_path"]),
                "--records",
                str(Path(data).resolve()),
                "--output",
                str(output / "classification.json"),
                "--device",
                device,
            ],
            "selected-private-classifier",
        )
        write_json(output / "strategy.json", selected["strategy"])
        stage(
            [
                sys.executable,
                "-m",
                "tools.decoding_sweep",
                "--adapter",
                str(paths["adapter"]),
                "--classifier-run",
                str(paths["classifier_path"].parent),
                "--output-dir",
                str(output / "decoding"),
                "--device",
                device,
                "--public-data",
                str(Path(data).resolve()),
                "--strategy",
                str(output / "strategy.json"),
            ],
            "selected-private-corrector",
        )
        records = load_records(data, require_labels=False)
        merge_export(
            records,
            json.loads((output / "classification.json").read_text()),
            json.loads((output / "decoding/predictions.json").read_text()),
            output,
        )
        write_json(
            output / "results.evaluation.json",
            {
                "evaluated": False,
                "count": len(records),
                "scope": "portable inference; no test labels used",
            },
        )
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["selection", "data", "output"]:
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    a = p.parse_args()
    replay(a.selection, a.data, a.output, a.device)


if __name__ == "__main__":
    main()
