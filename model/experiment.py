"""Run paired unadapted/QLoRA experiments across group-safe splits."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

from .train import write_json
from .config import PROJECT_ROOT, load_config, resolve_output


def comparison_row(metrics):
    return {"macro_f1": metrics["classification"]["macro_f1"],
            "error_gleu_plus": metrics["corrections"]["error_only"]["gleu_plus"],
            "error_exact_match": metrics["corrections"]["error_only"]["exact_match"],
            "error_rouge_l": metrics["corrections"]["error_only"]["rouge_l"],
            "clean_preservation": metrics["clean_preservation"]["exact_source_preservation"],
            "invalid_json_rate": metrics["invalid_json_rate"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data")
    parser.add_argument("--output-dir")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--device", choices=["cuda", "cpu"])
    args = parser.parse_args()
    config = load_config(args.config)
    seeds = args.seeds or config["run"]["seeds"]
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("Specify distinct, nonempty split seeds")
    output = resolve_output(config, args.output_dir)
    data = Path(args.data or config["data"]["path"] or PROJECT_ROOT / "dataset" / config["data"]["filename"]).resolve()
    device = args.device or config["run"]["device"]
    config["run"]["seeds"] = seeds
    config["run"]["device"] = device
    config["data"]["path"] = str(data)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    results = []
    for seed in seeds:
        split_dir = output / f"seed-{seed}"
        subprocess.run([sys.executable, "-m", "model.train", "--config", str(Path(args.config).resolve()),
                        "--data", str(data), "--output-dir", str(split_dir),
                        "--seed", str(seed), "--device", device], check=True)
        baseline = comparison_row(json.loads((split_dir / "baseline_metrics.json").read_text()))
        adapted = comparison_row(json.loads((split_dir / "metrics.json").read_text()))
        manifest = json.loads((split_dir / "run_manifest.json").read_text())
        results.append({"seed": seed, "best_epoch": manifest["best_epoch"],
                        "unadapted": baseline, "qlora": adapted,
                        "delta": {key: adapted[key] - baseline[key] if baseline[key] is not None and adapted[key] is not None else None for key in baseline}})
        report = {"model": config["model"]["name"], "revision": config["model"]["revision"],
                  "completed_seeds": [row["seed"] for row in results], "requested_seeds": seeds,
                  "selection_rule": manifest["selection_rule"], "splits": results,
                  "caveat": "Overlapping repeated holdouts, not independent test sets; each best epoch is selected on its own validation split."}
        report["aggregate"] = {}
        for stage in ("unadapted", "qlora", "delta"):
            report["aggregate"][stage] = {}
            for key in baseline:
                values = [row[stage][key] for row in results if row[stage][key] is not None]
                report["aggregate"][stage][key] = {"mean": statistics.mean(values) if values else None,
                    "sample_std": statistics.stdev(values) if len(values) > 1 else None,
                    "min": min(values) if values else None, "max": max(values) if values else None}
        write_json(output / "comparison.json", report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
    print(f"Completed run: {output}", flush=True)
    for seed in seeds:
        print(f"Saved adapter: {output / f'seed-{seed}' / 'adapter'}", flush=True)


if __name__ == "__main__":
    main()
