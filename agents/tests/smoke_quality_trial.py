"""Two actual five-epoch CPU fits and four controlled private-fixture exports."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import zipfile

from model.runtime import write_json
from tools.package_focused_training import package, sha256
from tools.run_quality_trial import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    fixture, output = args.fixture.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    baseline, = list((fixture / "baseline_runs").glob("*/run"))
    adapter_hash = sha256(baseline / "correction/adapter/adapter_model.safetensors")
    # Artificial two-row inference only; the baseline was already fitted by smoke_focused.
    if not (baseline / "results.zip").exists():
        subprocess.run([sys.executable, "-m", "model.pipeline_predict", "--classifier", str(baseline / "classification/classifier"),
                        "--adapter", str(baseline / "correction/adapter"), "--data", str(fixture / "data/private_test.json"),
                        "--output", str(baseline / "results.json"), "--device", "cpu", "--batch-size", "1",
                        "--input-scope", "explicit_submission"], check=True)
    packet = package(fixture / "data", baseline / "classification", baseline / "correction", output / "package",
                     archive_name="tiny-quality-classifier.zip", control_submission=baseline / "results.zip")
    config = json.loads((fixture / "classification.json").read_text())
    config["competition_id"] = 1039
    write_json(output / "classification.json", config)
    correction = json.loads((fixture / "correction.json").read_text())
    correction["competition_id"] = 1039
    write_json(output / "correction.json", correction)
    recipe = json.loads((fixture / "baseline_recipe.json").read_text())
    recipe.update(name="tiny-quality-trial", mode="both", classification_config=str(output / "classification.json"),
                  correction_config=str(output / "correction.json"), data_archive=packet["archive"],
                  data_sha256=packet["data_sha256"], submission_sha256=packet["submission_sha256"])
    write_json(output / "recipe.json", recipe)
    write_json(output / "proof.json", {"manifest_sha256": sha256(output / "package/focused_package_manifest.json")})
    plan = {"format": "vgec-quality-trial-v1", "name": "tiny-quality-classifier", "training_recipe": str(output / "recipe.json"),
            "package_proof": str(output / "proof.json"), "inference_sha256": packet["submission_sha256"],
            "max_stage_seconds": 3600, "max_wall_seconds": 7200}
    result = run(plan, output / "package", output / "runs", device="cpu",
                 source_identity={"commit": "artificial-classifier-trial", "dirty": True})
    manifest = json.loads((result / "pipeline_manifest.json").read_text())
    assert manifest["status"] == "completed" and manifest["new_classifier_fits"] == 1 and manifest["new_corrector_fits"] == 1
    assert (result / "quality_control/results.zip").read_bytes() == (baseline / "results.zip").read_bytes()
    assert sha256(baseline / "correction/adapter/adapter_model.safetensors") == adapter_hash
    training = Path(json.loads((result.parent / "experiment_manifest.json").read_text())["training"])
    fit = json.loads((training / "run/classification/run_manifest.json").read_text())
    assert fit["config"]["training"]["args"]["num_train_epochs"] == 5 and fit["optimizer_steps"] == 20
    assert fit["saved_classifier_reload_verified"] is True
    correction_fit = json.loads((training / "run/correction/run_manifest.json").read_text())
    assert correction_fit["config"]["training"]["args"]["num_train_epochs"] == 5
    assert correction_fit["saved_adapter_reload_verified"] is True
    control = json.loads((result / "quality_control/results.json").read_text())
    candidates = {}
    for relative in ["results.zip", "quality_control/results.zip", "corrector_challenger/results.zip", "both_challengers/results.zip"]:
        with zipfile.ZipFile(result / relative) as archive:
            assert archive.testzip() is None and archive.namelist() == ["results.json"]
            rows = json.loads(archive.read("results.json"))
            assert [type(r["ID"]) for r in rows] == [int, str]
            candidates[relative] = rows
    labels = lambda name: [r["Output"]["Error_Type_Id"] for r in candidates[name]]
    texts = lambda name: [r["Output"]["Fixed Sentence"] for r in candidates[name]]
    assert texts("results.zip") == texts("quality_control/results.zip")
    assert texts("both_challengers/results.zip") == texts("corrector_challenger/results.zip")
    assert labels("results.zip") == labels("both_challengers/results.zip")
    assert labels("quality_control/results.zip") == labels("corrector_challenger/results.zip")
    print("PASS: two actual full-five-epoch fits/reloads, preserved baseline and four controlled ZIPs:", result)


if __name__ == "__main__":
    main()
