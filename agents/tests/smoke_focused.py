"""Actual four tiny fits/reloads, private ZIP binding, decoding and selected export.

Artificial English fixture strings only; this proves runtime contracts, not ML quality.
Run from the repo: python -m agents.tests.smoke_focused --output <new-directory>.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

from agents.tests.smoke_pipeline import fixture
from model.runtime import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--plan", type=Path, default=Path("configs/focused_plan.json"))
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    native_recipe = json.loads(Path(plan["training_recipe"]).read_text())
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    data = fixture(root, "cpu", False)
    original = json.loads((data / "VGEC 2026_v1.json").read_text())
    for r in original:
        r["ID"] += 1000
        if r["Output"]["Error_Type_Id"] == 0:
            r["Output"]["Fixed Sentence"] = r["Input"]
    original[11]["Output"]["Fixed Sentence"] = "fixed2 item1 ..."
    write_json(data / "VGEC 2026_v1.json", original)
    write_json(
        data / "private_test.json",
        [
            {"ID": 1, "Input": "unseen private fixture alpha ."},
            {"ID": "1", "Input": "unseen private fixture beta ."},
        ],
    )
    from model.retained_balanced import seal as retained, FORMAT_V2
    from model.grouped_validation import seal as grouped, derive, select_validation
    from model.quality_data import seal as quality

    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    packet = {
        s: retained(
            s,
            original,
            original,
            original,
            original,
            [],
            {"original_file_sha256": sha(data / "VGEC 2026_v1.json")},
            FORMAT_V2,
        )
        for s in ["classification", "correction"]
    }
    ids, _ = select_validation(packet)
    training, validation, *_ = derive(packet, ids)
    for stage, name in [
        ("classification", "VGEC_classifier.json"),
        ("correction", "VGEC_balanced.json"),
    ]:
        write_json(
            data / name,
            quality(grouped(stage, packet, ids, training[stage], validation, {})),
        )
    for stage in ["classification", "correction"]:
        p = root / (stage + ".json")
        c = json.loads(p.read_text())
        native = json.loads(Path(native_recipe[stage + "_config"]).read_text())
        c["competition_id"] = native.get("competition_id", 1038)
        c.pop("ensemble", None)
        c["training"]["args"].update(
            num_train_epochs=5, max_steps=-1, learning_rate=1e-5
        )
        c["oof"] = {"enabled": False} if stage == "classification" else c.get("oof", {})
        if stage == "correction":
            c["inference"].update(hint_mode="none", output_format="sentence")
        write_json(p, c)
    recipe = json.loads((root / "experiment.json").read_text())
    recipe.update(
        name="tiny-baseline",
        classification_config=str(root / "classification.json"),
        correction_config=str(root / "correction.json"),
        submission_filename="private_test.json",
    )
    recipe["data_sha256"] = {
        role: sha(data / name)
        for role, name in [
            ("original", "VGEC 2026_v1.json"),
            ("classification", "VGEC_classifier.json"),
            ("correction", "VGEC_balanced.json"),
        ]
    }
    recipe["submission_sha256"] = sha(data / "private_test.json")
    write_json(root / "baseline_recipe.json", recipe)
    from tools.run_experiment import prepare_run

    work, command = prepare_run(
        root / "baseline_recipe.json",
        data,
        root / "baseline_runs",
        world_size=1,
        device="cpu",
    )
    subprocess.run(command + ["--defer-submission"], check=True)
    from tools.package_focused_training import package

    manifest = package(
        data, work / "run/classification", work / "run/correction", root / "package"
    )
    # Same new representation and training-format branch as native recipe.
    c = json.loads((root / "correction.json").read_text())
    c["training"]["reference_format_conditioning"] = "separate_abbreviated"
    write_json(root / "correction.json", c)
    recipe.update(
        name="tiny-anchored",
        data_archive=manifest["archive"],
        data_sha256=manifest["data_sha256"],
    )
    write_json(root / "training_recipe.json", recipe)
    write_json(
        root / "proof.json",
        {"manifest_sha256": sha(root / "package/focused_package_manifest.json")},
    )
    from tools.run_focused import run

    plan.update(
        name="tiny-focused",
        training_recipe=str(root / "training_recipe.json"),
        package_proof=str(root / "proof.json"),
        screen_per_class=1,
        max_stage_seconds=3600,
        max_wall_seconds=7200,
    )
    out = run(
        plan,
        root / "package",
        root / "focused_runs",
        device="cpu",
        source_identity={"commit": "tiny-runtime-fixture", "dirty": True},
    )
    assert (
        json.loads((out / "pipeline_manifest.json").read_text())["status"]
        == "completed"
    )
    with zipfile.ZipFile(out / "results.zip") as z:
        assert z.namelist() == ["results.json"] and z.testzip() is None
        rows = json.loads(z.read("results.json"))
        assert [type(x["ID"]) for x in rows] == [int, str]
    summary = json.loads((out / "tuning_summary.json").read_text())
    for base in [work / "run", Path(summary["training_run"])]:
        for stage in ["classification", "correction"]:
            m = json.loads((base / stage / "run_manifest.json").read_text())
            assert m["config"]["training"]["args"]["num_train_epochs"] == 5
            assert (
                m.get(
                    "saved_classifier_reload_verified",
                    m.get("saved_adapter_reload_verified"),
                )
                is True
            )
            history = json.loads((base / stage / "epoch_history.json").read_text())
            assert len(history) == 5 and history[-1]["epoch"] == 5.0
    print(
        "PASS: four actual five-epoch tiny fits/reloads, packaged baseline binding, bounded decoding, typed-ID private export:",
        out,
    )


if __name__ == "__main__":
    main()
