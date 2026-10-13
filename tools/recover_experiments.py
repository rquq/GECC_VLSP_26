"""Recover experiment evidence from existing artifacts and archives; never train."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tarfile


def recover(workspace, destination):
    destination.mkdir(parents=True, exist_ok=True)
    private = destination.parent / "private/recovered-notes"
    private.mkdir(parents=True, exist_ok=True)
    records = []
    for checkout in ("test", "alpha", "kaggle-linux"):
        root = workspace / checkout
        for path in sorted((root / "runs").rglob("metrics.json")):
            if any(part.startswith(("smoke-", "epoch-")) for part in path.parts) or any(part in ("oof", "wandb", "trainer") for part in path.parts):
                continue
            metrics = json.loads(path.read_text())
            if "classification" not in metrics:
                continue
            manifest_path = path.parent / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
            config = manifest.get("config", {})
            settings = config.get("training", {}).get("args", {})
            records.append({"id": f"artifact:{checkout}:{path.parent.relative_to(root)}",
                            "status": "measured_artifact", "evidence": str(path),
                            "evidence_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "count": metrics.get("count"), "classification": metrics["classification"],
                            "model": config.get("model", manifest.get("model")),
                            "epochs": settings.get("num_train_epochs"), "learning_rate": settings.get("learning_rate"),
                            "best_epoch": manifest.get("best_epoch"), "seed": manifest.get("seed"),
                            "scope": "local holdout; comparability requires checking data/precision/training budgets"})
    archive_path = workspace / "maintenance-pre-wandb-agents.tar.gz"
    if archive_path.is_file():
        with tarfile.open(archive_path) as archive:
            for number, member in enumerate(archive.getmembers()):
                if not member.isfile() or not member.name.endswith(".md"):
                    continue
                content = archive.extractfile(member).read()
                path = private / f"archive-{number:03d}-{Path(member.name).name}"
                path.write_bytes(content)
                path.chmod(0o600)
                lines = content.decode("utf-8").splitlines()
                starts, code = [], False
                for i, line in enumerate(lines):
                    if line.lstrip().startswith("```"):
                        code = not code
                    elif not code and re.match(r"^#{1,6} ", line):
                        starts.append(i)
                for i, start in enumerate(starts):
                    end = starts[i + 1] if i + 1 < len(starts) else len(lines)
                    section = "\n".join(lines[start:end])
                    if not re.search(r"ablation|experiment|ensemble|verification|rerun|fit|baseline|improvement|corpus|overlay|comparison|encoder|classifier", lines[start], re.I):
                        continue
                    records.append({"id": f"archive:{number}:{start + 1}", "status": "recovered_documentation",
                                    "title": lines[start].lstrip("# "), "evidence": str(path),
                                    "lines": [start + 1, end], "archive_member": member.name,
                                    "evidence_sha256": hashlib.sha256(content).hexdigest(),
                                    "section_sha256": hashlib.sha256(section.encode()).hexdigest(),
                                    "scope": "historical reported evidence; not newly verified or rerun"})
    project = workspace / "kaggle-linux"
    history = subprocess.check_output(["git", "log", "--all", "--date=iso-strict", "--format=%H\t%ad\t%s"], cwd=project, text=True)
    for line in history.splitlines():
        commit, date, title = line.split("\t", 2)
        records.append({"id": f"commit:{commit}", "status": "code_history", "title": title,
                        "commit": commit, "date": date, "scope": "commit existence alone does not establish a training result"})
    for name in ("progress.json", "cleanup.json", "integrity.json", "baseline-runtime-verification.json"):
        path = workspace / "maintenance" / name
        if path.is_file():
            target = private / f"maintenance-{name}"
            target.write_bytes(path.read_bytes())
            target.chmod(0o600)
            records.append({"id": f"maintenance:{name}", "status": "recovered_maintenance_record",
                            "evidence": str(target), "evidence_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                            "scope": "cleanup/runtime record, not a new training result"})
    test_repo = workspace / "test"
    stashes = subprocess.check_output(["git", "stash", "list", "--format=%H"], cwd=test_repo, text=True).splitlines()
    for stash in stashes:
        # Include the optional untracked-file parent, where deleted draft recipes lived.
        for ref in (stash, stash + "^3"):
            listing = subprocess.run(["git", "ls-tree", "-r", "--name-only", ref], cwd=test_repo,
                                     capture_output=True, text=True)
            if listing.returncode:
                continue
            for name in listing.stdout.splitlines():
                if not name.startswith("configs/") or not name.endswith(".json"):
                    continue
                content = subprocess.check_output(["git", "show", f"{ref}:{name}"], cwd=test_repo)
                target = private / f"stash-{stash[:7]}-{ref.endswith('^3')}-{Path(name).name}"
                target.write_bytes(content)
                target.chmod(0o600)
                records.append({"id": f"stash:{ref}:{name}", "status": "stashed_recipe_only", "title": name,
                                "evidence": str(target), "evidence_sha256": hashlib.sha256(content).hexdigest(),
                                "scope": "recovered draft recipe; no completion or metric inferred"})
    probes = project / "runs/audit-kano-20261013/classifier-probes.json"
    if probes.is_file():
        values = json.loads(probes.read_text())
        for name in ("original_only", "current_augmentation"):
            records.append({"id": f"lexical-probe:{name}", "status": "measured_probe_not_promoted",
                            "evidence": str(probes), "evidence_sha256": hashlib.sha256(probes.read_bytes()).hexdigest(),
                            "metrics": values[name], "scope": "training-only grouped CV selected C; same 81-row holdout"})
    output = destination / "history.json"
    output.write_text(json.dumps({"schema": "vgec-experiment-history-v1", "training_rerun": False,
                                  "records": records}, ensure_ascii=False, indent=2) + "\n")
    text = ["# Recovered experiment history", "", "Recovery uses surviving run artifacts, archived agent notes, Git history, and maintenance records. No historical training was rerun. Private recovered notes can contain corpus text and remain ignored by Git.", "", "## Surviving measured artifacts", "", "| Run | Macro-F1 | Micro-F1 | Evidence |", "|---|---:|---:|---|"]
    for row in records:
        if row["status"] == "measured_artifact":
            scores = row["classification"]
            text.append(f"| {row['id'].removeprefix('artifact:')} | {scores.get('macro_f1', 0):.6f} | {scores.get('micro_f1', 0):.6f} | `{row['evidence']}` |")
    text += ["", "## Recovered notes", "", "These entries index historical claims; consult their private source lines before using a metric. Duplicate archive versions are retained explicitly.", ""]
    for row in records:
        if row["status"] == "recovered_documentation":
            text.append(f"- {row['title']} — `{row['evidence']}` lines {row['lines'][0]}–{row['lines'][1]}.")
    text += ["", "## Code and recipe chronology", "", "Implementation and recipe commits are evidence of code changes, not proof that their experiments completed.", ""]
    text += ["Recovered draft recipes from stashes (no training outcome inferred):", ""]
    for row in records:
        if row["status"] == "stashed_recipe_only":
            text.append(f"- `{row['id']}` → `{row['evidence']}`")
    text.append("")
    for row in reversed(records):
        if row["status"] == "code_history":
            text.append(f"- {row['date'][:10]} `{row['commit'][:7]}`: {row['title']}")
    (destination / "HISTORY.md").write_text("\n".join(text) + "\n")
    return {"records": len(records), "measured_artifacts": sum(row["status"] == "measured_artifact" for row in records),
            "archived_sections": sum(row["status"] == "recovered_documentation" for row in records), "training_rerun": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    project = Path(__file__).resolve().parents[1]
    parser.add_argument("--workspace", type=Path, default=project.parent)
    parser.add_argument("--output", type=Path, default=project / "agents/experiments")
    args = parser.parse_args()
    print(json.dumps(recover(args.workspace, args.output), indent=2))


if __name__ == "__main__":
    main()
