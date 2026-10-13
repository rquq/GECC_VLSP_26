#!/usr/bin/env python3
"""Derive a private deterministic contrast pool from original training rows only.

The raw grouped seed-42, fraction-0.2 split precedes every audit/transformation.
Validation text is used only as an exclusion boundary, never as rule evidence.
Complete organizer corrections and clean training sentences supply all wording.
No external generator or human adjudication is involved. Derived resources remain
private, noncommercial research-only; redistribution is prohibited.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.data import split_records
from tools.build_dataset import audit_record, id_key, normalized, read_input

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORIGINAL = ROOT / "dataset/original/VGEC 2026_v1.json"
DEFAULT_OUTPUT = ROOT / "dataset/newdataset/synthetic_pool.json"
REVIEW_STATUS = "deterministic_rule_checked_not_human_adjudicated"
ORIGIN = "corpus_derived_deterministic_augmentation"

# These nominal units are boundaries, not a product of invented vocabulary.
# A rule can move only a quantifier and a complete, already adjacent noun head.
NOMINAL_HEADS = (
    "nguồn cảm hứng", "thành phần", "kim loại", "giai đoạn", "hiện tượng",
    "phương tiện", "kĩ năng", "câu hỏi", "khối khí", "đặc trưng", "phát minh",
    "cống hiến", "cuộc cách mạng", "giá trị", "quốc gia", "nhà khảo cổ",
    "đối tượng", "điểm", "lớp vỏ", "hình thức", "tù binh", "nông dân", "thời kì",
    "vấn đề", "bộ phận", "công cụ", "thành tố", "đơn vị", "biểu đồ",
    "khoáng vật", "vi khuẩn", "nhà văn", "nguồn tài nguyên",
    "tác phẩm", "bộ lạc", "vật liệu",
)
QUANTIFIER = re.compile(
    r"\b(?P<quantifier>những|các) (?P<head>"
    + "|".join(re.escape(head) for head in sorted(NOMINAL_HEADS, key=lambda s: (-len(s), s)))
    + r")\b"
)
COPULAR = re.compile(r"^(?P<subject>[^,;:!?()\"“”]+) là (?P<predicate>[^,;:!?()\"“”]+)\.$")
NOMINAL_SUBJECT = re.compile(
    r"^(?:Quy luật|Cảm hứng|Ngôn từ|Truyện Kiều|Nô lệ|Nguồn năng lượng|Sóng biển|"
    r"Chữ viết|Cách mạng|Nội lực|Thức ăn|Thời gian|Đỉnh cao|Nguyên nhân|"
    r"Sự (?:ra đời|xuất hiện)|Mâu thuẫn|Thành phần|Phương pháp|Thao tác)\b"
)


def structural_candidates(target: str):
    """Yield single-span corruptions without lexical/semantic substitutions."""
    match = COPULAR.fullmatch(target)
    if (match and NOMINAL_SUBJECT.match(match["subject"])
            and not re.search(r"\b(?:đã|đang|sẽ|được|gọi|có|ảnh hưởng|tạo nên)\b", match["subject"])):
        yield 1, "missing_main_copular_predicate", match["subject"] + ".", {
            "start": len(match["subject"]), "clean_span": " là " + match["predicate"],
            "error_span": "", "basis": "Conservative nominal fragment: delete the entire sole main copular predicate; not subject ellipsis or optional copula deletion.",
        }
    for match in QUANTIFIER.finditer(target):
        replacement = match["head"] + " " + match["quantifier"]
        yield 2, "quantifier_after_nominal_head", target[:match.start()] + replacement + target[match.end():], {
            "start": match.start(), "clean_span": match[0], "error_span": replacement,
            "basis": "Move an overt plural marker after its existing complete nominal head; exclude numeral postnominal numbering, year numerals, idioms, and adjective/semantic substitutions.",
        }
    # Only unambiguous copular/aspect function words, not lexical repetitions.
    for match in re.finditer(r"\b(?:là|đã|đang|sẽ)\b", target):
        replacement = match[0] + " " + match[0]
        yield 3, "adjacent_copula_or_aspect_repetition", target[:match.start()] + replacement + target[match.end():], {
            "start": match.start(), "clean_span": match[0], "error_span": replacement,
            "basis": "Adjacent identical copula/aspect markers are incompatible in ordinary edited prose; keep all semantic words, numbers and grammatical fronts.",
        }
    if target.endswith("."):
        yield 4, "comma_after_terminal_period", target + ",", {
            "start": len(target), "clean_span": "", "error_span": ",",
            "basis": "A dangling comma after the existing terminal full stop is mechanical punctuation corruption, not a question/declarative conversion.",
        }


def original_hashes(original: Path) -> dict[str, str]:
    """Snapshot every regular organizer file beside the input, including XLSX."""
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(original.parent.iterdir()) if path.is_file()}


def make_row(label: int, parent: dict, source: str, target: str, rule: str,
             evidence: dict, original_sha: str) -> dict:
    parent_key = hashlib.sha256(json.dumps([type(parent["id"]).__name__, parent["id"]], ensure_ascii=False).encode()).hexdigest()[:20]
    target_key = hashlib.sha256(normalized(target).encode()).hexdigest()[:20]
    row_key = hashlib.sha256(json.dumps([label, source, target, rule], ensure_ascii=False).encode()).hexdigest()[:24]
    return {
        "ID": f"corpus:{label}:{parent_key}:{row_key}", "Input": source,
        "Output": {"Error_Type_Id": label, "Fixed Sentence": target},
        "Provenance": {
            "parent_ids": [parent["id"]], "template_rule": rule, "rule": rule,
            # Parent-level templates avoid joining the entire corpus into one OOF
            # component; target families still connect all exact counterparts.
            "template_id": f"corpus_parent_{parent_key}", "family_id": f"corpus_target_{target_key}",
            "origin": ORIGIN, "authoring_method": "deterministic_training_partition_minimal_contrast",
            "review_status": REVIEW_STATUS,
            "review_limit": "Automated original audit and conservative rules only; not human adjudication or evidence of model benefit. Quoted replays retain organizer labels without claiming their linguistic correctness.",
            "uses_source_corpus": True, "uses_external_teacher": False, "training_only": True,
            "original_sha256": original_sha, "split_seed": 42, "validation_fraction": 0.2,
            "evidence": evidence, "private_dataset": True, "publication_allowed": False,
            "terms": "VLSP 2026 private noncommercial research resource; modifications allowed, redistribution prohibited; acknowledge VLSP 2026 and the VGEC organizers.",
        },
    }


def author_pool(original: Path = DEFAULT_ORIGINAL) -> list[dict]:
    hashes_before = original_hashes(original)
    content, raw, records = read_input(original)
    original_sha = hashlib.sha256(content).hexdigest()
    train, validation = split_records(records, 0.2, 42)
    raw_by_id = {id_key(row["ID"]): row for row in raw}
    heldout = {normalized(row[field]) for row in validation for field in ("source", "target")}
    original_sources = {normalized(row["source"]) for row in train}
    candidates = []
    for parent in sorted(train, key=lambda row: (type(row["id"]).__name__, str(row["id"]))):
        audit = audit_record(parent, raw_by_id[id_key(parent["id"])], original=True)
        if audit["quarantine_reasons"] or any(normalized(parent[field]) in heldout for field in ("source", "target")):
            continue
        target = parent["target"]
        if not target or target != target.strip() or "\n" in target or "\r" in target:
            continue
        # Complete corrected targets supply novel exact-identity clean examples.
        if parent["label"]:
            candidates.append(make_row(0, parent, target, target, "complete_training_gold_identity", {
                "basis": "Exact complete original training correction; never infer clean labels from annotation-like or shortened gold.",
            }, original_sha))
        for label, rule, source, evidence in structural_candidates(target):
            candidates.append(make_row(label, parent, source, target, rule, evidence, original_sha))
        if parent["label"] == 1:
            # Replay the exact observed error/gold edit inside one balanced quoted
            # utterance. No clause reconstruction or cross-sentence word product.
            # Three ordinary quote conventions provide enough unique capped rows.
            for name, opener, closer in (("curly", "“", "”"), ("ascii", '"', '"'), ("single", "‘", "’")):
                if opener in parent["source"] + target or closer in parent["source"] + target:
                    continue
                source, quoted_target = opener + parent["source"] + closer, opener + target + closer
                candidates.append(make_row(1, parent, source, quoted_target, f"quoted_original_clause_edit_{name}", {
                    "basis": "Exact organizer class-1 edit, in the same lexical and grammatical context, with identical balanced quotation framing on both sides.",
                    "organizer_label": 1, "quote_open": opener, "quote_close": closer,
                }, original_sha))
                candidates.append(make_row(0, parent, quoted_target, quoted_target, f"quoted_complete_training_gold_identity_{name}", {
                    "basis": "Balanced quotation of a complete original training correction, exact identity target.",
                    "quote_open": opener, "quote_close": closer,
                }, original_sha))
    annotations = defaultdict(set)
    for row in candidates:
        annotations[normalized(row["Input"])].add((row["Output"]["Error_Type_Id"], normalized(row["Output"]["Fixed Sentence"])))
    rows, seen = [], set()
    for row in candidates:
        source, target = row["Input"], row["Output"]["Fixed Sentence"]
        key = normalized(source)
        if (key in original_sources or key in heldout or normalized(target) in heldout
                or len(annotations[key]) != 1 or key in seen):
            continue
        record = {"id": row["ID"], "source": source, "target": target, "label": row["Output"]["Error_Type_Id"]}
        if audit_record(record, row)["quarantine_reasons"]:
            continue
        rows.append(row)
        seen.add(key)
    if original_hashes(original) != hashes_before:
        raise ValueError("Organizer files changed during augmentation; no pool written")
    return rows


def private_directory(path: Path) -> None:
    """Create every missing ancestor privately, then protect the destination."""
    missing = []
    ancestor = path
    while not ancestor.exists():
        missing.append(ancestor)
        ancestor = ancestor.parent
    for directory in reversed(missing):
        directory.mkdir(mode=0o700)
    if path.is_symlink() or not path.is_dir():
        raise ValueError("Refusing symlink/non-directory destination")
    os.chmod(path, 0o700)


def write_private_pool(output: Path, rows: list[dict], original: Path = DEFAULT_ORIGINAL) -> None:
    if output.name != "synthetic_pool.json":
        raise ValueError("Output must be named synthetic_pool.json")
    if output.is_symlink() or (output.exists() and (not output.is_file() or output.stat().st_nlink != 1)):
        raise ValueError("Refusing symlink, hard-linked or non-file output")
    for path in original.parent.iterdir():
        if output.resolve() == path.resolve() or (output.exists() and path.is_file() and os.path.samefile(output, path)):
            raise ValueError("Output would overwrite an organizer file")
    if output.exists():
        previous = json.loads(output.read_text(encoding="utf-8"))
        if not isinstance(previous, list) or not previous or any(
                not isinstance(row, dict) or not isinstance(row.get("Provenance"), dict)
                or row["Provenance"].get("origin") not in {"locally_authored_synthetic", ORIGIN}
                or row["Provenance"].get("uses_external_teacher") is not False for row in previous):
            raise ValueError("Refusing to overwrite an unrelated existing file")
    private_directory(output.parent)
    descriptor, temporary = tempfile.mkstemp(prefix=".synthetic_pool-", suffix=".json", dir=output.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    hashes_before = original_hashes(args.original)
    rows = author_pool(args.original)
    write_private_pool(args.output, rows, args.original)
    hashes_after = original_hashes(args.original)
    if hashes_before != hashes_after:
        raise ValueError("Organizer files changed while writing the derived pool")
    print(json.dumps({"pool_rows": len(rows), "class_counts": dict(sorted(Counter(row["Output"]["Error_Type_Id"] for row in rows).items())),
                      "output": str(args.output), "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                      "original_hashes_before": hashes_before, "original_hashes_after": hashes_after,
                      "originals_unchanged": True}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
