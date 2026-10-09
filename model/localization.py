"""Train-only source ERROR / lexical-gap INSERT supervision (eta = 1).

Offsets are Python Unicode code-point offsets into unmodified source text.
Ambiguous optimal alignments and incomplete annotations never supply negatives.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import re
import unicodedata

import torch
from torch import nn

IGNORE = -100


def algorithm():
    return {
        "name": "hierarchical_source_edit_localization", "version": 1, "eta": 1.0,
        "alignment": "unique unit-cost lexical Levenshtein; Unicode letters/marks/numbers grouped; punctuation separate",
        "loss": "mean valid ERROR BCE per row + mean valid INSERT BCE per row; sum / full accumulation-window rows",
        "head": "train-only final-backbone-state ERROR linear and adjacent-state INSERT linear with learned start/end boundaries",
        "artifact": "localization_head.pt (separate from classification PEFT adapter)",
        "causal_limitation": "ERROR at token sees only its prefix; interior INSERT uses right neighbor but cannot see later suffix; no bidirectional claim",
        "exclusions": "ellipsis; annotations/optional alternatives; incomplete nonidentity; label mismatch; whitespace-only edits; nonunique optimum; truncation; unmappable offsets",
    }


def lexical_units(text):
    """Preserve combining marks and punctuation, without Unicode normalization."""
    units = []
    start = None
    for i, character in enumerate(text):
        word = unicodedata.category(character)[0] in "LMN" or character == "_"
        if word:
            if start is None:
                start = i
        else:
            if start is not None:
                units.append((text[start:i], start, i))
                start = None
            if not character.isspace():
                units.append((character, i, i + 1))
    if start is not None:
        units.append((text[start:], start, len(text)))
    return units


def align_pair(source, target, label):
    """Return source spans/gap boundaries only for a uniquely audited edit path."""
    result = {"eligible": False, "reason": None, "error_spans": [], "insert_gaps": []}
    def reject(reason):
        return {**result, "reason": reason}
    if not isinstance(target, str) or not source.strip() or not target.strip():
        return reject("empty_or_missing_pair")
    if "..." in source + target or "…" in source + target:
        return reject("ellipsis")
    if (target.lstrip().startswith(("(", "["))
            or re.match(r"(?i)\s*(?:sửa|câu này|câu đúng|chỉnh sửa|đáp án|ghi chú)\b", target)
            or any(c in source + target for c in "\n\r/|")):
        return reject("annotation_or_alternative")
    if source != target and any(c in source + target for c in "()[]"):
        return reject("optional_or_parenthetical_pair")
    if (label == 0) != (source == target):
        return reject("label_pair_mismatch")
    if source != target and any(text.rstrip().rstrip("\"'”’").rstrip()[-1:] not in (".", "!", "?") for text in (source, target)):
        return reject("incomplete_sentence")
    a, b = lexical_units(source), lexical_units(target)
    if not a or not b:
        return reject("empty_lexical_pair")
    if source == target:
        return {**result, "eligible": True, "reason": "clean_identity", "units": a}
    if [x[0] for x in a] == [x[0] for x in b]:
        return reject("whitespace_only_edit")
    n, m = len(a), len(b)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    ways = [[0] * (m + 1) for _ in range(n + 1)]
    ways[0][0] = 1
    for i in range(n + 1):
        for j in range(m + 1):
            if i == j == 0:
                continue
            options = []
            if i and j:
                options.append((cost[i-1][j-1] + (a[i-1][0] != b[j-1][0]), ways[i-1][j-1]))
            if i:
                options.append((cost[i-1][j] + 1, ways[i-1][j]))
            if j:
                options.append((cost[i][j-1] + 1, ways[i][j-1]))
            best = min(c for c, _ in options)
            cost[i][j] = best
            ways[i][j] = min(2, sum(w for c, w in options if c == best))
    if ways[n][m] != 1:
        return reject("ambiguous_alignment")
    errors, gaps = [], []
    i, j = n, m
    while i or j:
        if i and j and cost[i][j] == cost[i-1][j-1] + (a[i-1][0] != b[j-1][0]):
            if a[i-1][0] != b[j-1][0]:
                errors.append(a[i-1][1:])
            i, j = i-1, j-1
        elif i and cost[i][j] == cost[i-1][j] + 1:
            errors.append(a[i-1][1:])
            i -= 1
        else:
            gaps.append(i)
            j -= 1
    return {**result, "eligible": True, "reason": "unique_complete_edit", "units": a,
            "error_spans": sorted(errors), "insert_gaps": sorted(set(gaps))}


def encode_localization(tokenizer, records, max_length):
    """Tokenize ONLY sources. Reject entire truncated pairs, including hidden edits."""
    if not tokenizer.is_fast:
        raise ValueError("Localization supervision requires a fast offset-mapping tokenizer")
    tokens = tokenizer([r["source"] for r in records], truncation=True, max_length=max_length,
                       padding=False, return_offsets_mapping=True)
    full = tokenizer([r["source"] for r in records], truncation=False, padding=False)
    encoded, rows = [], []
    for index, record in enumerate(records):
        feature = {key: values[index] for key, values in tokens.items() if key != "offset_mapping"}
        offsets = [tuple(x) for x in tokens["offset_mapping"][index]]
        alignment = align_pair(record["source"], record["target"], record["label"])
        length = len(feature["input_ids"])
        error, insert = [IGNORE] * length, [IGNORE] * (length + 1)
        reason = alignment["reason"]
        eligible = alignment["eligible"]
        if eligible and feature["input_ids"] != full["input_ids"][index]:
            eligible, reason = False, "truncated_pair"
        if eligible:
            units = alignment["units"]
            mapping = []
            for _, start, end in units:
                indices = [k for k, (left, right) in enumerate(offsets) if right > left and left < end and right > start]
                # Byte-level tokenizers can repeat character spans; that is fine,
                # but no lexical unit may disappear or leak across punctuation.
                if not indices or min(offsets[k][0] for k in indices) > start or max(offsets[k][1] for k in indices) < end:
                    eligible, reason = False, "unmapped_source_unit"
                    break
                mapping.append(indices)
            gap_indices = []
            if eligible:
                for gap in range(len(units) + 1):
                    left = mapping[gap-1][-1] if gap else None
                    right = mapping[gap][0] if gap < len(units) else None
                    if left is not None and right is not None and left >= right:
                        eligible, reason = False, "token_crosses_lexical_gap"
                        break
                    gap_indices.append(right if right is not None else left + 1)
            if eligible:
                for indices in mapping:
                    for k in indices:
                        error[k] = 0
                for start, end in alignment["error_spans"]:
                    for k, (left, right) in enumerate(offsets):
                        if right > left and left < end and right > start:
                            error[k] = 1
                for k in gap_indices:
                    insert[k] = 0
                for gap in alignment["insert_gaps"]:
                    insert[gap_indices[gap]] = 1
        if not eligible:
            error, insert = [IGNORE] * length, [IGNORE] * (length + 1)
        feature.update(labels=record["label"], error_labels=error, insert_labels=insert)
        encoded.append(feature)
        rows.append({"id": record["id"], "class": record["label"], "source": record["source"],
                     "target": record["target"], "eligible": eligible, "reason": reason,
                     "pair_sha256": hashlib.sha256((record["source"] + "\0" + record["target"]).encode()).hexdigest(),
                     "offsets": offsets, "error_spans": alignment["error_spans"],
                     "insert_lexical_gaps": alignment["insert_gaps"],
                     "error_labels": error, "insert_labels": insert})
    summary = {"rows": len(rows), "eligible_rows": sum(r["eligible"] for r in rows),
               "exclusion_counts": dict(Counter(r["reason"] for r in rows if not r["eligible"])),
               "eligible_by_class": dict(Counter(str(r["class"]) for r in rows if r["eligible"]))}
    for kind in ("error", "insert"):
        summary[kind + "_counts"] = dict(Counter(str(label) for r in rows for label in r[kind + "_labels"]))
        summary[kind + "_counts_by_class"] = {
            str(label): dict(Counter(str(value) for r in rows if r["class"] == label for value in r[kind + "_labels"]))
            for label in range(5)}
    return encoded, {"algorithm": algorithm(), "max_length": max_length, "summary": summary, "rows": rows}


class LocalizationCollator:
    def __init__(self, tokenizer):
        from transformers import DataCollatorWithPadding
        self.collator = DataCollatorWithPadding(tokenizer)

    def __call__(self, features):
        batch = self.collator([{k: v for k, v in row.items() if k not in ("error_labels", "insert_labels")} for row in features])
        length = batch["input_ids"].shape[1]
        for kind, size in (("error", length), ("insert", length + 1)):
            batch[kind + "_labels"] = torch.tensor([row[kind + "_labels"] + [IGNORE] * (size - len(row[kind + "_labels"])) for row in features])
        return batch


class LocalizationHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.error = nn.Linear(hidden_size, 1)
        self.insert = nn.Linear(2 * hidden_size, 1)
        self.start = nn.Parameter(torch.zeros(hidden_size))
        self.end = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, hidden, attention_mask, valid_gaps=None):
        # The real end boundary replaces padding, not the batch's longest row.
        hidden = hidden.to(self.error.weight.dtype)
        batch, length, _ = hidden.shape
        left = torch.cat((self.start.expand(batch, 1, -1), hidden), dim=1)
        right = torch.cat((hidden, self.end.expand(batch, 1, -1)), dim=1)
        positions = torch.arange(length + 1, device=hidden.device)[None, :]
        if valid_gaps is None:
            starts = torch.zeros((batch, 1), dtype=torch.long, device=hidden.device)
            ends = attention_mask.sum(-1, keepdim=True)
        else:
            starts = torch.where(valid_gaps, positions, length + 1).amin(-1, keepdim=True)
            ends = torch.where(valid_gaps, positions, -1).amax(-1, keepdim=True)
        left = torch.where((positions == starts).unsqueeze(-1), self.start, left)
        right = torch.where((positions == ends).unsqueeze(-1), self.end, right)
        return self.error(hidden).squeeze(-1), self.insert(torch.cat((left, right), dim=-1)).squeeze(-1)


def localization_loss(error_logits, insert_logits, error_labels, insert_labels, num_items_in_batch=None):
    """No class weighting, no eligible-only denominator, exactly zero ignored gradient."""
    total = error_logits.sum() * 0
    for logits, labels in ((error_logits, error_labels), (insert_logits, insert_labels)):
        valid = labels != IGNORE
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), labels.clamp_min(0).float(), reduction="none")
        total = total + ((loss * valid).sum(-1) / valid.sum(-1).clamp_min(1)).sum()
    denominator = error_labels.shape[0] if num_items_in_batch is None else num_items_in_batch
    return total / denominator


def save_head(head, path):
    torch.save({"algorithm": algorithm(), "hidden_size": head.hidden_size,
                "state_dict": {k: v.detach().cpu() for k, v in head.state_dict().items()}}, path)


def load_head(path, device="cpu"):
    artifact = torch.load(path, map_location=device, weights_only=True)
    if artifact["algorithm"] != algorithm():
        raise ValueError("Unsupported localization auxiliary artifact")
    head = LocalizationHead(artifact["hidden_size"]).to(device)
    head.load_state_dict(artifact["state_dict"])
    return head
