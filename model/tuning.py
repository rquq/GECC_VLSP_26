"""Bounded development-set comparisons; never treat tuning scores as unseen results."""

import math
import random
from .data import _id_key


def calibrated_vote(predictions, weights, temperature=1.0, phrase_scale=1.0):
    from .correction import _hint_index

    if (
        len(predictions) != 3
        or len(weights) != 3
        or any(not math.isfinite(w) or w < 0 for w in weights)
        or sum(weights) <= 0
        or not math.isfinite(temperature)
        or temperature <= 0
        or not math.isfinite(phrase_scale)
        or phrase_scale <= 0
    ):
        raise ValueError("Invalid soft-vote calibration")
    rows = [{"id": p["id"]} for p in predictions[0]]
    indexed = [_hint_index(rows, p) for p in predictions]
    result = []
    for row in rows:
        values = [0.0] * 5
        for index, weight in zip(indexed, weights):
            p = [
                v ** (1 / temperature)
                for v in index[_id_key(row["id"])]["probabilities"]
            ]
            total = sum(p)
            values = [a + weight * b / total / sum(weights) for a, b in zip(values, p)]
        values[2] *= phrase_scale
        total = sum(values)
        values = [v / total for v in values]
        result.append(
            {
                "id": row["id"],
                "label": max(range(5), key=values.__getitem__),
                "probabilities": values,
            }
        )
    return result


def decoding_strategies():
    def strategy(name, **kwargs):
        return {
            "name": name,
            "generation": {"do_sample": False, "repetition_penalty": 1.0, **kwargs},
        }

    return [
        strategy("greedy_reference", repetition_penalty=1.05),
        strategy("greedy_no_penalty"),
        strategy("beam2", num_beams=2),
        strategy("beam4", num_beams=4),
        strategy("beam8", num_beams=8),
        strategy("beam4_short", num_beams=4, length_penalty=0.6),
        strategy(
            "diverse_beam4", num_beams=4, num_beam_groups=2, diversity_penalty=0.3
        ),
        strategy(
            "beam_sampling4",
            num_beams=4,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
        ),
        strategy("contrastive", penalty_alpha=0.6, top_k=4, low_memory=True),
        {**strategy("schema_beam4", num_beams=4), "schema": True},
        {
            **strategy("beam4_edit_rerank", num_beams=4, num_return_sequences=4),
            "edit_rerank": True,
            "edit_penalty": 0.15,
        },
    ]


def panel(records, per_class=2, seed=42):
    rng = random.Random(seed)
    selected = set()
    for label in range(5):
        group = [r for r in records if r["label"] == label]
        rng.shuffle(group)
        selected.update(_id_key(r["id"]) for r in group[:per_class])
    return [r for r in records if _id_key(r["id"]) in selected]


def inference_rows(records):
    return [{"id": r["id"], "source": r["source"]} for r in records]


def classifier_search(root, output):
    import json
    from pathlib import Path
    from .runtime import write_json
    from .metrics import score

    root = Path(root)
    gold = json.loads((root / "validation.json").read_text())
    predictions = []
    for name in ["qwen7b", "sailor7b", "mistral7b"]:
        rows = json.loads((root / f"members/{name}/predictions.json").read_text())
        predictions.append(
            [
                {
                    "id": r["id"],
                    "label": r["predicted_class"],
                    "probabilities": r["probabilities"],
                }
                for r in rows
            ]
        )
    results = []
    for weights in [[1, 1, 1], [2, 1, 1], [1, 2, 1], [1, 1, 2]]:
        for temperature in [1.0, 0.8, 1.2]:
            for phrase_scale in [1.0, 1.25, 1.5]:
                predicted = calibrated_vote(
                    predictions, weights, temperature, phrase_scale
                )
                sources = {_id_key(r["id"]): r["source"] for r in gold}
                measured = score(
                    gold,
                    [{**p, "target": sources[_id_key(p["id"])]} for p in predicted],
                )["classification"]
                results.append(
                    {
                        "weights": weights,
                        "temperature": temperature,
                        "phrase_scale": phrase_scale,
                        "metrics": measured,
                    }
                )
    results.sort(key=lambda r: r["metrics"]["macro_f1"], reverse=True)
    report = {
        "scope": "post-hoc tuning on the same selected78-row development set; not independent validation",
        "trials": results,
        "best": results[0],
    }
    write_json(Path(output), report)
    return report
