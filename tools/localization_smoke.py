"""Executable CPU-only tokenizer/audit smoke; does not load model weights.

Run: python -m tools.localization_smoke --output /absolute/path/localization-audit.json
Tiny Qwen gradient/Trainer/reload smoke: python -m unittest agents.tests.test_localization
"""
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from model.classification import _configure_padding, _encoded, _preprocessing
from model.data import load_classifier_split, load_records, split_records
from model.localization import encode_localization
from model.runtime import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="/home/quq/machineLearning/competitions/VLSP26_4/runs/hierarchical-localization-seed42-config.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    train, _ = load_classifier_split(config["data"]["path"], config["data"]["augmentation_path"],
                                     config["data"]["validation_fraction"], config["run"]["seed"])
    _, validation = split_records(load_records(config["data"]["path"]),
                                  config["data"]["validation_fraction"], config["run"]["seed"])
    settings = config["model"]
    # Preserve baseline tokenizer selection, using cached files and no GPU.
    tokenizer = AutoTokenizer.from_pretrained(settings["name"], revision=settings["revision"],
                                               use_fast=False, local_files_only=True)
    _configure_padding(tokenizer)
    encoded, audit = encode_localization(tokenizer, train, config["training"]["max_length"])
    baseline = _encoded(tokenizer, train, _preprocessing(config))
    for feature, original in zip(encoded, baseline):
        for key, value in original.items():
            if feature[key] != value:
                raise AssertionError(f"Localization changed baseline source inputs: {key}")
    if len(train) != 480 or len(validation) != 81 or not audit["summary"]["eligible_rows"]:
        raise AssertionError("Unexpected matched split or empty eligible supervision")
    write_json(Path(args.output), audit)
    print(json.dumps({"tokenizer_class": type(tokenizer).__name__, "is_fast": tokenizer.is_fast,
                      "source_inputs_identical": True, "train": len(train), "validation": len(validation),
                      "audit": str(Path(args.output).resolve()), "summary": audit["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
