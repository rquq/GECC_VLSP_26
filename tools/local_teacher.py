"""Bounded private eligible-teacher pilot; never reads holdout or test prompts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time

MODEL = "Qwen/Qwen2.5-7B-Instruct"
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
PARAMETERS = 7615616512
MECHANISMS = {
    1: ["missing finite predicate", "missing required subject", "unattached subordinate clause", "incomplete parallel clauses", "missing main clause", "broken subject predicate construction"],
    2: ["noun phrase modifier order", "numeral classifier noun order", "verb complement phrase structure", "adjective phrase complement order", "prepositional phrase constituent order", "nested noun phrase constituent order"],
    3: ["missing required preposition", "wrong preposition", "redundant copula", "duplicated aspect particle", "missing correlative function word", "wrong correlative function word"],
    4: ["missing enumeration comma", "missing introduction colon", "unmatched quotation mark", "incorrect sentence boundary", "wrong clause separator", "missing parenthetical comma"],
}
TAXONOMY = "0 câu đúng; 1 lỗi cấu trúc câu (thiếu chủ ngữ/vị ngữ/mệnh đề chính); 2 lỗi cấu trúc cụm từ (trật tự thành tố trong cụm danh từ/động từ/tính từ); 3 lỗi hư từ (giới từ, liên từ, trợ từ); 4 lỗi dấu câu. A punctuation-only error is 4, never 1. Vietnamese verbs do NOT require a tense particle. Spelling, style, vocabulary and semantic changes are outside this taxonomy."


def parse_json(text):
    decoder = json.JSONDecoder()
    for offset, character in enumerate(text):
        if character == "{":
            try:
                value, _ = decoder.raw_decode(text[offset:])
                if isinstance(value, dict):
                    return value
            except ValueError:
                pass
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--repairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Continue remaining keys within the original call/time budget")
    args = parser.parse_args()
    from model.data import load_records, split_records
    from tools.experiment_log import record
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    args.output.mkdir(parents=True, exist_ok=True)
    journal = args.output / "calls.jsonl"
    if journal.exists() and not args.resume:
        raise ValueError("Pilot output already exists; no implicit repeat of generation")
    previous = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
    train, _ = split_records(load_records(args.original), .2, 42)
    original_ids = {r["id"] for r in train}
    repairs = json.loads(args.repairs.read_text())["candidate_changes"]
    if any(r["id"] not in original_ids for r in repairs):
        raise ValueError("Repair parent outside training split")
    set_seed(42)
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    meta = dict(model=MODEL, revision=REVISION, parameters=PARAMETERS, source_commit=source_commit,
                original_sha256=hashlib.sha256(args.original.read_bytes()).hexdigest(), seed=42,
                max_calls=144, max_gpu_minutes=90, max_input_tokens=768, max_new_tokens=384)
    if not args.resume:
        (args.output / "manifest.json").write_text(json.dumps(meta, indent=2)+"\n")
    else:
        prior_manifest = json.loads((args.output / "manifest.json").read_text())
        if any(prior_manifest.get(k) != meta[k] for k in ("model", "revision", "original_sha256", "max_calls")):
            raise ValueError("Resume model/data/budget differs")
        (args.output / "resume_manifest.json").write_text(json.dumps(meta, indent=2)+"\n")
    record(args.output, "started", experiment="approved-local-teacher-pilot", **meta)
    began = time.monotonic() - (previous[-1]["elapsed_seconds"] if previous else 0)
    calls = len(previous)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16),
        torch_dtype=torch.float16, device_map={"": 0}, attn_implementation="sdpa")
    model.eval()

    def call(kind, key, prompt, sampling=False):
        nonlocal calls
        if calls >= 144 or time.monotonic()-began >= 90*60:
            raise TimeoutError("Approved pilot budget exhausted")
        messages = [{"role": "system", "content": "You are a careful native Vietnamese grammar reviewer. Return only a JSON object. " + TAXONOMY},
                    {"role": "user", "content": prompt}]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        batch = tokenizer(rendered, return_tensors="pt").to(model.device)
        if batch.input_ids.shape[1] > 768:
            raise ValueError("Prompt exceeds approved input budget")
        options = dict(max_new_tokens=384, do_sample=sampling, pad_token_id=tokenizer.eos_token_id)
        if sampling:
            options.update(temperature=.65, top_p=.9)
        with torch.inference_mode():
            tokens = model.generate(**batch, **options)
        output = tokenizer.decode(tokens[0, batch.input_ids.shape[1]:], skip_special_tokens=True)
        result = parse_json(output)
        calls += 1
        event = dict(index=calls, kind=kind, key=key, prompt=prompt, output=output,
                     output_sha256=hashlib.sha256(output.encode()).hexdigest(), parsed=result,
                     elapsed_seconds=time.monotonic()-began)
        with journal.open("a") as handle:
            handle.write(json.dumps(event, ensure_ascii=False)+"\n")
        print(json.dumps(dict(call=calls, kind=kind, key=key, valid_json=result is not None,
                             elapsed_seconds=round(time.monotonic()-began))), flush=True)
        return event

    def verify(key, source, target):
        return call("blind_review", key, "Independently judge the source and target without knowing the proposed class. "
                    "Return {\"source_label\": integer 0..4, \"target_label\": integer 0..4, "
                    "\"single_error\": boolean, \"meaning_preserved\": boolean, \"required_edit\": boolean, "
                    "\"reason\": brief English explanation}. If source is grammatical choose 0; "
                    "optional stylistic edits are not errors. Assess ambiguity conservatively.\n" +
                    json.dumps(dict(source=source, target=target), ensure_ascii=False))

    try:
        for repair in repairs:
            if any(e["kind"] == "blind_review" and e["key"] == "repair-"+str(repair["id"]) for e in previous):
                continue
            verify("repair-"+str(repair["id"]), repair["original"]["Input"], repair["proposed_full_correction"])
        contexts = [r for r in train if r["label"] == 0 and len(tokenizer.encode(r["source"])) <= 220]
        random.Random(42).shuffle(contexts)
        for label in range(1, 5):
            for attempt in range(16):
                mechanism = MECHANISMS[label][attempt % 6]
                parent = contexts[((label-1)*16+attempt) % len(contexts)]
                key = f"teacher-{label}-{attempt:02d}"
                if any(e["kind"] == "generation" and e["key"] == key for e in previous):
                    continue
                event = call("generation", key, "Build a Vietnamese minimal error/clean pair. "
                    "The TARGET must be the given grammatical training sentence copied EXACTLY, including punctuation. "
                    "Create SOURCE by introducing one clear required grammatical error into that target. "
                    "Include exactly one clear grammatical error, requiring a minimal correction. "
                    "Preserve meaning, entities and numbers in the pair. No ellipsis, explanations inside sentences, "
                    "trivial repeated terminal punctuation, misspellings or optional stylistic revisions. "
                    "Do not add/remove optional tense particles merely to imply tense. Do not leave source identical to target. "
                    f"Required error class: {label}; mechanism: {mechanism}. "
                    "Return {\"source\": erroneous sentence, \"target\": grammatical complete sentence, "
                    "\"label\": integer, \"reason\": brief English explanation}. Training context:\n"+parent["source"], True)
                with (args.output / "parents.jsonl").open("a") as handle:
                    handle.write(json.dumps(dict(key=key, parent_ids=[parent["id"]], mechanism=mechanism,
                                                 label=label))+"\n")
                value = event["parsed"]
                if (isinstance(value, dict) and type(value.get("label")) is int and value["label"] == label
                        and all(isinstance(value.get(f), str) and value[f].strip() for f in ("source", "target"))):
                    verify(key, value["source"], value["target"])
        record(args.output, "completed", experiment="approved-local-teacher-pilot", calls=calls,
               elapsed_seconds=time.monotonic()-began)
    except Exception as exc:
        record(args.output, "failed", experiment="approved-local-teacher-pilot", calls=calls,
               elapsed_seconds=time.monotonic()-began, error_type=type(exc).__name__)
        raise


if __name__ == "__main__":
    main()
