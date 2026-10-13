"""Native Sailor second opinion on existing reference-span copies; bounded pilot budget."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from model.data import load_records, split_records
from tools.experiment_log import record
from tools.local_teacher import TAXONOMY, parse_json

MODEL = "sail/Sailor-7B-Chat"
REVISION = "19066fae0a8a3ba029c190d8e3dacccf4d1234b8"
PARAMETERS = 7721324544


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original", "repairs", "pilot"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    import torch
    from accelerate import init_empty_weights
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        set_seed,
    )

    events = [
        json.loads(line)
        for line in (args.pilot / "calls.jsonl").read_text().splitlines()
    ]
    manifest = json.loads((args.pilot / "manifest.json").read_text())
    if any(e.get("teacher_name") == MODEL for e in events):
        raise ValueError(
            "Native second opinion already attempted; no implicit repeated reviews"
        )
    if (
        manifest["original_sha256"]
        != hashlib.sha256(args.original.read_bytes()).hexdigest()
    ):
        raise ValueError("Second opinion original differs")
    candidates = [
        r
        for r in json.loads(args.repairs.read_text())["candidate_changes"]
        if r["proposed_label"] != 0
    ]
    train, _ = split_records(load_records(args.original), 0.2, 42)
    allowed = {r["id"] for r in train}
    if (
        len(candidates) != 5
        or len(events) + 5 > 144
        or any(r["id"] not in allowed for r in candidates)
    ):
        raise ValueError(
            "Native review scope exceeds approved training-only pilot budget"
        )
    config = AutoConfig.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    AutoModelForCausalLM._model_mapping[type(config)]
    with init_empty_weights(include_buffers=True):
        skeleton = AutoModelForCausalLM.from_config(config)
    skeleton.tie_weights()
    if (
        sum(p.numel() for p in skeleton.parameters()) != PARAMETERS
        or PARAMETERS > 8_000_000_000
    ):
        raise ValueError("Native teacher parameter audit differs")
    del skeleton
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    work = args.pilot / "native-second-opinion"
    if work.exists():
        raise ValueError("Native review output exists")
    work.mkdir()
    began = time.monotonic()
    previous_seconds = max(e["elapsed_seconds"] for e in events)
    record(
        work,
        "started",
        experiment="bounded-native-reference-second-opinion",
        source_commit=source,
        model=MODEL,
        revision=REVISION,
        parameters=PARAMETERS,
        previous_calls=len(events),
        original_sha256=manifest["original_sha256"],
        reviews=5,
        new_sentences=0,
    )
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        revision=REVISION,
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map={"": 0},
        attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        ),
    )
    model.eval()
    try:
        for candidate in candidates:
            if previous_seconds + time.monotonic() - began >= 90 * 60:
                raise TimeoutError("Aggregate approved teacher time budget exhausted")
            # No original error label, generator judgement or reference annotation in this prompt.
            prompt = (
                "Phân loại từng câu độc lập theo: "
                + TAXONOMY
                + " Không mặc định câu nào đúng. Một thay đổi tùy chọn về phong cách KHÔNG phải lỗi. "
                "So sánh ý nghĩa và xác định sửa đổi có bắt buộc không. Chỉ trả về JSON: "
                '{"source_label":0..4,"target_label":0..4,"single_error":true/false,'
                '"meaning_preserved":true/false,"required_edit":true/false,"reason":"giải thích ngắn"}.\n'
                + json.dumps(
                    dict(
                        source=candidate["original"]["Input"],
                        target=candidate["proposed_full_correction"],
                    ),
                    ensure_ascii=False,
                )
            )
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            batch = tokenizer(rendered, return_tensors="pt").to(model.device)
            if batch.input_ids.shape[1] > 768:
                raise ValueError("Native review input exceeds approved token cap")
            with torch.inference_mode():
                tokens = model.generate(
                    **batch,
                    max_new_tokens=384,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            text = tokenizer.decode(
                tokens[0, batch.input_ids.shape[1] :], skip_special_tokens=True
            )
            event = dict(
                index=len(events) + 1,
                kind="blind_review",
                key="repair-" + str(candidate["id"]),
                prompt=prompt,
                output=text,
                parsed=parse_json(text),
                output_sha256=hashlib.sha256(text.encode()).hexdigest(),
                teacher_name=MODEL,
                teacher_revision=REVISION,
                teacher_parameters=PARAMETERS,
                source_commit=source,
                elapsed_seconds=previous_seconds + time.monotonic() - began,
            )
            events.append(event)
            with (args.pilot / "calls.jsonl").open("a") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            print(
                json.dumps(
                    dict(call=event["index"], key=event["key"], parsed=event["parsed"]),
                    ensure_ascii=False,
                ),
                flush=True,
            )
        record(
            work,
            "completed",
            experiment="bounded-native-reference-second-opinion",
            reviews=5,
            total_teacher_calls=len(events),
            elapsed_seconds=time.monotonic() - began,
        )
    except BaseException as exc:
        record(
            work,
            "failed",
            experiment="bounded-native-reference-second-opinion",
            error_type=type(exc).__name__,
            total_teacher_calls=len(events),
            elapsed_seconds=time.monotonic() - began,
        )
        raise


if __name__ == "__main__":
    main()
