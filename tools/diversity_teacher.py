"""Bounded local <=8B inference worker with private, replayable schema-constrained evidence."""

import argparse
import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
import hashlib
import json
from pathlib import Path
import time

MODELS = {
    "author": {
        "name": "Qwen/Qwen3-4B-Instruct-2507",
        "revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
        "parameters": 4022468096,
        "license": "Apache-2.0",
    },
    "reviewer": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
        "parameters": 7615616512,
        "license": "Apache-2.0",
    },
}

MODELS["reviewer4b"] = dict(MODELS["author"])


def native_parameter_count(settings):
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    architecture = AutoConfig.from_pretrained(
        settings["name"], revision=settings["revision"], local_files_only=True
    )
    AutoModelForCausalLM._model_mapping[type(architecture)]
    with init_empty_weights(include_buffers=True):
        model = AutoModelForCausalLM.from_config(architecture)
    model.tie_weights()
    return sum(p.numel() for p in model.parameters())


def run(tasks_path, output, role, max_seconds=3600):
    import torch
    from accelerate import init_empty_weights
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.integrations.transformers import (
        build_token_enforcer_tokenizer_data,
        build_transformers_prefix_allowed_tokens_fn,
    )
    from model.reproducibility import seed_everything
    from model.retained_rules import digest
    from model.runtime import write_json

    tasks_path, output = Path(tasks_path), Path(output)
    tasks = json.loads(tasks_path.read_text())
    if not tasks or len(tasks) > 384 or len({t["id"] for t in tasks}) != len(tasks):
        raise ValueError("Expected 1..384 unique bounded tasks")
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    began = time.monotonic()
    seed_everything(42)
    settings = MODELS[role]
    count = native_parameter_count(settings)
    if count != settings["parameters"] or count > 8_000_000_000:
        raise ValueError("Native architecture eligibility differs")
    tokenizer = AutoTokenizer.from_pretrained(
        settings["name"], revision=settings["revision"], local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        settings["name"],
        revision=settings["revision"],
        local_files_only=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        ),
        torch_dtype=torch.float16,
        device_map={"": 0},
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.generation_config.temperature = 1.0
    model.generation_config.top_p = 1.0
    model.generation_config.top_k = 50
    token_data = build_token_enforcer_tokenizer_data(tokenizer)
    manifest = {
        "format": "vgec-diversity-teacher-v1",
        "model": settings,
        "native_parameters": count,
        "tasks_sha256": hashlib.sha256(tasks_path.read_bytes()).hexdigest(),
        "seed": 42,
        "precision": "local NF4 inference; native parameter count unchanged; standard-LoRA training is separate",
        "max_seconds": max_seconds,
        "tasks": len(tasks),
        "schema_enforced": True,
        "scope": "automated screening, not human adjudication",
    }
    write_json(output / "manifest.json", manifest)
    with (output / "calls.jsonl").open("w") as journal:
        for task in tasks:
            if time.monotonic() - began > max_seconds:
                break
            rendered = tokenizer.apply_chat_template(
                task["messages"], tokenize=False, add_generation_prompt=True
            )
            batch = tokenizer(rendered, return_tensors="pt").to("cuda:0")
            if batch.input_ids.shape[1] > (
                4096 if role in ("author", "reviewer4b") else 1536
            ):
                raise ValueError("Input budget exceeded; no truncation")
            torch.manual_seed(42 + task["seed_offset"])
            prefix = build_transformers_prefix_allowed_tokens_fn(
                token_data, JsonSchemaParser(task["schema"])
            )
            options = {
                "max_new_tokens": task.get("max_new_tokens", 256),
                "do_sample": role == "author",
                "pad_token_id": tokenizer.eos_token_id,
                "use_cache": True,
                "prefix_allowed_tokens_fn": prefix,
                "max_time": 90,
            }
            if role == "author":
                options.update(temperature=0.65, top_p=0.9)
            with torch.inference_mode():
                generated = model.generate(**batch, **options)
            text = tokenizer.decode(
                generated[0, batch.input_ids.shape[1] :], skip_special_tokens=True
            )
            try:
                value = json.loads(text)
            except ValueError:
                value = None
            event = {
                "task": task,
                "task_sha256": digest(task),
                "model": settings,
                "output": text,
                "parsed": value,
                "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "elapsed_seconds": time.monotonic() - began,
                "generated_tokens": len(generated[0]) - batch.input_ids.shape[1],
            }
            journal.write(json.dumps(event, ensure_ascii=False) + "\n")
            journal.flush()
            print(
                json.dumps(
                    {
                        "id": task["id"],
                        "valid_json": value is not None,
                        "elapsed_seconds": round(event["elapsed_seconds"]),
                        "peak_vram_gib": round(
                            torch.cuda.max_memory_allocated() / 1024**3, 2
                        ),
                    }
                ),
                flush=True,
            )
    write_json(
        output / "completed.json",
        {
            "calls": sum(1 for _ in (output / "calls.jsonl").open()),
            "elapsed_seconds": time.monotonic() - began,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--role", choices=MODELS, required=True)
    p.add_argument("--max-seconds", type=int, default=3600)
    a = p.parse_args()
    run(a.tasks, a.output, a.role, a.max_seconds)
