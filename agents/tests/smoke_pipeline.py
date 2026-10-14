"""Exercise the real two-stage pipeline with a tiny local Qwen fixture, no downloads.

Run with --device cuda for NF4, or --device cpu --world-size 2 for DDP handoffs.
The toy corpus and whole-JSON fixture tokens test mechanics, not language quality.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import string
import sys
import zipfile

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from model.runtime import resolve_output, write_json
from tools.build_dataset import build
from tools.run_experiment import prepare_run


def fixture(root, device, final_lora=False):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from tokenizers.decoders import Fuse
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(42)
    original = [{"ID": label * 10 + i, "Input": f"label{label} item{i} token .",
                 "Output": {"Error_Type_Id": label, "Fixed Sentence": None if label == 0 else f"fixed{label} item{i} token ."}}
                for label in range(5) for i in range(5)]
    pool = [{"ID": f"syn:{i}", "Input": f"clean syn{i} token .",
             "Output": {"Error_Type_Id": 0, "Fixed Sentence": f"clean syn{i} token ."},
             "Provenance": {"template_rule": "smoke-clean", "parent_ids": []}} for i in range(10)]
    original_path, pool_path = root / "original.json", root / "pool.json"
    write_json(original_path, original)
    write_json(pool_path, pool)
    data = root / "data"
    build(original_path, pool_path, data, 0.2)
    (data / "VGEC 2026_v1.json").write_bytes(original_path.read_bytes())
    write_json(data / "public_test.json", [{"ID": 1, "Input": "label1 item1 token ."},
                                            {"ID": "1", "Input": "label0 item2 token ."}])
    targets = [row["Output"]["Fixed Sentence"] or row["Input"] for row in original + pool]
    answers = sorted({json.dumps({"corrected_sentence": target}) for target in targets})
    vocabulary = {"<pad>": 0, "<unk>": 1, "<eos>": 2, "token": 3, ".": 4, "0": 5}
    for token in string.ascii_letters + string.digits + string.punctuation + "".join(answers):
        if token not in vocabulary and not token.isspace():
            vocabulary[token] = len(vocabulary)
    for answer in answers:
        vocabulary[answer] = len(vocabulary)
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = WhitespaceSplit()
    backend.decoder = Fuse()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", unk_token="<unk>", eos_token="<eos>")
    # LMFE enumerates the base vocabulary; register whole-answer matches there
    # before adding no-split behavior, rather than hide them in added-only IDs.
    tokenizer.add_tokens(answers)
    tokenizer.chat_template = "{% for message in messages %}{{message['content']}} {% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
    base = root / "base"
    tokenizer.save_pretrained(base)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
                             num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                             max_position_embeddings=512, pad_token_id=0, eos_token_id=2, tie_word_embeddings=False))
    # A deterministic native-model protocol fixture: one complete JSON token,
    # then EOS. Its embedding changes sign after that answer so FP16/NF4
    # rounding cannot turn the random base into an endless string generator.
    with torch.no_grad():
        model.model.embed_tokens.weight.zero_()
        model.model.embed_tokens.weight[:, 0] = 1
        for answer in answers:
            model.model.embed_tokens.weight[vocabulary[answer], 0] = -1
        model.model.norm.weight.zero_()
        model.model.norm.weight[0] = 1
        model.lm_head.weight.zero_()
        model.lm_head.weight[vocabulary[answers[0]], 0] = 5
        model.lm_head.weight[2, 0] = -4
    model.save_pretrained(base)
    for task, filename in (("classification", "baseline.json"), ("correction", "correction.json")):
        config = json.loads((PROJECT / "configs" / filename).read_text())
        config["model"].update(name=str(base), revision="fixture", load_in_4bit=device == "cuda" and not final_lora, device_map="single",
                               compute_dtype="float16" if device == "cuda" else "float32", attn_implementation="eager")
        config["data"]["validation_fraction"] = 0 if final_lora else .2
        if not final_lora:
            config["model"].update(quant_type="nf4", use_double_quant=True)
        config["lora"].update(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"])
        config["training"].update(max_length=128, effective_batch_size=4)
        config["training"]["args"].update(num_train_epochs=1, optim="adamw_torch", learning_rate=1e-3,
                                           logging_steps=1, warmup_steps=0, warmup_ratio=0.0)
        config["tracking"]["enabled"] = False
        if task == "classification":
            config["oof"].update(folds=2, epochs=1)
        else:
            config["generation"]["max_new_tokens"] = 8
        write_json(root / f"{task}.json", config)
    if final_lora:
        from tools.package_final_training import final_bundle
        for stage, filename in [("classification", "VGEC_classifier.json"), ("correction", "VGEC_balanced.json")]:
            write_json(data / filename, final_bundle(data / filename, stage))
    recipe = json.loads((PROJECT / "configs/experiment.json").read_text())
    recipe.update(name="tiny-pipeline", classification_config=str(root / "classification.json"),
                  correction_config=str(root / "correction.json"),mode="both",classifier_run=None)
    recipe["data_sha256"] = {role: hashlib.sha256((data / filename).read_bytes()).hexdigest()
                             for role, filename in (("original", "VGEC 2026_v1.json"), ("classification", "VGEC_classifier.json"),
                                                    ("correction", "VGEC_balanced.json"))}
    write_json(root / "experiment.json", recipe)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--world-size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--final-lora", action="store_true", help="Exercise full training without quantization or validation")
    args = parser.parse_args()
    root = resolve_output({"run": {"name": f"smoke-{args.device}-{args.world_size}", "output_root": "runs"}})
    root.mkdir(mode=0o700)
    data = fixture(root, args.device, args.final_lora)
    work, command = prepare_run(root / "experiment.json", data, root / "outputs",
                                device=args.device, world_size=args.world_size)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "USE_TF": "0", "USE_FLAX": "0"}
    with (root / "pipeline.log").open("w") as log:
        subprocess.run(command, cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    run = work / "run"
    with zipfile.ZipFile(run / "results.zip") as archive:
        assert archive.namelist() == ["results.json"]
        rows = json.loads(archive.read("results.json"))
        assert len(rows) == 2 and [row["ID"] for row in rows] == [1, "1"]
        assert all(isinstance(row["Output"]["Fixed Sentence"], list) and len(row["Output"]["Fixed Sentence"]) == 1 for row in rows)
    manifest = json.loads((run / "pipeline_manifest.json").read_text())
    assert manifest["submission"]["status"] == "completed" and not manifest["submission"]["evaluated"]
    assert (run / "run_handoff.json").is_file()
    classifier_config = json.loads((run / "classification/config.json").read_text())
    expected_folds = classifier_config["oof"]["enabled"]
    assert (run / "classification/oof_metrics.json").is_file() == expected_folds
    assert (run / "classification/hints.json").is_file() == expected_folds
    assert (run / "correction/metrics.json").is_file()
    if args.final_lora:
        for stage, flag in [("classification", "saved_classifier_reload_verified"), ("correction", "saved_adapter_reload_verified")]:
            stage_manifest = json.loads((run / stage / "run_manifest.json").read_text())
            assert stage_manifest["final_fit"] and stage_manifest[flag]
            assert not stage_manifest["model"]["load_in_4bit"]
            assert json.loads((run / stage / "validation.json").read_text()) == []
            assert json.loads((run / stage / "metrics.json").read_text())["classification"] is None
            assert len(json.loads((run / stage / "train.json").read_text())) >= 25
        assert manifest["combined_score"] is None
    inference = json.loads((run / "correction/adapter/inference.json").read_text())
    assert inference["hint_mode"] == json.loads((root / "correction.json").read_text())["inference"]["hint_mode"]
    print(json.dumps({"status": "passed", "device": args.device, "world_size": args.world_size,
                      "run": str(run), "log": str(root / "pipeline.log"), "submission_rows": len(rows),
                      "oof_enabled": expected_folds, "hint_mode": inference["hint_mode"]}, indent=2))


if __name__ == "__main__":
    main()
