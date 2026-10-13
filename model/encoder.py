"""Reload a native five-output encoder with the shared hierarchical decision."""
import json
from pathlib import Path


def classify(checkpoint, records, device="cuda", batch_size=8):
    import gc
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from .classification import _predict, _require_hierarchy
    checkpoint=Path(checkpoint)
    saved=json.loads((checkpoint/"classifier_config.json").read_text())
    _require_hierarchy(saved)
    if saved.get("backend")!="encoder" or saved.get("labels")!=list(range(5)):
        raise ValueError("Saved encoder backend/labels differ")
    model=AutoModelForSequenceClassification.from_pretrained(checkpoint,local_files_only=True)
    tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True)
    if (model.config.num_labels!=5 or sum(p.numel() for p in model.parameters())!=saved["parameters"]
            or saved["parameters"]>8_000_000_000 or len(tokenizer)!=saved["tokenizer_length"]
            or tokenizer.pad_token_id!=saved["pad_token_id"]):
        raise ValueError("Saved encoder architecture/tokenizer audit differs")
    preprocessing=json.loads((checkpoint/"preprocessing.json").read_text())
    model.to(device)
    try:
        return _predict(model,tokenizer,records,preprocessing,device,batch_size)
    finally:
        del model,tokenizer
        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
