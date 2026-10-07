"""Single-device model loading shared by QLoRA training and saved-adapter inference."""
import json
from pathlib import Path


def load_base(settings, device="cuda"):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for QLoRA")
    load_in_4bit = settings["load_in_4bit"]
    if load_in_4bit and device != "cuda":
        raise ValueError("4-bit QLoRA requires CUDA; disable quantization for CPU diagnostics")
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device == "cuda" else torch.float32
    if settings["compute_dtype"] != "auto":
        dtype = getattr(torch, settings["compute_dtype"])
    kwargs = dict(revision=settings["revision"], torch_dtype=dtype,
                  attn_implementation=settings["attn_implementation"])
    if load_in_4bit:
        kwargs.update(quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type=settings["quant_type"],
            bnb_4bit_use_double_quant=settings["use_double_quant"], bnb_4bit_compute_dtype=dtype),
            device_map={"": torch.cuda.current_device()})
    model = AutoModelForCausalLM.from_pretrained(settings["name"], **kwargs)
    if not load_in_4bit:
        model = model.to(device)
    return model, dtype


def load_adapter(adapter, device="cuda"):
    from peft import PeftConfig, PeftModel, prepare_model_for_kbit_training
    from transformers import AutoTokenizer

    config = PeftConfig.from_pretrained(adapter)
    settings_path = Path(adapter) / "loading.json"
    # Legacy adapters saved only load_in_4bit and compute_dtype.
    settings = {"quant_type": "nf4", "use_double_quant": True,
                "attn_implementation": "sdpa", **json.loads(settings_path.read_text())}
    settings.update(name=config.base_model_name_or_path, revision=config.revision)
    base, _ = load_base(settings, device)
    if settings["load_in_4bit"]:
        # Match training's FP32 nonquantized parameters without enabling checkpointing.
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
    return PeftModel.from_pretrained(base, adapter), AutoTokenizer.from_pretrained(adapter)
