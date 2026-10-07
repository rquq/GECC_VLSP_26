"""Load complete experiment settings and resolve project-local run destinations."""
import json
import math
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# These arguments belong to the response-only, single-device training workflow.
OWNED_TRAINING_ARGS = frozenset({
    "output_dir", "use_cpu", "no_cuda", "fp16", "bf16", "seed", "data_seed",
    "gradient_accumulation_steps", "save_strategy", "eval_strategy",
    "evaluation_strategy", "report_to", "label_names", "load_best_model_at_end",
    "eval_on_start", "do_eval", "do_train",
})


def _number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def load_config(path):
    """Read a complete nested JSON config; tuning values have no hidden defaults."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Config must be a JSON object")
    required = {
        "run": {"name", "output_root", "device", "seeds", "archive"},
        "data": {"path", "filename", "input_root", "validation_fraction", "train_limit", "validation_limit"},
        "model": {"name", "revision", "load_in_4bit", "compute_dtype", "quant_type", "use_double_quant", "attn_implementation"},
        "lora": {"r", "lora_alpha", "lora_dropout", "target_modules", "bias"},
        "training": {"max_length", "effective_batch_size", "fp16_initial_scale", "args"},
        "generation": set(),
        "evaluation": {"selection_metrics", "save_epoch_adapters"},
    }
    missing = set(required) | {"prompt"}
    missing -= config.keys()
    if missing:
        raise ValueError(f"Missing config fields: {', '.join(sorted(missing))}")
    for section, keys in required.items():
        if not isinstance(config[section], dict):
            raise ValueError(f"{section} must be an object")
        missing = keys - config[section].keys()
        if missing:
            raise ValueError(f"Missing {section} fields: {', '.join(sorted(missing))}")
    if not isinstance(config["prompt"], str) or not config["prompt"].strip():
        raise ValueError("prompt must be a nonempty system prompt string")
    run, data, model = config["run"], config["data"], config["model"]
    for section, keys in ((run, ("name", "output_root")), (data, ("filename", "input_root")),
                          (model, ("name", "revision", "quant_type", "attn_implementation"))):
        for key in keys:
            if not isinstance(section[key], str) or not section[key].strip():
                raise ValueError(f"{key} must be a nonempty string")
    if Path(run["name"]).name != run["name"] or run["name"] in {".", ".."}:
        raise ValueError("run.name must be a directory name, not a path")
    if run["device"] not in {"cuda", "cpu"}:
        raise ValueError("run.device must be cuda or cpu")
    seeds = run["seeds"]
    if not isinstance(seeds, list) or not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("run.seeds must be a nonempty list of nonnegative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("run.seeds must be distinct")
    for section, key in ((run, "archive"), (model, "load_in_4bit"), (model, "use_double_quant"),
                         (config["evaluation"], "save_epoch_adapters")):
        if type(section[key]) is not bool:
            raise ValueError(f"{key} must be a boolean")
    if model["compute_dtype"] not in {"auto", "float16", "bfloat16", "float32"}:
        raise ValueError("model.compute_dtype must be auto, float16, bfloat16, or float32")
    if data["path"] is not None and (not isinstance(data["path"], str) or not data["path"].strip()):
        raise ValueError("data.path must be null or a nonempty path string")
    if data["path"] is not None:
        data_path = Path(data["path"]).expanduser()
        if not data_path.is_absolute():
            data_path = PROJECT_ROOT / data_path
        data["path"] = str(data_path.resolve())
    _number(data["validation_fraction"], "data.validation_fraction")
    if not 0 < data["validation_fraction"] < 1:
        raise ValueError("data.validation_fraction must lie strictly between 0 and 1")
    for key in ("train_limit", "validation_limit"):
        if data[key] is not None:
            _positive_int(data[key], f"data.{key}")
    training = config["training"]
    for key in ("max_length", "effective_batch_size"):
        _positive_int(training[key], f"training.{key}")
    _number(training["fp16_initial_scale"], "training.fp16_initial_scale", positive=True)
    args = training["args"]
    if not isinstance(args, dict):
        raise ValueError("training.args must be an object of TrainingArguments settings")
    owned = OWNED_TRAINING_ARGS.intersection(args)
    if owned:
        raise ValueError(f"Workflow-owned training.args cannot be overridden: {', '.join(sorted(owned))}")
    missing = {"num_train_epochs", "max_steps", "per_device_train_batch_size", "gradient_checkpointing"} - args.keys()
    if missing:
        raise ValueError(f"Missing training.args fields: {', '.join(sorted(missing))}")
    _positive_int(args["per_device_train_batch_size"], "training.args.per_device_train_batch_size")
    accumulation, remainder = divmod(training["effective_batch_size"], args["per_device_train_batch_size"])
    if remainder or accumulation < 1:
        raise ValueError("Effective batch must be a positive multiple of per_device_train_batch_size")
    _number(args["num_train_epochs"], "training.args.num_train_epochs")
    if type(args["max_steps"]) is not int or args["max_steps"] == 0 or args["max_steps"] < -1:
        raise ValueError("training.args.max_steps must be -1 or a positive integer")
    if args["max_steps"] < 0 and args["num_train_epochs"] <= 0:
        raise ValueError("Training needs positive num_train_epochs or max_steps")
    if type(args["gradient_checkpointing"]) is not bool:
        raise ValueError("training.args.gradient_checkpointing must be a boolean")
    if "gradient_checkpointing_kwargs" in args and not isinstance(args["gradient_checkpointing_kwargs"], dict):
        raise ValueError("training.args.gradient_checkpointing_kwargs must be an object")
    for key in ("learning_rate", "weight_decay", "warmup_ratio", "max_grad_norm", "adam_epsilon"):
        if key in args:
            _number(args[key], f"training.args.{key}")
            if args[key] < 0:
                raise ValueError(f"training.args.{key} must be nonnegative")
    for key in ("warmup_ratio", "adam_beta1", "adam_beta2"):
        if key in args:
            _number(args[key], f"training.args.{key}")
            if not 0 <= args[key] <= 1:
                raise ValueError(f"training.args.{key} must lie between 0 and 1")
    _positive_int(config["lora"]["r"], "lora.r")
    _number(config["lora"]["lora_alpha"], "lora.lora_alpha", positive=True)
    _number(config["lora"]["lora_dropout"], "lora.lora_dropout")
    if not 0 <= config["lora"]["lora_dropout"] <= 1:
        raise ValueError("lora.lora_dropout must lie between 0 and 1")
    target_modules = config["lora"]["target_modules"]
    if not (isinstance(target_modules, str) and target_modules or
            isinstance(target_modules, list) and target_modules and
            all(isinstance(module, str) and module for module in target_modules)):
        raise ValueError("lora.target_modules must be a nonempty regex string or list of module names")
    for key in ("max_new_tokens", "max_length", "num_beams", "num_return_sequences"):
        if key in config["generation"]:
            _positive_int(config["generation"][key], f"generation.{key}")
    for key in ("task_type", "revision"):
        if key in config["lora"]:
            raise ValueError(f"lora.{key} is workflow-owned")
    paths = config["evaluation"]["selection_metrics"]
    if not isinstance(paths, list) or not paths or any(not isinstance(path, str) or not all(path.split(".")) for path in paths):
        raise ValueError("evaluation.selection_metrics must be a nonempty list of dotted metric paths")
    return config


def resolve_output(config, explicit=None):
    """Resolve CLI paths from cwd; default runs live under the project root."""
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    run = config["run"]
    root = Path(run["output_root"]).expanduser()
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return (root / f"{run['name']}-{timestamp}").resolve()
