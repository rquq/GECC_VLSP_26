"""Opt-in beams ranked by frozen-base grammar judgment and multilingual similarity."""
import math
from contextlib import nullcontext
from .correction_edits import encode_edits

DEFAULT_RERANKING = {
    "enabled": True, "num_candidates": 4,
    "semantic_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "semantic_device": "cpu", "grammar_weight": 0.5, "semantic_weight": 0.5,
}
GRAMMAR_PROMPT = (
    "Đánh giá ngữ pháp của câu Tiếng Việt sau. Câu có đúng ngữ pháp, cấu trúc câu, "
    "từ công cụ và dấu câu không? Chỉ trả lời Có hoặc Không.\nCâu:\n"
)


def reranking_config(value):
    if not isinstance(value, dict):
        raise ValueError("reranking must be an object")
    config = {**DEFAULT_RERANKING, **value}
    if type(config["enabled"]) is not bool:
        raise ValueError("reranking.enabled must be boolean")
    if type(config["num_candidates"]) is not int or not 2 <= config["num_candidates"] <= 8:
        raise ValueError("num_candidates must be an integer in 2..8")
    for name in ("grammar_weight", "semantic_weight"):
        weight = config[name]
        if type(weight) not in (int, float) or not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if config["semantic_device"] != "cpu":
        raise ValueError("Semantic reranking uses CPU to avoid a second CUDA model")
    if not isinstance(config["semantic_model"], str) or not config["semantic_model"]:
        raise ValueError("semantic_model must name a pretrained multilingual encoder")
    return config


def edit_cost(source, target):
    return sum(max(edit["end"] - edit["start"], len(edit["replacement"]))
               for edit in encode_edits(source, target)["edits"])


def select_candidate(source, candidates, config):
    """No thresholds/gating: grammar+semantics first, fewer edited characters only on ties."""
    valid = []
    total_weight = config["grammar_weight"] + config["semantic_weight"]
    for index, candidate in enumerate(candidates):
        if candidate["parse_error"] is not None:
            continue
        scores = candidate["scores"]
        grammar, semantic = scores["grammar_probability"], scores["semantic_cosine"]
        if not math.isfinite(grammar) or not 0 <= grammar <= 1 or not math.isfinite(semantic) or not -1 <= semantic <= 1:
            raise ValueError("Nonfinite or out-of-range reranking score")
        score = (config["grammar_weight"] * grammar + config["semantic_weight"] * (semantic + 1) / 2) / total_weight
        cost = edit_cost(source, candidate["target"])
        scores.update(combined=score, edited_characters=cost)
        valid.append((score, -cost, -index, index))
    if not valid:
        raise ValueError("Candidate set contains no valid candidate")
    return max(valid)[-1]


class FrozenReranker:
    """Reuse base weights with adapter disabled; never use trained corrector scores as judge."""

    def __init__(self, model, tokenizer, config, max_length, *, user_role="user"):
        from sentence_transformers import SentenceTransformer
        self.model, self.tokenizer = model, tokenizer
        self.config, self.max_length = config, max_length
        self.user_role = user_role
        self.encoder = SentenceTransformer(config["semantic_model"], device="cpu")
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

    def _answer_log_probability(self, prefix, answer):
        import torch
        answer_ids = self.tokenizer.encode(answer, add_special_tokens=False)
        if not answer_ids or not prefix or len(prefix) + len(answer_ids) > self.max_length:
            raise ValueError("Frozen grammar judgment exceeds token budget or has an empty answer")
        device = self.model.get_input_embeddings().weight.device
        ids = torch.tensor([prefix + answer_ids], device=device)
        logits = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
        answer_logits = logits[0, len(prefix) - 1:len(prefix) + len(answer_ids) - 1].float()
        labels = torch.tensor(answer_ids, device=answer_logits.device)
        return torch.log_softmax(answer_logits, dim=-1).gather(1, labels[:, None]).sum().item()

    def score(self, source, candidates):
        import torch
        valid = [candidate for candidate in candidates if candidate["parse_error"] is None]
        texts = [source] + [candidate["target"] for candidate in valid]
        embeddings = self.encoder.encode(texts, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
        similarities = (embeddings[1:] @ embeddings[0]).clamp(-1, 1).tolist()
        # PEFT provides a context manager restoring adapter state on exceptions. A bare base
        # is already frozen/unadapted (used only for explicit baseline evaluation).
        context = self.model.disable_adapter() if hasattr(self.model, "disable_adapter") else nullcontext()
        with torch.inference_mode(), context:
            for candidate, similarity in zip(valid, similarities):
                prefix = self.tokenizer.apply_chat_template(
                    [{"role": self.user_role, "content": GRAMMAR_PROMPT + candidate["target"]}],
                    tokenize=True, add_generation_prompt=True, return_dict=False)
                yes = self._answer_log_probability(prefix, "Có")
                no = self._answer_log_probability(prefix, "Không")
                probability = torch.softmax(torch.tensor([yes, no], dtype=torch.float64), dim=0)[0].item()
                candidate["scores"] = {"grammar_probability": probability, "semantic_cosine": similarity,
                                       "grammar_yes_log_probability": yes, "grammar_no_log_probability": no}
        return select_candidate(source, candidates, self.config)
