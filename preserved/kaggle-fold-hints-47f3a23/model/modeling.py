"""Response-only batch collation for correction training."""


class ResponseCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, records):
        import torch
        length = max(len(r["input_ids"]) for r in records)
        return {key: torch.tensor([r[key] + [pad] * (length - len(r[key])) for r in records])
                for key, pad in (("input_ids", self.pad_token_id), ("attention_mask", 0), ("labels", -100))}
