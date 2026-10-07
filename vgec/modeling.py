"""Shared training/inference prompt and strict generated-output parsing."""
import json

SYSTEM_PROMPT = (
    "Bạn sửa lỗi ngữ pháp tiếng Việt và phân loại lỗi. "
    "Chỉ trả lời một đối tượng JSON với hai khóa error_type_id và corrected_sentence. "
    "error_type_id là một số nguyên thuộc [0, 1, 2, 3, 4]. "
    "Nhãn 0: câu không có lỗi, giữ nguyên câu. "
    "Nhãn 1: lỗi cấu trúc câu (thiếu hoặc sai quan hệ chủ ngữ, vị ngữ, các vế câu). "
    "Nhãn 2: lỗi cấu trúc ngữ đoạn (sắp xếp hoặc kết hợp thành phần trong cụm từ). "
    "Nhãn 3: lỗi về sử dụng từ công cụ (quan hệ từ, liên từ, từ phụ trợ). "
    "Nhãn 4: lỗi cách chấm câu (thiếu, thừa hoặc dùng sai dấu câu). "
    "Sửa ít nhất có thể, giữ nguyên ý nghĩa; không giải thích."
)


def prompt_ids(tokenizer, source):
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": source}],
        tokenize=True, add_generation_prompt=True, return_dict=False,
    )


def encode_record(tokenizer, record, max_length):
    prompt = prompt_ids(tokenizer, record["source"])
    answer = json.dumps({"error_type_id": record["label"],
                         "corrected_sentence": record["target"]}, ensure_ascii=False)
    completion = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
    ids = prompt + completion
    if len(ids) > max_length:
        raise ValueError(f"ID {record['id']}: {len(ids)} tokens exceeds max_length={max_length}; increase it, do not truncate labels")
    return {"input_ids": ids, "attention_mask": [1] * len(ids),
            "labels": [-100] * len(prompt) + completion}


class ResponseCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, records):
        import torch
        length = max(len(r["input_ids"]) for r in records)
        return {key: torch.tensor([r[key] + [pad] * (length - len(r[key])) for r in records])
                for key, pad in (("input_ids", self.pad_token_id), ("attention_mask", 0), ("labels", -100))}


def parse_completion(text):
    try:
        value = json.loads(text)
        label, target = value["error_type_id"], value["corrected_sentence"]
        if set(value) != {"error_type_id", "corrected_sentence"} or type(label) is not int or label not in range(5) or not isinstance(target, str) or not target.strip():
            raise ValueError("Invalid label or corrected sentence")
        return label, target, None
    except (ValueError, TypeError, KeyError, IndexError) as error:
        # No source-copy or class-0 fallback: invalid generations remain visible.
        return None, "", str(error)


def generate_predictions(model, tokenizer, records, max_new_tokens=192):
    import torch
    model.eval()
    results = []
    with torch.inference_mode():
        for record in records:
            ids = torch.tensor([prompt_ids(tokenizer, record["source"])], device=model.device)
            output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                    max_new_tokens=max_new_tokens, do_sample=False,
                                    pad_token_id=tokenizer.pad_token_id,
                                    eos_token_id=tokenizer.eos_token_id, use_cache=True)
            text = tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True).strip()
            label, target, error = parse_completion(text)
            results.append({"id": record["id"], "label": label, "target": target,
                            "raw_generation": text, "parse_error": error})
    return results
