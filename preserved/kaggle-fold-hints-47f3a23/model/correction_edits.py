"""Strict localized corrections using Python Unicode code-point offsets."""
import difflib
import json

EDIT_PROMPT = (
    "Hãy sửa lỗi ngữ pháp Tiếng Việt, sửa ít nhất có thể và giữ nguyên ý nghĩa. "
    "Giữ nguyên câu đúng. Chỉ trả lời JSON với duy nhất khóa edits là mảng các đối tượng "
    "có đúng ba khóa start, end, replacement. start và end là số nguyên chỉ số ký tự Unicode "
    "(code point, không phải byte), tính từ 0 trong câu gốc, khoảng [start,end). "
    "Các edits phải được sắp tăng theo start, không chồng lấn; chèn dùng start=end, "
    "xóa dùng replacement rỗng. Không có lỗi thì trả lời {\"edits\":[]}. Không giải thích."
)
SENTENCE_NO_HINT_PROMPT = (
    "Hãy sửa lỗi ngữ pháp Tiếng Việt. Sửa ít nhất có thể, giữ nguyên ý nghĩa và hãy giữ nguyên "
    "câu đúng. Chỉ trả lời JSON với duy nhất khóa corrected_sentence chứa câu hoàn chỉnh; không giải thích."
)


def encode_edits(source, target):
    """Deterministic source-relative diff; no Unicode normalization or byte conversion."""
    edits = []
    for tag, start, end, target_start, target_end in difflib.SequenceMatcher(
            None, source, target, autojunk=False).get_opcodes():
        if tag != "equal":
            edits.append({"start": start, "end": end, "replacement": target[target_start:target_end]})
    return {"edits": edits}


def apply_edits(source, value):
    if not isinstance(value, dict) or set(value) != {"edits"} or not isinstance(value["edits"], list):
        raise ValueError("Expected JSON with only an edits array")
    parts, cursor, previous_start = [], 0, -1
    for edit in value["edits"]:
        if not isinstance(edit, dict) or set(edit) != {"start", "end", "replacement"}:
            raise ValueError("Each edit requires only start, end, replacement")
        start, end, replacement = edit["start"], edit["end"], edit["replacement"]
        if type(start) is not int or type(end) is not int or not isinstance(replacement, str):
            raise ValueError("Edit offsets must be integers and replacement a string")
        if not 0 <= start <= end <= len(source):
            raise ValueError("Edit offsets outside source bounds")
        if start < cursor or start <= previous_start:
            raise ValueError("Edits must be sorted, nonoverlapping, with distinct start offsets")
        parts.extend((source[cursor:start], replacement))
        cursor, previous_start = end, start
    parts.append(source[cursor:])
    return "".join(parts)


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def parse_edits(source, text):
    try:
        return apply_edits(source, json.loads(text, object_pairs_hook=_unique_object)), None
    except (ValueError, TypeError, KeyError) as error:
        return "", str(error)
