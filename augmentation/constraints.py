"""LMFE's public parser core with a Transformers-version-independent callback."""
from functools import partial


def _decode(tokenizer, tokens):
    return tokenizer.decode(tokens).rstrip('\ufffd')


def tokenizer_data(tokenizer):
    from lmformatenforcer import TokenEnforcerTokenizerData
    prefix = tokenizer.encode('0', add_special_tokens=False)[-1]
    special = set(tokenizer.all_special_ids)
    tokens = []
    for index in range(len(tokenizer)):
        if index in special:
            continue
        after_prefix = tokenizer.decode([prefix, index])[1:]
        standalone = tokenizer.decode([index])
        tokens.append((index, after_prefix, len(after_prefix) > len(standalone)))
    return TokenEnforcerTokenizerData(tokens, partial(_decode, tokenizer), tokenizer.eos_token_id, False, len(tokenizer))


def prefix_filter(data, schema):
    from lmformatenforcer import JsonSchemaParser, TokenEnforcer
    enforcer = TokenEnforcer(data, JsonSchemaParser(schema))
    def allowed(batch_id, input_ids):
        return enforcer.get_allowed_tokens(input_ids.tolist()).allowed_tokens
    return allowed
