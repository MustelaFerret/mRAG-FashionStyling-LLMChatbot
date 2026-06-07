from __future__ import annotations

import functools
from typing import Any, Callable, Dict, List

try:
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.tokenenforcer import TokenEnforcer, TokenEnforcerTokenizerData

    LMFE_AVAILABLE = True
except Exception:
    LMFE_AVAILABLE = False


def _build_regular_tokens_list(tokenizer, vocab_size: int) -> List[tuple[int, str, bool]]:
    token_0 = tokenizer.encode("0")[-1]
    special_ids = set(tokenizer.all_special_ids)
    regular_tokens: List[tuple[int, str, bool]] = []
    for token_idx in range(vocab_size):
        if token_idx in special_ids:
            continue
        decoded_after_0 = tokenizer.decode([token_0, token_idx])[1:]
        decoded_regular = tokenizer.decode([token_idx])
        is_word_start = len(decoded_after_0) > len(decoded_regular)
        regular_tokens.append((token_idx, decoded_after_0, is_word_start))
    return regular_tokens


def _decode(tokenizer, tokens: List[int]) -> str:
    return tokenizer.decode(tokens).rstrip("�")


def build_tokenizer_data(tokenizer):
    vocab_size = len(tokenizer)
    regular_tokens = _build_regular_tokens_list(tokenizer, vocab_size)
    decode_fn = functools.partial(_decode, tokenizer)
    return TokenEnforcerTokenizerData(regular_tokens, decode_fn, tokenizer.eos_token_id, False, vocab_size)


def build_prefix_allowed_tokens_fn(tokenizer_data, schema: Dict[str, Any]) -> Callable[[int, Any], List[int]]:
    parser = JsonSchemaParser(schema)
    enforcer = TokenEnforcer(tokenizer_data, parser)

    def prefix_allowed_tokens_fn(_batch_id: int, input_ids) -> List[int]:
        return enforcer.get_allowed_tokens(input_ids.tolist()).allowed_tokens

    return prefix_allowed_tokens_fn
