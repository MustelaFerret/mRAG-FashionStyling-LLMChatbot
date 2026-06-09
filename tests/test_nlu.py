"""NLU schema + filter normalization. Uses __new__ to avoid loading any model."""
from src.backend.retrieval.llm import (
    INTENT_CHAT,
    INTENT_GRAPH,
    INTENT_SIMILAR,
    QwenMultimodalService,
)

VOCAB = {
    "product_type": ["Jacket", "Trousers", "Sneakers"],
    "colour_group": ["Black", "Dark Blue"],
    "fit": ["Regular/Straight", "Unknown"],
    "occasion": ["Casual/Everyday"],
    "seasonality": ["Spring/Summer"],
}


def _svc():
    return QwenMultimodalService.__new__(QwenMultimodalService)


def test_schema_product_type_free_text_others_enum():
    schema = _svc()._build_filter_schema(VOCAB)
    props = schema["properties"]["must_filters"]["properties"]
    # product_type is free text -> NO enum (OOV terms like "parka" must pass through)
    assert props["product_type"] == {"type": "string"}
    # closed-vocabulary fields stay enum-constrained, "Unknown" filtered out
    assert "enum" in props["colour_group"]
    assert "Unknown" not in props["fit"]["enum"]
    # must_not_filters removed from the constrained schema (hallucination-prone)
    assert "must_not_filters" not in schema["properties"]
    assert "search_query_en" in schema["required"]


def test_normalize_filters_list_and_allowed():
    out = QwenMultimodalService._normalize_filters(
        {"product_type": ["pants"], "colour_group": "", "bogus_key": "x"}
    )
    assert out == {"product_type": "pants"}  # list -> first non-empty; empty + unknown dropped


def test_normalize_intent_aliases():
    svc = _svc()
    assert svc._normalize_intent("graph_pairing") == INTENT_GRAPH
    assert svc._normalize_intent("matching") == INTENT_GRAPH
    assert svc._normalize_intent("similar") == INTENT_SIMILAR
    assert svc._normalize_intent("chitchat") == INTENT_CHAT
    assert svc._normalize_intent("nonsense-label") == INTENT_SIMILAR  # safe default
