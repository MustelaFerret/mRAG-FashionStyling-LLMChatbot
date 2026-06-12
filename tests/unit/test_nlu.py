"""NLU schema + filter normalization + attribute gazetteer. __new__ avoids loading models."""
from src.backend.retrieval.llm import (
    INTENT_CHAT,
    INTENT_GRAPH,
    INTENT_SIMILAR,
    QwenMultimodalService,
)
from src.backend.services.attribute_gazetteer import AttributeGazetteer

VOCAB = {
    "product_type": ["Jacket", "Trousers", "Sneakers"],
    "colour_group": ["Black", "Dark Blue", "Greenish Khaki"],
    "fit": ["Regular/Straight", "Tight/Skinny/Bodycon", "Unknown"],
    "occasion": ["Casual/Everyday", "Party/Evening/Wedding"],
    "seasonality": ["Spring/Summer", "Autumn/Winter"],
}


def _svc():
    return QwenMultimodalService.__new__(QwenMultimodalService)


def test_schema_only_product_type_free_text():
    # enum fields moved to the gazetteer (md/audit_nlu.md); the LLM schema is product_type
    # (free text, so OOV terms like "parka" pass through) + the query rewrite only.
    schema = _svc()._build_filter_schema(VOCAB)
    props = schema["properties"]["must_filters"]["properties"]
    assert props["product_type"] == {"type": "string"}
    assert set(props) == {"product_type"}  # no colour/fit/occasion/seasonality enum
    assert "must_not_filters" not in schema["properties"]
    assert "search_query_en" in schema["required"]


def test_gazetteer_extracts_enums_correctly():
    g = AttributeGazetteer()
    # winter -> Autumn/Winter (not Spring/Summer), wedding -> Party (not Beach): the
    # hallucinations the LLM made. Longest-match handles "dark blue".
    assert g.extract("a warm winter coat", VOCAB).get("seasonality") == "Autumn/Winter"
    assert g.extract("dress for a wedding", VOCAB).get("occasion") == "Party/Evening/Wedding"
    assert g.extract("black skinny jeans", VOCAB) == {
        "colour_group": "Black", "fit": "Tight/Skinny/Bodycon"
    }
    assert g.extract("navy trousers", VOCAB).get("colour_group") == "Dark Blue"


def test_gazetteer_no_false_positive_and_vocab_restricts():
    g = AttributeGazetteer()
    # "parka" names no enum attribute -> nothing extracted (no spurious filter)
    assert g.extract("find me a parka", VOCAB) == {}
    # a target absent from the live vocab is never emitted
    assert "occasion" not in g.extract("loungewear for sleeping", {"occasion": ["Casual/Everyday"]})


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
