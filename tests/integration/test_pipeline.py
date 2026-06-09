"""End-to-end scenarios that previously regressed — locks the real query -> result
behavior. Marked `integration` (loads models). Run: python -m pytest -m integration"""
import pytest

from src.scripts.graph.outfit_slots import get_slot

pytestmark = pytest.mark.integration

JACKET_ANCHOR = "0549330005"   # Dark Blue Jacket, cold (not in co-buy graph)
HOODIE = "0569973006"          # Grey Hoodie


def test_find_trousers_filters_to_trousers(run):
    r = run("find me trousers")
    assert r["log"]["must_filters"].get("product_type") == "Trousers"
    assert r["types"] and all(t == "Trousers" for t in r["types"])


def test_parka_oov_resolves_to_jacket(run):
    # "parka" is OOV -> corpus canonicalization must map it to Jacket (not Polo shirt / Coat).
    r = run("I want to find some dark blue parkas with a hood and a zip for outdoor adventures")
    assert r["log"]["must_filters"].get("product_type") == "Jacket"
    assert all(t == "Jacket" for t in r["types"])
    assert JACKET_ANCHOR in r["ids"]


def test_shoe_pairing_returns_footwear(run):
    # graph_pairing on a cold jacket -> compat fallback, category-level (slot) not "Other shoe".
    r = run("do you have shoe to match with that?", anchor=JACKET_ANCHOR)
    assert r["has_ctx"] and r["items"]
    assert "compat_pairing_fallback" in r["log"]["retrieval_path"]
    assert all(get_slot(t) == "shoe" for t in r["types"])


def test_anchor_type_not_mistaken_for_target(run):
    # "the jacket one ... shoe to match" -> target is shoe, not the anchor's own type (Jacket).
    r = run("the jacket one looks good, do you have shoe to match with that?", anchor=JACKET_ANCHOR)
    assert r["items"]
    assert all(get_slot(t) == "shoe" for t in r["types"])
    assert "Jacket" not in r["types"]


def test_chitchat_short_circuits(run):
    r = run("hello there")
    intent = (r["direct"] or {}).get("type", "") or r["log"].get("intent_hint", "")
    assert intent == "chit_chat"
    assert not r["items"]


def test_color_variant_cross_turn(rag_full, run):
    # Seed the session anchor (as if a hoodie was just shown), then ask for the black variant.
    sid, state = rag_full.sessions.get_or_create("variant-itest")
    rag_full.sessions.touch_anchor(state, HOODIE, [HOODIE])
    r = run("Do you have black color?", session="variant-itest")
    assert r["log"]["intent_hint"] == "color_variant"
    assert r["items"]
    assert all(t == "Hoodie" for t in r["types"])
    assert any("Black" in c for c in r["colours"])
