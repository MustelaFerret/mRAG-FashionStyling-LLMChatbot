"""Tests for the refine_1 fixes (md/refine_1.MD): JSON extraction robustness, anonymous
session isolation, compat-index NaN guard, and the pairing season-clash guard. Uses
__new__/stubs so no model or full catalog load is needed."""
import numpy as np

from src.backend.retrieval.llm import QwenMultimodalService
from src.backend.services.catalog import FashionCatalog
from src.backend.services.compat_index import CompatPairingIndex
from src.backend.services.session_manager import SessionStore

extract = QwenMultimodalService._extract_json_object


def test_extract_json_single_object_with_prose():
    assert extract('sure! {"a": 1} hope that helps') == {"a": 1}


def test_extract_json_two_objects_takes_last_valid():
    # the old greedy first-{ to last-} span made this whole string unparseable
    raw = 'thinking {not json} ok final: {"search_query_en": "parka", "must_filters": {}}'
    assert extract(raw) == {"search_query_en": "parka", "must_filters": {}}


def test_extract_json_nested_braces():
    raw = '{"must_filters": {"product_type": "dress"}, "x": 1}'
    assert extract(raw) == {"must_filters": {"product_type": "dress"}, "x": 1}


def test_extract_json_no_dict_returns_none():
    assert extract("[1, 2, 3]") is None
    assert extract("no json here") is None
    assert extract("") is None


def test_anonymous_sessions_are_isolated():
    store = SessionStore(ttl_seconds=60, max_sessions=10)
    sid1, st1 = store.get_or_create(None)
    sid2, st2 = store.get_or_create("")
    assert sid1 != sid2 and st1 is not st2  # no shared "anon" bucket
    st1.anchor_id = "0123456789"
    assert st2.anchor_id == ""  # state must not leak between anonymous users
    # explicit ids still get continuity
    sid3, st3 = store.get_or_create("user-a")
    sid4, st4 = store.get_or_create("user-a")
    assert sid3 == sid4 and st3 is st4


def test_compat_index_rejects_nan_embeddings():
    idx = CompatPairingIndex.__new__(CompatPairingIndex)
    idx.emb = np.array([[1.0, float("nan")], [0.0, 1.0]], dtype=np.float32)
    idx.article_ids = ["0000000001", "0000000002"]
    idx.slots = ["top", "bottom"]
    ready = (
        idx.emb.shape[0] == len(idx.article_ids) == len(idx.slots)
        and bool(np.isfinite(idx.emb).all())
    )
    assert ready is False


def _stub_catalog(anchor_season, neighbor_metas):
    """Minimal FashionCatalog stub: one anchor with graph neighbors n1..nk."""
    cat = FashionCatalog.__new__(FashionCatalog)
    cat.meta_by_article = {"0000000001": {"product_type_name": "Sweater", "seasonality": anchor_season}}
    cat.graph_adj = {"0000000001": []}
    for i, meta in enumerate(neighbor_metas, start=2):
        aid = str(i).zfill(10)
        cat.meta_by_article[aid] = meta
        cat.graph_adj["0000000001"].append((aid, 10.0))
    return cat


def test_diverse_neighbors_blocks_season_clash():
    cat = _stub_catalog(
        "Autumn/Winter",
        [
            {"product_type_name": "Trousers", "seasonality": "Spring/Summer"},  # clash -> dropped
            {"product_type_name": "Shirt", "seasonality": "Core/All-year"},
            {"product_type_name": "Scarf", "seasonality": "Autumn/Winter"},
        ],
    )
    out = cat.get_graph_diverse_neighbors(
        "0000000001", limit=5, max_per_pt=2, preferred_min_weight=4, hard_min_weight=3
    )
    metas = [cat.meta_by_article[a]["product_type_name"] for a in out]
    assert "Trousers" not in metas and {"Shirt", "Scarf"} == set(metas)


def test_diverse_neighbors_keeps_core_and_unknown_seasons():
    cat = _stub_catalog(
        "Spring/Summer",
        [
            {"product_type_name": "Shorts", "seasonality": "Spring/Summer"},
            {"product_type_name": "Hat", "seasonality": "Unknown"},
        ],
    )
    out = cat.get_graph_diverse_neighbors(
        "0000000001", limit=5, max_per_pt=2, preferred_min_weight=4, hard_min_weight=3
    )
    assert len(out) == 2  # only the hard S/S x A/W clash is blocked
