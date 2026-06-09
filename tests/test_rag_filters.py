"""FashionRAGService retrieval logic with stub embedder/store/llm (no models loaded)."""
import pytest

from src.backend.services.rag_service import FashionRAGService, HARD_FILTER_KEYS


class _Embedder:
    siglip = None  # forces _resolve_product_type to skip the SigLIP branch


class _Store:
    """hybrid_search returns results only when no hard filter is applied (simulates
    over-filtering), so the relaxation cascade must fall through to relax_no_filters."""
    def __init__(self, point):
        self.point = point

    def hybrid_search(self, **kw):
        return [] if kw.get("must_filters") else [self.point]


@pytest.fixture
def rag(catalog):
    return FashionRAGService(_Embedder(), _Store(None), object(), catalog,
                             limit=5, personalization=None, sessions=None)


def test_hard_filters_keeps_only_type_and_colour(rag):
    full = {"product_type": "Jacket", "colour_group": "Black",
            "fit": "Slim/Tailored", "occasion": "Casual/Everyday", "seasonality": "Spring/Summer"}
    hard = rag._hard_filters(full)
    assert set(hard) == set(HARD_FILTER_KEYS)
    assert hard == {"product_type": "Jacket", "colour_group": "Black"}


def test_resolve_product_type_sources(rag):
    assert rag._resolve_product_type("jacket") == ("Jacket", "exact")
    assert rag._resolve_product_type("trouser")[0] == "Trousers"   # synonym -> confident tier
    assert rag._resolve_product_type("parka") == ("Jacket", "corpus")
    # OOV with no corpus support + no SigLIP (siglip=None) -> unresolved, not hard
    assert rag._resolve_product_type("xyzblahnonsense") == ("", "")


def test_filter_points_by_product_type(rag, make_point):
    pts = [make_point("1", "Jacket"), make_point("2", "Trousers")]
    kept = rag._filter_points(pts, {"product_type": "Jacket"}, {})
    assert [p.payload["article_id"] for p in kept] == ["1"]


def test_relaxation_cascade_falls_through(catalog, make_point):
    point = make_point("99", "Jacket", "Black")
    rag = FashionRAGService(_Embedder(), _Store(point), object(), catalog,
                            limit=5, personalization=None, sessions=None)
    path = []
    out = rag._hybrid_with_relaxation(
        text_dense=None, image_dense=None, sparse_idx=[], sparse_val=[],
        must_filters={"product_type": "Jacket", "colour_group": "Black"},
        must_not_filters={}, limit=5, retrieval_path=path,
    )
    assert out == [point]
    assert path == ["hard_filters", "relax_to_product_type", "relax_no_filters"]


def test_image_knn_excludes_reference(catalog, make_point):
    ref = make_point("0549330005", "Jacket")
    others = [make_point("1", "Trousers"), make_point("2", "Trousers")]

    class _S:
        def hybrid_search(self, **kw):
            return [ref] + others

    rag = FashionRAGService(_Embedder(), _S(), object(), catalog,
                            limit=5, personalization=None, sessions=None)
    out = rag._image_knn_points([0.1] * 4, "0549330005", {}, {})
    ids = [p.payload["article_id"] for p in out]
    assert "0549330005" not in ids
    assert ids == ["1", "2"]
