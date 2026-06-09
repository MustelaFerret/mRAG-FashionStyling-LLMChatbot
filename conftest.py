"""Shared pytest fixtures. Tests here are model-free: they exercise the deterministic
NLU/retrieval logic (slot inference, OOV canonicalization, hard/soft filters, compat
index) without loading SigLIP/Qwen/DeBERTa, so they run in seconds on CPU."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from src.backend.core.config import settings
from src.backend.services.catalog import FashionCatalog


@pytest.fixture(scope="session")
def catalog():
    return FashionCatalog(settings.meta_file, settings.graph_file, settings.image_dir)


class StubPoint:
    """Minimal stand-in for a Qdrant point (only .payload is used downstream)."""
    def __init__(self, article_id="", product_type="", colour_group="", **extra):
        self.payload = {"article_id": article_id, "product_type": product_type,
                        "colour_group": colour_group, **extra}


@pytest.fixture
def make_point():
    return StubPoint
