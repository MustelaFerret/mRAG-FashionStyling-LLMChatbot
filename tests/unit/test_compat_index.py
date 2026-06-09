"""CompatPairingIndex: complement retrieval + slot constraint. Loads the trained
compat_emb artifact (numpy, no model)."""
import pytest

from src.backend.core.config import settings
from src.backend.services.compat_index import CompatPairingIndex

ANCHOR = "0549330005"  # Dark Blue Jacket (outerwear), cold / not in co-buy graph


@pytest.fixture(scope="module")
def index():
    idx = CompatPairingIndex(settings.compat_dir)
    if not idx.ready:
        pytest.skip("compat artifacts not built (data/processed/compat)")
    return idx


def test_complement_excludes_anchor_and_returns_ids(index):
    out = index.complement_ids(ANCHOR, 5)
    assert 0 < len(out) <= 5
    assert ANCHOR not in out


def test_target_slot_filters_to_slot(index):
    out = index.complement_ids(ANCHOR, 8, target_slot="shoe")
    assert out, "expected shoe complements for the jacket"
    for aid in out:
        assert index.slots[index.aid_to_idx[aid]] == "shoe"


def test_unknown_anchor_returns_empty(index):
    assert index.complement_ids("0000000000", 5) == []
