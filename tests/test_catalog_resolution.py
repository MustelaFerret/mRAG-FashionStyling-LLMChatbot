"""Product-type canonicalization + slot inference (the logic that caused the
parka/shoe regressions). Model-free."""


def test_canonical_exact_and_synonym(catalog):
    assert catalog.canonical_product_type("jacket") == "Jacket"      # exact (case-insensitive)
    assert catalog.canonical_product_type("Trousers") == "Trousers"
    assert catalog.canonical_product_type("trouser") == "Trousers"   # synonym
    assert catalog.canonical_product_type("pants") == "Trousers"     # synonym
    assert catalog.canonical_product_type("parka") == ""             # OOV -> not exact/synonym
    assert catalog.canonical_product_type("") == ""


def test_corpus_majority_canonicalization(catalog):
    # parka is out-of-vocab but corpus labelling says Jacket (not Coat as word-embedding would).
    assert catalog.corpus_product_type("parka") == "Jacket"
    assert catalog.corpus_product_type("anorak") == "Jacket"
    # generic / unseen terms -> no confident majority
    assert catalog.corpus_product_type("xyzblahnonsense") == ""
    assert catalog.corpus_product_type("") == ""


def test_infer_slot_from_text(catalog):
    assert catalog.infer_slot_from_text("do you have a shoe to match") == "shoe"
    assert catalog.infer_slot_from_text("some trousers please") == "bottom"
    assert catalog.infer_slot_from_text("a warm jacket") == "outerwear"
    assert catalog.infer_slot_from_text("") == ""


def test_infer_slot_excludes_anchor_slot(catalog):
    # "the jacket one ... shoe to match" -> target must NOT be the anchor's slot (outerwear).
    q = "the jacket one looks good do you have shoe to match with that"
    assert catalog.infer_slot_from_text(q) == "shoe"
    assert catalog.infer_slot_from_text(q, exclude_slot="outerwear") == "shoe"
    # if the only mentioned slot is excluded, fall through to empty
    assert catalog.infer_slot_from_text("a warm jacket", exclude_slot="outerwear") == ""
