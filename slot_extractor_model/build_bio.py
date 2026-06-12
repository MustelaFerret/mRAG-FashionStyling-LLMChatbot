"""Turn validated (query + per-field surface span) rows into word-level BIO sequences for
token classification, and split train/val. The span substring is located in the word list
(consistent regex tokenisation) and tagged B-/I-<FIELD>. Canonical values are carried along
for the linker/eval (the tagger only predicts the span TYPE; normalisation is a separate
stage). Output: jsonl with {tokens, tags, query, *_canon}.

Run: python -m slot_extractor_model.build_bio
"""
from __future__ import annotations

import json
import random
import re

import pandas as pd

from slot_extractor_model import config as C

AB = {"colour_group": "COL", "fit": "FIT", "occasion": "OCC", "seasonality": "SEA"}
LABELS = ["O"] + [f"{p}-{a}" for a in AB.values() for p in ("B", "I")]
_TOK = re.compile(r"[A-Za-z0-9/'-]+")
# function words stripped from span ENDS so the label is the minimal head word(s):
# "for a wedding" -> "wedding", "to the gym" -> "gym". Consistent boundaries lift seqeval F1
# (esp. occasion, which the LLM phrases as prepositional phrases).
_EDGE = {"for", "to", "in", "on", "at", "a", "an", "the", "with", "my", "this", "that", "these",
         "those", "some", "of", "and", "is", "are", "i", "want", "need", "looking", "please",
         "something", "anything", "perfect", "good", "ideal", "suitable", "wear", "use", "vibes",
         "style", "made", "great"}


def _words(s: str):
    return _TOK.findall(str(s))


def _trim(toks):
    lo, hi = 0, len(toks)
    while lo < hi and toks[lo].lower() in _EDGE:
        lo += 1
    while hi > lo and toks[hi - 1].lower() in _EDGE:
        hi -= 1
    return toks[lo:hi]


def _find_span(words_lc, span_lc):
    n = len(span_lc)
    for i in range(len(words_lc) - n + 1):
        if words_lc[i:i + n] == span_lc:
            return i
    return None


def _row_to_bio(row):
    words = _words(row["query"])
    wl = [w.lower() for w in words]
    tags = ["O"] * len(words)
    for field, ab in AB.items():
        if row.get(f"{field}_canon", "none") == "none":
            continue
        span = _trim(_words(str(row.get(f"{field}_surface", ""))))
        if not span:
            return None  # surface was all function words -> drop row (keeps labels clean)
        idx = _find_span(wl, [w.lower() for w in span])
        if idx is None:
            return None  # span not locatable at word level -> skip row
        tags[idx] = f"B-{ab}"
        for j in range(idx + 1, idx + len(span)):
            tags[j] = f"I-{ab}"
    return {
        "query": row["query"], "tokens": words, "tags": tags,
        **{f"{f}_canon": row.get(f"{f}_canon", "none") for f in C.FIELDS},
    }


def _dump(rows, path):
    with open(C.DATA_DIR / path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _convert(csv_names):
    rows = []
    for name in csv_names:
        p = C.DATA_DIR / name
        if not p.exists():
            continue
        df = pd.read_csv(p).fillna("")
        rows += [r for _, r in df.iterrows()]
    out, seen = [], set()
    for r in rows:
        q = str(r["query"]).strip()
        if q.lower() in seen:  # dedup identical queries
            continue
        b = _row_to_bio(r)
        if b:
            out.append(b); seen.add(q.lower())
    print(f"  {csv_names}: {len(rows)} rows -> {len(out)} unique BIO seqs")
    return out


def main() -> None:
    train = _convert(["train_clean.csv", "train_clean2.csv"])
    test = _convert(["test_clean.csv", "test_clean2.csv"])
    # leakage guard: drop any train query that also appears in test
    test_q = {r["query"].lower() for r in test}
    before = len(train)
    train = [r for r in train if r["query"].lower() not in test_q]
    if before != len(train):
        print(f"  leakage guard: removed {before - len(train)} train seqs also in test")
    rng = random.Random(0)
    rng.shuffle(train)
    n_val = max(1, int(0.12 * len(train)))
    val, tr = train[:n_val], train[n_val:]
    _dump(tr, "bio_train.jsonl"); _dump(val, "bio_val.jsonl"); _dump(test, "bio_test.jsonl")

    # label distribution sanity
    from collections import Counter
    cnt = Counter(t for r in tr for t in r["tags"])
    print(f"[bio] train {len(tr)} / val {len(val)} / test {len(test)}")
    print("  label counts (train):", dict(cnt))
    print("  LABELS:", LABELS)


if __name__ == "__main__":
    main()
