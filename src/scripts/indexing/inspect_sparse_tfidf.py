"""Inspect sparse TF-IDF structure + sample vocab."""
from __future__ import annotations

import json
from pathlib import Path


def main():
    repo = Path(__file__).resolve().parents[2]
    path = repo / "data" / "processed" / "sparse_tfidf.json"
    with open(path) as f:
        d = json.load(f)

    print(f"Top-level keys: {list(d.keys())}")
    for k, v in d.items():
        if k == "vocab":
            print(f"  vocab: dict len={len(v)}")
        elif isinstance(v, list):
            print(f"  {k}: list len={len(v)}, sample={v[:5]}")
        elif isinstance(v, dict):
            print(f"  {k}: dict len={len(v)}, keys sample={list(v.keys())[:5]}")
        else:
            print(f"  {k}: {v}")

    vocab = d.get("vocab", {})
    print(f"\nVocab size: {len(vocab)}")
    print("First 40 vocab items (sorted by index):")
    for k, v in sorted(vocab.items(), key=lambda x: x[1])[:40]:
        print(f"  {v:5d}: {k!r}")

    print("\nLast 20 vocab items:")
    for k, v in sorted(vocab.items(), key=lambda x: x[1])[-20:]:
        print(f"  {v:5d}: {k!r}")

    print("\nKeyword search trong vocab:")
    for keyword in ["dress", "blue", "denim", "elegant", "casual", "summer", "lace", "floral", "pattern", "vibe"]:
        matches = [(k, v) for k, v in vocab.items() if keyword in k]
        print(f"  {keyword!r}: {matches[:10]}")


if __name__ == "__main__":
    main()
