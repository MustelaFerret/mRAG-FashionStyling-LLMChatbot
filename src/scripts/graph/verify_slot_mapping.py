"""Verify slot mapping mới trên toàn dataset.

Check:
- Tổng phân bố slot
- Riêng PT=Unknown items: recover được bao nhiêu, còn bao nhiêu rơi vào "other"
- PT=Costumes, Sarong, Garment Set: confirm slot đúng
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path
from collections import Counter

from src.scripts.graph.outfit_slots import get_slot, PT_TO_SLOT


def main():
    repo = Path(__file__).resolve().parents[2]
    csv = repo / "data" / "processed" / "dataset_qwen_completed.csv"
    df = pd.read_csv(csv, low_memory=False, dtype={"article_id": str})

    df["slot"] = df.apply(
        lambda r: get_slot(
            r.get("product_type_name", ""),
            r.get("product_group_name", ""),
            r.get("garment_group_name", ""),
            r.get("prod_name", ""),
            r.get("section_name", ""),
            r.get("department_name", ""),
        ),
        axis=1,
    )

    print("=== Slot distribution toan dataset ===")
    print(df["slot"].value_counts().to_string())
    print(f"Total: {len(df):,}\n")

    print("=== Other slot breakdown (PT distribution) ===")
    other = df[df["slot"] == "other"]
    print(f"Total in other: {len(other):,}")
    print(other["product_type_name"].value_counts().head(20).to_string())
    print()

    print("=== Unknown PT items recovery ===")
    unk = df[df["product_type_name"] == "Unknown"]
    print(f"Total Unknown PT items: {len(unk)}")
    print("Slot distribution sau fallback:")
    print(unk["slot"].value_counts().to_string())
    print()

    print("Sample Unknown PT items con roi vao 'other':")
    other_unk = unk[unk["slot"] == "other"]
    cols = ["article_id", "prod_name", "garment_group_name", "section_name"]
    available = [c for c in cols if c in unk.columns]
    print(other_unk[available].head(15).to_string(index=False))
    print()

    print("=== Cac PT da sua check (Sarong, Costumes, Garment Set) ===")
    for pt in ["Sarong", "Costumes", "Garment Set"]:
        sub = df[df["product_type_name"] == pt]
        slot_dist = sub["slot"].value_counts().to_dict()
        print(f"  {pt} ({len(sub)} items): {slot_dist}")
    print()

    print("=== Mixed GG items breakdown ===")
    for gg in ["Jersey Basic", "Jersey Fancy"]:
        sub = df[(df["product_type_name"] == "Unknown") & (df["garment_group_name"] == gg)]
        print(f"\n--- garment_group = {gg!r}, PT=Unknown ({len(sub)} items) ---")
        for _, row in sub.iterrows():
            print(f"  slot={row['slot']:10s}  {row['prod_name']!r}")


if __name__ == "__main__":
    main()
