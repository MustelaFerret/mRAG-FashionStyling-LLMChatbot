"""Kiểm tra distribution của Unknown PT items theo các columns fallback candidates.

Mục tiêu: xác định layer-3 fallback đáng tin nhất cho 111 items có PT=Unknown.
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path
from collections import Counter


class UnknownPTInspector:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path, low_memory=False)
        self.unknown = self.df[self.df["product_type_name"] == "Unknown"].copy()

    def overview(self):
        n_total = len(self.df)
        n_unknown = len(self.unknown)
        print(f"Tong items: {n_total:,}")
        print(f"Items co PT=Unknown: {n_unknown:,} ({n_unknown/n_total*100:.2f}%)")
        print()

    def distribution(self, col: str):
        counts = self.unknown[col].value_counts(dropna=False)
        print(f"=== Distribution cua PT=Unknown items theo `{col}` ({counts.size} unique) ===")
        for val, cnt in counts.items():
            print(f"  {cnt:4d}  {val!r}")
        print()
        return counts

    def cross_tab(self, col_a: str, col_b: str):
        ct = pd.crosstab(self.unknown[col_a].fillna("(nan)"), self.unknown[col_b].fillna("(nan)"))
        print(f"=== Cross-tab PT=Unknown: {col_a} x {col_b} ===")
        print(ct.to_string())
        print()

    def sample_by_garment_group(self, n_per_group: int = 5):
        print("=== Sample items theo garment_group_name (10 mau / nhom) ===")
        for gg, sub in self.unknown.groupby("garment_group_name"):
            print(f"\n--- garment_group_name = {gg!r} (total {len(sub)}) ---")
            cols = ["article_id", "prod_name", "product_group_name", "section_name", "department_name", "colour_group_name"]
            available = [c for c in cols if c in sub.columns]
            print(sub[available].head(n_per_group).to_string(index=False))
        print()

    def all_unknown_check(self):
        """Dem so item co PT=Unknown AND PG=Unknown AND GG=Unknown."""
        mask = (
            (self.unknown["product_type_name"] == "Unknown")
            & (self.unknown["product_group_name"] == "Unknown")
            & (self.unknown["garment_group_name"] == "Unknown")
        )
        n_all_unknown = mask.sum()
        print(f"Items voi CA 3 cot (PT, PG, GG) = Unknown: {n_all_unknown}")
        if n_all_unknown > 0:
            cols = ["article_id", "prod_name", "section_name", "department_name", "index_name", "index_group_name"]
            available = [c for c in cols if c in self.unknown.columns]
            print("Sample 20 items truong hop nay (xem section/department con cuu duoc khong):")
            print(self.unknown[mask][available].head(20).to_string(index=False))
        print()


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[2]
    csv = repo / "data" / "processed" / "dataset_qwen_completed.csv"
    insp = UnknownPTInspector(str(csv))
    insp.overview()
    insp.distribution("product_group_name")
    insp.distribution("garment_group_name")
    insp.distribution("section_name")
    insp.distribution("index_name")
    insp.cross_tab("product_group_name", "garment_group_name")
    insp.all_unknown_check()
    insp.sample_by_garment_group(n_per_group=8)
