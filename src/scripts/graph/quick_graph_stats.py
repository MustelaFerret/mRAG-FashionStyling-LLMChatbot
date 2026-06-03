"""Quick stats so sanh graph moi vs cu."""
from __future__ import annotations

import pandas as pd
from pathlib import Path

from src.scripts.graph.outfit_slots import get_slot, PT_FAMILY


def analyze(graph_path: str, meta_path: str, label: str):
    g = pd.read_csv(graph_path, dtype={"item_a": str, "item_b": str})
    m = pd.read_csv(meta_path, dtype={"article_id": str},
                    usecols=["article_id", "product_type_name", "product_group_name",
                             "garment_group_name", "prod_name", "section_name", "department_name"]).fillna("")
    m["article_id"] = m["article_id"].str.zfill(10)
    m["slot"] = m.apply(lambda r: get_slot(r["product_type_name"], r["product_group_name"],
                                            r["garment_group_name"], r["prod_name"],
                                            r["section_name"], r["department_name"]), axis=1)
    meta_idx = m.set_index("article_id")

    g["item_a"] = g["item_a"].str.zfill(10)
    g["item_b"] = g["item_b"].str.zfill(10)
    # unique undirected edges
    g["pair"] = g.apply(lambda r: tuple(sorted([r["item_a"], r["item_b"]])), axis=1)
    uniq = g.drop_duplicates("pair")
    n_edges = len(uniq)
    n_nodes = len(set(g["item_a"]) | set(g["item_b"]))

    # join meta
    uniq = uniq.copy()
    uniq["pt_a"] = uniq["item_a"].map(meta_idx["product_type_name"])
    uniq["pt_b"] = uniq["item_b"].map(meta_idx["product_type_name"])
    uniq["pg_a"] = uniq["item_a"].map(meta_idx["product_group_name"])
    uniq["pg_b"] = uniq["item_b"].map(meta_idx["product_group_name"])
    uniq["slot_a"] = uniq["item_a"].map(meta_idx["slot"])
    uniq["slot_b"] = uniq["item_b"].map(meta_idx["slot"])

    intra_pt = (uniq["pt_a"] == uniq["pt_b"]).sum()
    intra_pg = (uniq["pg_a"] == uniq["pg_b"]).sum()

    # family duplicate check
    def same_fam(a, b):
        fa, fb = PT_FAMILY.get(a), PT_FAMILY.get(b)
        return fa is not None and fa == fb
    family_dups = uniq.apply(lambda r: same_fam(r["pt_a"], r["pt_b"]), axis=1).sum()

    # slot pair distribution
    uniq["slot_pair"] = uniq.apply(lambda r: " + ".join(sorted([r["slot_a"], r["slot_b"]])), axis=1)
    slot_dist = uniq["slot_pair"].value_counts()

    print(f"=== {label} ===")
    print(f"Nodes:           {n_nodes:,}")
    print(f"Unique edges:    {n_edges:,}")
    print(f"Intra-PT edges:  {intra_pt:,} ({intra_pt/n_edges*100:.2f}%)")
    print(f"Intra-PG edges:  {intra_pg:,} ({intra_pg/n_edges*100:.2f}%)")
    print(f"Family-dup edges: {family_dups:,} ({family_dups/n_edges*100:.4f}%)")
    print(f"\nSlot pair distribution (top 15):")
    print(slot_dist.head(15).to_string())
    print()


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[2]
    meta = str(repo / "data" / "processed" / "dataset_qwen_completed.csv")

    analyze(str(repo / "data" / "processed" / "graph_archive" / "final_outfit_graph_cobuy_full_redesigned_step3.csv"),
            meta, "Step 3 v1 (truoc family check)")
    analyze(str(repo / "data" / "processed" / "final_outfit_graph.csv"),
            meta, "Step 3 v2 (sau family check + slot mapping fixes)")
