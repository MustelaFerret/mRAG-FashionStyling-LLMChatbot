from __future__ import annotations

import gc
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.scripts.graph.build_graph import BuilderConfig, OutfitGraphBuilder

log = logging.getLogger(__name__)


UNWANTED_GROUPS = {"Under-, Nightwear"}
ALLOWED_SAME_GROUPS_LEGACY = {"Accessories", "Swimwear"}

NEUTRALS = {"Black", "White", "Off White", "Grey", "Light Grey", "Dark Grey", "Beige", "Light Beige", "Greyish Beige", "Silver", "Gold", "Transparent"}
NAVY_BLUES = {"Dark Blue", "Navy"}
BLUES = {"Blue", "Light Blue", "Other Blue", "Turquoise", "Light Turquoise", "Dark Turquoise"}
GREENS = {"Green", "Light Green", "Dark Green", "Greenish Khaki", "Other Green", "Olive"}
PINKS_REDS = {"Red", "Light Red", "Dark Red", "Pink", "Light Pink", "Dark Pink", "Other Red", "Other Pink", "Burgundy"}
YELLOWS_ORANGES = {"Orange", "Light Orange", "Dark Orange", "Yellow", "Light Yellow", "Dark Yellow", "Other Yellow", "Other Orange", "Bronze/Copper"}
BROWNS = {"Brown", "Dark Brown", "Yellowish Brown"}

SPORT_SECTIONS = {"Ladies H&M Sport", "Men H&M Sport", "Kids Sports"}
FORMAL_SECTIONS = {"Womens Tailoring", "Men Suits & Tailoring", "Contemporary Smart"}


@dataclass
class RuleAuditConfig:
    transactions_file: str
    meta_file: str
    output_md: str
    min_cooc: int = 3
    subsample_size: int = 2_000_000
    sample_size: int = 8
    seed: int = 42


class RuleAuditor:
    RULE_NAMES = [
        "rule_0_unwanted_groups",
        "rule_1_same_index",
        "rule_2_diff_pt",
        "rule_3_diff_pg_and_gg",
        "rule_5_diff_pcode",
        "rule_6_color_compat",
        "rule_7_season_compat",
        "rule_8_section_compat",
    ]

    def __init__(self, config: RuleAuditConfig):
        self.config = config
        self.pair_df: pd.DataFrame | None = None
        self.total_candidate_pairs: int = 0

    def prepare(self) -> None:
        builder_cfg = BuilderConfig(
            transactions_file=self.config.transactions_file,
            meta_file=self.config.meta_file,
            output_file="",
            method="cobuy",
            min_cooc=self.config.min_cooc,
            train_cutoff_date=None,
            content_filter="none",
        )
        builder = OutfitGraphBuilder(builder_cfg)
        builder.load_meta()
        builder.build_baskets()
        builder.compute_pair_counts()

        pairs_iter = (
            (a, b, count)
            for (a, b), count in builder.pair_counts.items()
            if count >= self.config.min_cooc
        )
        pair_df = pd.DataFrame(pairs_iter, columns=["item_a", "item_b", "count"])
        self.total_candidate_pairs = len(pair_df)
        log.info("total candidate pairs after min_cooc>=%d: %d", self.config.min_cooc, self.total_candidate_pairs)

        if self.total_candidate_pairs > self.config.subsample_size:
            pair_df = pair_df.sample(n=self.config.subsample_size, random_state=self.config.seed).reset_index(drop=True)
            log.info("subsampled to %d", len(pair_df))

        meta_cols = [
            "product_type_name",
            "product_group_name",
            "garment_group_name",
            "index_name",
            "section_name",
            "colour_group_name",
            "seasonality",
        ]
        meta = builder.meta_df[meta_cols].copy()
        pair_df = pair_df.merge(meta.add_suffix("_a"), left_on="item_a", right_index=True, how="left")
        pair_df = pair_df.merge(meta.add_suffix("_b"), left_on="item_b", right_index=True, how="left")
        pair_df = pair_df.fillna("")
        self.pair_df = pair_df
        log.info("pair_df ready: %d rows, %d cols", len(pair_df), len(pair_df.columns))

        del builder
        gc.collect()

    def rule_0_unwanted_groups(self, df: pd.DataFrame) -> pd.Series:
        return ~df["garment_group_name_a"].isin(UNWANTED_GROUPS) & ~df["garment_group_name_b"].isin(UNWANTED_GROUPS)

    def rule_1_same_index(self, df: pd.DataFrame) -> pd.Series:
        return df["index_name_a"] == df["index_name_b"]

    def rule_2_diff_pt(self, df: pd.DataFrame) -> pd.Series:
        same_pt = df["product_type_name_a"] == df["product_type_name_b"]
        allowed_same = df["product_group_name_a"].isin(ALLOWED_SAME_GROUPS_LEGACY)
        return ~same_pt | allowed_same

    def rule_3_diff_pg_and_gg(self, df: pd.DataFrame) -> pd.Series:
        same_pg = df["product_group_name_a"] == df["product_group_name_b"]
        same_gg = df["garment_group_name_a"] == df["garment_group_name_b"]
        allowed_same = df["product_group_name_a"].isin(ALLOWED_SAME_GROUPS_LEGACY)
        return ~(same_pg & same_gg) | allowed_same

    def rule_5_diff_pcode(self, df: pd.DataFrame) -> pd.Series:
        return df["item_a"].str[:6] != df["item_b"].str[:6]

    def rule_6_color_compat(self, df: pd.DataFrame) -> pd.Series:
        ca, cb = df["colour_group_name_a"], df["colour_group_name_b"]
        ga, gb = df["garment_group_name_a"], df["garment_group_name_b"]
        same = ca == cb
        either_neutral = ca.isin(NEUTRALS) | cb.isin(NEUTRALS)
        either_navy = ca.isin(NAVY_BLUES) | cb.isin(NAVY_BLUES)
        either_denim = (ga == "Trousers Denim") | (gb == "Trousers Denim")
        tonal_blue = ca.isin(BLUES) & cb.isin(BLUES)
        tonal_green = ca.isin(GREENS) & cb.isin(GREENS)
        tonal_redpink = ca.isin(PINKS_REDS) & cb.isin(PINKS_REDS)
        tonal_warm = ca.isin(YELLOWS_ORANGES) & cb.isin(YELLOWS_ORANGES)
        tonal_brown = ca.isin(BROWNS) & cb.isin(BROWNS)
        earth_neighbours = GREENS | YELLOWS_ORANGES
        cross_earth = (ca.isin(BROWNS) & cb.isin(earth_neighbours)) | (cb.isin(BROWNS) & ca.isin(earth_neighbours))
        cross_blue_pink = (ca.isin(BLUES) & cb.isin(PINKS_REDS)) | (cb.isin(BLUES) & ca.isin(PINKS_REDS))
        return same | either_neutral | either_navy | either_denim | tonal_blue | tonal_green | tonal_redpink | tonal_warm | tonal_brown | cross_earth | cross_blue_pink

    def rule_7_season_compat(self, df: pd.DataFrame) -> pd.Series:
        sa = df["seasonality_a"].astype(str).str.lower()
        sb = df["seasonality_b"].astype(str).str.lower()
        a_winter = sa.str.contains("winter|fall|snow|cold|chill", na=False)
        a_summer = sa.str.contains("summer|beach|heat|hot", na=False)
        b_winter = sb.str.contains("winter|fall|snow|cold|chill", na=False)
        b_summer = sb.str.contains("summer|beach|heat|hot", na=False)
        a_all = sa.str.contains("all-season|all-year|any season|transition", na=False)
        b_all = sb.str.contains("all-season|all-year|any season|transition", na=False)
        conflict_ws = a_winter & b_summer & ~a_all & ~b_all
        conflict_sw = a_summer & b_winter & ~a_all & ~b_all
        return ~(conflict_ws | conflict_sw)

    def rule_8_section_compat(self, df: pd.DataFrame) -> pd.Series:
        sa = df["section_name_a"]
        sb = df["section_name_b"]
        a_sport = sa.isin(SPORT_SECTIONS)
        b_sport = sb.isin(SPORT_SECTIONS)
        a_formal = sa.isin(FORMAL_SECTIONS)
        b_formal = sb.isin(FORMAL_SECTIONS)
        return ~((a_sport & b_formal) | (a_formal & b_sport))

    def measure(self, mask: pd.Series) -> Dict:
        kept = self.pair_df[mask]
        n = len(kept)
        if n == 0:
            return {"edges_kept": 0, "fraction_kept": 0.0, "intra_product_type": 0.0,
                    "intra_product_group": 0.0, "intra_garment_group": 0.0, "intra_color": 0.0}
        return {
            "edges_kept": int(n),
            "fraction_kept": float(mask.mean()),
            "intra_product_type": float((kept["product_type_name_a"] == kept["product_type_name_b"]).mean()),
            "intra_product_group": float((kept["product_group_name_a"] == kept["product_group_name_b"]).mean()),
            "intra_garment_group": float((kept["garment_group_name_a"] == kept["garment_group_name_b"]).mean()),
            "intra_color": float((kept["colour_group_name_a"] == kept["colour_group_name_b"]).mean()),
        }

    def _rule_mask(self, rule_name: str) -> pd.Series:
        return getattr(self, rule_name)(self.pair_df)

    def ablation_individual(self) -> Dict:
        baseline = self.measure(pd.Series(True, index=self.pair_df.index))
        results = {"baseline_no_rules": baseline}
        for rn in self.RULE_NAMES:
            mask = self._rule_mask(rn)
            m = self.measure(mask)
            m["dropped"] = int((~mask).sum())
            m["dropped_pct"] = float((~mask).mean()) * 100
            results[rn] = m
        return results

    def ablation_cumulative(self) -> List[Dict]:
        mask = pd.Series(True, index=self.pair_df.index)
        results = [{"after": "baseline (no rules)", **self.measure(mask)}]
        for rn in self.RULE_NAMES:
            mask = mask & self._rule_mask(rn)
            results.append({"after": f"+ {rn}", **self.measure(mask)})
        return results

    def ablation_leave_one_out(self) -> Dict:
        rule_masks = {rn: self._rule_mask(rn) for rn in self.RULE_NAMES}
        full_mask = pd.Series(True, index=self.pair_df.index)
        for rn in self.RULE_NAMES:
            full_mask = full_mask & rule_masks[rn]
        full_metrics = self.measure(full_mask)
        results = {"all_rules": full_metrics}
        for skip in self.RULE_NAMES:
            loo_mask = pd.Series(True, index=self.pair_df.index)
            for rn in self.RULE_NAMES:
                if rn == skip:
                    continue
                loo_mask = loo_mask & rule_masks[rn]
            m = self.measure(loo_mask)
            m["added_vs_full"] = m["edges_kept"] - full_metrics["edges_kept"]
            m["delta_intra_pt"] = m["intra_product_type"] - full_metrics["intra_product_type"]
            m["delta_intra_color"] = m["intra_color"] - full_metrics["intra_color"]
            results[f"all_minus_{skip}"] = m
        return results

    def _sample_rows(self, rows: pd.DataFrame, n: int) -> pd.DataFrame:
        if len(rows) == 0:
            return rows
        return rows.sample(min(n, len(rows)), random_state=self.config.seed)

    def sample_dropped_kept(self, rule_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        mask = self._rule_mask(rule_name)
        return (
            self._sample_rows(self.pair_df[~mask], self.config.sample_size),
            self._sample_rows(self.pair_df[mask], self.config.sample_size),
        )

    @staticmethod
    def _format_pair_row(row: pd.Series) -> str:
        return (
            f"- `#{row['item_a']}` ({row['product_type_name_a']} / {row['colour_group_name_a']} / {row['seasonality_a']}) ↔ "
            f"`#{row['item_b']}` ({row['product_type_name_b']} / {row['colour_group_name_b']} / {row['seasonality_b']}) "
            f"| count={int(row['count'])}"
        )

    def generate_report(self) -> str:
        individual = self.ablation_individual()
        cumulative = self.ablation_cumulative()
        loo = self.ablation_leave_one_out()

        lines: List[str] = []
        lines.append("# Rule Audit Report — Outfit Graph Filter")
        lines.append("")
        lines.append(f"**Total candidate pairs (min_cooc≥{self.config.min_cooc})**: {self.total_candidate_pairs:,}")
        lines.append(f"**Sub-sampled for audit**: {len(self.pair_df):,} (seed={self.config.seed})")
        lines.append("")
        lines.append("## 1. Individual rule impact")
        lines.append("")
        lines.append("Mỗi rule áp dụng độc lập trên toàn bộ pair candidates. Drop % = pairs bị rule đó loại.")
        lines.append("`intra_*` = trong pairs kept (sau rule đó), % cùng product_type / product_group / colour_group.")
        lines.append("")
        lines.append("| rule | dropped | drop_% | intra_PT | intra_PG | intra_GG | intra_color |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        base = individual["baseline_no_rules"]
        lines.append(f"| baseline (no rules) | 0 | 0.0% | {base['intra_product_type']:.4f} | {base['intra_product_group']:.4f} | {base['intra_garment_group']:.4f} | {base['intra_color']:.4f} |")
        for rn in self.RULE_NAMES:
            r = individual[rn]
            lines.append(
                f"| {rn} | {r['dropped']:,} | {r['dropped_pct']:.1f}% | "
                f"{r['intra_product_type']:.4f} | {r['intra_product_group']:.4f} | "
                f"{r['intra_garment_group']:.4f} | {r['intra_color']:.4f} |"
            )
        lines.append("")
        lines.append("## 2. Cumulative ablation (add rules one-by-one in legacy order)")
        lines.append("")
        lines.append("| step | edges_kept | %kept | intra_PT | intra_PG | intra_GG | intra_color |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for r in cumulative:
            lines.append(
                f"| {r['after']} | {r['edges_kept']:,} | {r['fraction_kept']:.4f} | "
                f"{r.get('intra_product_type', 0):.4f} | {r.get('intra_product_group', 0):.4f} | "
                f"{r.get('intra_garment_group', 0):.4f} | {r.get('intra_color', 0):.4f} |"
            )
        lines.append("")
        lines.append("## 3. Leave-one-out (full ruleset minus 1 rule)")
        lines.append("")
        lines.append("Đo edges added khi BỎ rule đó khỏi full set. Số càng cao → rule đó càng quan trọng (loại nhiều pairs unique).")
        lines.append("Δ intra_PT, Δ intra_color: thay đổi tỷ lệ nội tại khi remove rule (số dương = removal làm tệ hơn).")
        lines.append("")
        full = loo["all_rules"]
        lines.append(f"**Full ruleset baseline**: kept={full['edges_kept']:,}, intra_PT={full['intra_product_type']:.4f}, intra_color={full['intra_color']:.4f}")
        lines.append("")
        lines.append("| removed | edges_added | Δ intra_PT | Δ intra_color |")
        lines.append("|---|---:|---:|---:|")
        for rn in self.RULE_NAMES:
            r = loo[f"all_minus_{rn}"]
            lines.append(f"| {rn} | +{r['added_vs_full']:,} | {r['delta_intra_pt']:+.4f} | {r['delta_intra_color']:+.4f} |")
        lines.append("")
        lines.append("## 4. Sample pairs per rule (qualitative inspection)")
        lines.append("")
        lines.append("Mỗi rule: 8 pairs random rule loại bỏ + 8 pairs random rule giữ lại.")
        lines.append("Format: `item_a (PT_a / color_a / season_a) ↔ item_b (PT_b / color_b / season_b) | count=N`")
        lines.append("")
        for rn in self.RULE_NAMES:
            dropped, kept = self.sample_dropped_kept(rn)
            lines.append(f"### {rn}")
            lines.append("")
            lines.append("**Dropped samples** (rule loại):")
            lines.append("")
            for _, row in dropped.iterrows():
                lines.append(self._format_pair_row(row))
            lines.append("")
            lines.append("**Kept samples** (rule giữ):")
            lines.append("")
            for _, row in kept.iterrows():
                lines.append(self._format_pair_row(row))
            lines.append("")
        return "\n".join(lines)

    def write_report(self) -> None:
        report = self.generate_report()
        Path(self.config.output_md).parent.mkdir(parents=True, exist_ok=True)
        with open(self.config.output_md, "w", encoding="utf-8") as f:
            f.write(report)
        log.info("wrote report: %s", self.config.output_md)


def _config_from_env() -> RuleAuditConfig:
    base = Path(__file__).resolve().parent.parent.parent
    return RuleAuditConfig(
        transactions_file=str(base / os.getenv("TRANS_FILE", "data/raw/transactions_train.csv")),
        meta_file=str(base / os.getenv("META_FILE", "data/processed/dataset_qwen_completed.csv")),
        output_md=str(base / os.getenv("OUTPUT_MD", "md/step_3_rule_audit.md")),
        min_cooc=int(os.getenv("MIN_COOC", "3")),
        subsample_size=int(os.getenv("SUBSAMPLE_SIZE", "2000000")),
        sample_size=int(os.getenv("SAMPLE_SIZE", "8")),
        seed=int(os.getenv("SEED", "42")),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    config = _config_from_env()
    auditor = RuleAuditor(config)
    auditor.prepare()
    auditor.write_report()


if __name__ == "__main__":
    main()
