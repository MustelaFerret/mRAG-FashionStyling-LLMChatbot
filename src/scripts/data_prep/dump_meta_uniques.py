from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


class UniqueDumper:
    def __init__(self, meta_file: str, output_file: str, limit_per_column: int = 400):
        self.meta_file = meta_file
        self.output_file = output_file
        self.limit_per_column = limit_per_column

    def dump(self) -> None:
        df = pd.read_csv(self.meta_file, dtype=str, keep_default_na=False)
        lines = []
        lines.append(f"# Unique values per column — `{Path(self.meta_file).name}`")
        lines.append("")
        lines.append(f"Total rows: {len(df):,}. Columns truncated to {self.limit_per_column} unique values if more.")
        lines.append("")
        for col in df.columns:
            series = df[col].fillna("").astype(str)
            uniques = series[series != ""].unique()
            n = len(uniques)
            avg_len = float(series.str.len().mean()) if len(series) else 0.0
            lines.append(f"## {col} — {n} unique values, avg_len={avg_len:.0f}")
            lines.append("")
            if avg_len > 80:
                lines.append(f"_skipped (avg length {avg_len:.0f} chars; likely free text)._")
                lines.append("")
                continue
            sorted_unique = sorted(uniques)
            if n > self.limit_per_column:
                lines.append(f"_truncated to first {self.limit_per_column} (sorted)._")
                lines.append("")
                sorted_unique = sorted_unique[: self.limit_per_column]
            for v in sorted_unique:
                lines.append(f"- {v}")
            lines.append("")

        Path(self.output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        log.info("wrote uniques to %s", self.output_file)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    base = Path(__file__).resolve().parent.parent.parent
    meta_file = str(base / os.getenv("META_FILE", "data/processed/dataset_qwen_completed.csv"))
    output_file = str(base / os.getenv("OUTPUT_FILE", "md/unique.txt"))
    limit = int(os.getenv("LIMIT_PER_COLUMN", "400"))
    dumper = UniqueDumper(meta_file, output_file, limit)
    dumper.dump()


if __name__ == "__main__":
    main()
