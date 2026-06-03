from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List

from src.scripts.graph.build_graph import BuilderConfig, OutfitGraphBuilder
from src.scripts.graph.eval_graph import (
    AdjacencyMap,
    EvalConfig,
    EvaluationResult,
    OutfitGraphEvaluator,
    OutfitGraphLoader,
)
from src.scripts.graph.profile_graph import GraphProfile, GraphProfiler

log = logging.getLogger(__name__)


@dataclass
class VariantSpec:
    name: str
    method: str
    min_cooc: int = 3
    npmi_threshold: float = 0.0
    content_filter: str = "minimal"
    use_train_cutoff: bool = True


@dataclass
class VariantOutcome:
    spec: VariantSpec
    graph_path: str
    profile: GraphProfile
    eval_result_all: EvaluationResult
    eval_result_cross_pt: EvaluationResult


class GraphVariantExperiment:
    def __init__(
        self,
        eval_config: EvalConfig,
        base_builder_config: BuilderConfig,
        output_dir: str,
        train_cutoff_date: str,
    ):
        self.eval_config = eval_config
        self.base_builder_config = base_builder_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.train_cutoff_date = train_cutoff_date
        self.variants: List[VariantSpec] = []
        self.evaluator_all: OutfitGraphEvaluator | None = None
        self.evaluator_cross: OutfitGraphEvaluator | None = None
        self.profiler: GraphProfiler | None = None

    def add_variant(self, spec: VariantSpec) -> None:
        self.variants.append(spec)

    def prepare(self) -> None:
        from dataclasses import replace as dc_replace
        self.evaluator_all = OutfitGraphEvaluator(dc_replace(self.eval_config, ground_truth_filter="all"))
        self.evaluator_all.prepare()
        self.evaluator_cross = OutfitGraphEvaluator(dc_replace(self.eval_config, ground_truth_filter="cross_product_type"))
        self.evaluator_cross._train_baskets = self.evaluator_all._train_baskets
        self.evaluator_cross._test_baskets = self.evaluator_all._test_baskets
        self.evaluator_cross._cutoff_date = self.evaluator_all._cutoff_date
        self.evaluator_cross._valid_ids = self.evaluator_all._valid_ids
        self.evaluator_cross.load_product_types(self.base_builder_config.meta_file)
        self.profiler = GraphProfiler(self.base_builder_config.meta_file)

    def _builder_config_for(self, spec: VariantSpec) -> BuilderConfig:
        graph_path = self.output_dir / f"{spec.name}.csv"
        return replace(
            self.base_builder_config,
            method=spec.method,
            min_cooc=spec.min_cooc,
            npmi_threshold=spec.npmi_threshold,
            content_filter=spec.content_filter,
            train_cutoff_date=self.train_cutoff_date if spec.use_train_cutoff else None,
            output_file=str(graph_path),
        )

    def run_variant(self, spec: VariantSpec) -> VariantOutcome:
        builder_config = self._builder_config_for(spec)
        graph_path = builder_config.output_file
        if os.path.exists(graph_path):
            log.info("variant %s already built at %s, skipping build", spec.name, graph_path)
        else:
            log.info("building variant: %s", spec.name)
            builder = OutfitGraphBuilder(builder_config)
            builder.run()

        adjacency = OutfitGraphLoader.from_csv(graph_path)
        profile = self.profiler.profile(adjacency)
        result_all = self.evaluator_all.evaluate(adjacency, name=spec.name)
        result_cross = self.evaluator_cross.evaluate(adjacency, name=spec.name)
        return VariantOutcome(
            spec=spec,
            graph_path=graph_path,
            profile=profile,
            eval_result_all=result_all,
            eval_result_cross_pt=result_cross,
        )

    def run_all(self, external_baselines: Dict[str, str] | None = None) -> List[VariantOutcome]:
        outcomes: List[VariantOutcome] = []
        for name, path in (external_baselines or {}).items():
            adjacency = OutfitGraphLoader.from_csv(path)
            profile = self.profiler.profile(adjacency)
            spec = VariantSpec(name=name, method="external")
            result_all = self.evaluator_all.evaluate(adjacency, name=name)
            result_cross = self.evaluator_cross.evaluate(adjacency, name=name)
            outcomes.append(VariantOutcome(
                spec=spec,
                graph_path=path,
                profile=profile,
                eval_result_all=result_all,
                eval_result_cross_pt=result_cross,
            ))
        for spec in self.variants:
            outcomes.append(self.run_variant(spec))
        return outcomes

    @staticmethod
    def _format_eval_table(title: str, outcomes: List[VariantOutcome], attr: str) -> List[str]:
        results = [getattr(o, attr) for o in outcomes]
        k_values = sorted(results[0].recall_at_k.keys())
        lines: List[str] = []
        lines.append(f"### {title}")
        header_cells = ["variant", "nodes", "edges", "coverage"]
        header_cells += [f"recall@{k}" for k in k_values]
        header_cells += [f"map@{k}" for k in k_values]
        header_cells += [f"hit@{k}" for k in k_values]
        lines.append("| " + " | ".join(header_cells) + " |")
        lines.append("|" + "|".join(["---"] * len(header_cells)) + "|")
        for outcome, r in zip(outcomes, results):
            p = outcome.profile
            cells = [
                outcome.spec.name,
                str(p.num_nodes),
                str(p.num_unique_undirected_edges),
                f"{r.coverage:.4f}",
            ]
            cells += [f"{r.recall_at_k[k]:.4f}" for k in k_values]
            cells += [f"{r.map_at_k[k]:.4f}" for k in k_values]
            cells += [f"{r.hit_at_k[k]:.4f}" for k in k_values]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        return lines

    @staticmethod
    def format_combined_report(outcomes: List[VariantOutcome]) -> str:
        if not outcomes:
            return ""
        lines: List[str] = []
        lines.append("## Graph variants comparison")
        lines.append("")
        lines.extend(GraphVariantExperiment._format_eval_table(
            "Eval mode = all (any held-out item counts)", outcomes, "eval_result_all"
        ))
        lines.extend(GraphVariantExperiment._format_eval_table(
            "Eval mode = cross_product_type (outfit pairing only)", outcomes, "eval_result_cross_pt"
        ))
        lines.append("### Pairing nature")
        lines.append("| variant | intra_product_type | intra_product_group | intra_color |")
        lines.append("|---|---|---|---|")
        for outcome in outcomes:
            p = outcome.profile
            lines.append(
                f"| {outcome.spec.name} | {p.intra_product_type_ratio:.4f} "
                f"| {p.intra_product_group_ratio:.4f} | {p.intra_color_ratio:.4f} |"
            )
        return "\n".join(lines)


def _make_default_experiment() -> GraphVariantExperiment:
    base = Path(__file__).resolve().parent.parent.parent
    eval_config = EvalConfig(
        transactions_file=str(base / "data/raw/transactions_train.csv"),
        meta_file=str(base / "data/processed/dataset_final_qwen_filled.csv"),
        cache_dir=str(base / "data/processed/eval_cache"),
    )
    base_builder = BuilderConfig(
        transactions_file=str(base / "data/raw/transactions_train.csv"),
        meta_file=str(base / "data/processed/dataset_final_qwen_filled.csv"),
        output_file="",
        basket_cache_pickle=str(base / "data/processed/eval_cache/eval_baskets_98e4905692c809f6.pkl"),
    )
    return GraphVariantExperiment(
        eval_config=eval_config,
        base_builder_config=base_builder,
        output_dir=str(base / "data/processed/graph_experiments"),
        train_cutoff_date="2020-04-28",
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    experiment = _make_default_experiment()

    experiment.add_variant(VariantSpec(name="cobuy_clean_minimal", method="cobuy", content_filter="minimal"))
    experiment.add_variant(VariantSpec(name="cobuy_clean_standard", method="cobuy", content_filter="standard"))
    experiment.add_variant(VariantSpec(name="cobuy_clean_legacy", method="cobuy", content_filter="legacy"))
    experiment.add_variant(VariantSpec(name="npmi_t0.0_minimal", method="npmi", npmi_threshold=0.0, content_filter="minimal"))
    experiment.add_variant(VariantSpec(name="npmi_t0.1_minimal", method="npmi", npmi_threshold=0.1, content_filter="minimal"))
    experiment.add_variant(VariantSpec(name="npmi_t0.2_minimal", method="npmi", npmi_threshold=0.2, content_filter="minimal"))
    experiment.add_variant(VariantSpec(name="npmi_t0.1_standard", method="npmi", npmi_threshold=0.1, content_filter="standard"))
    experiment.add_variant(VariantSpec(name="npmi_t0.1_legacy", method="npmi", npmi_threshold=0.1, content_filter="legacy"))

    experiment.prepare()
    base = Path(__file__).resolve().parent.parent.parent
    external = {
        "baseline_original_with_leak": str(base / "data/processed/final_outfit_graph.csv"),
    }
    outcomes = experiment.run_all(external_baselines=external)

    report = GraphVariantExperiment.format_combined_report(outcomes)
    print(report)

    report_path = base / "markdown/graph_rebuild_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(report)
    log.info("wrote report: %s", report_path)


if __name__ == "__main__":
    main()
