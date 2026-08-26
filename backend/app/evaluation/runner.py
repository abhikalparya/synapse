"""CLI entrypoint for Synapse evaluation benchmarks."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

_BACKEND_DIR = Path(__file__).resolve().parents[2]
for _env_path in (_BACKEND_DIR.parent / ".env", _BACKEND_DIR / ".env"):
    with suppress(OSError):
        load_dotenv(_env_path)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    default_quality = _repo_root() / "data" / "eval" / "learning_graph_quality_v1.jsonl"
    fallback_quality = _repo_root() / "data" / "eval" / "learning_graph_eval_v1.jsonl"
    default_dataset = default_quality if default_quality.is_file() else fallback_quality
    default_out = _repo_root() / "results" / "benchmarks"
    p = argparse.ArgumentParser(
        prog="python -m app.evaluation.runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Synapse evaluation CLI (quality, reliability, audit, curriculum).",
        epilog="""
Command groups
--------------
QUALITY (PRODUCTION-RELATED)
  --benchmark quality [--systems synapse …]

RELIABILITY / AUDIT (PRODUCTION-RELATED)
  --benchmark reliability|audit [--no-llm]

CURRICULUM / INVENTORY (EXPERIMENTAL + OFFLINE)
  --curriculum-inventory-check
  --domain-coverage-report
  --inventory-v2-comparison V1 V2

FINAL / REPORTING (OFFLINE-ANALYSIS)
  --final-40-case [ARTIFACT]
  --curriculum-prior-analysis ARTIFACT
  --expansion-benchmark-analysis ARTIFACT

HISTORICAL EXPERIMENTS (DEPRECATED for product; eval reproducibility only)
  systems: concept_first, baseline_coverage_recovery, domain_prior_edge_classifier
  flags: --inventory-pruning, --representation-alignment, --edge-classifier-prompt-ab, …

Product default generation strategy is always baseline.
Domain curriculum prior is opt-in experimental.
""",
    )
    p.add_argument(
        "--benchmark",
        choices=["quality", "reliability", "audit"],
        default="quality",
        help="Which evaluation to run (default: quality, backward compatible)",
    )
    p.add_argument("--dataset", type=Path, default=None, help="Path to JSONL dataset (quality default: quality_v1)")
    p.add_argument(
        "--systems",
        nargs="+",
        default=["linear_baseline", "direct_llm_graph", "synapse"],
        choices=[
            "linear_baseline",
            "direct_llm_graph",
            "synapse",
            "concept_first",
            "baseline_coverage_recovery",
            "domain_curriculum_prior",
            "domain_prior_edge_classifier",
            "linear",
            "direct",
            "concept-first",
            "coverage_recovery",
            "baseline-coverage-recovery",
            "curriculum_prior",
            "domain_prior",
            "edge_classifier",
            "constrained_dependency",
        ],
        help=(
            "Systems to compare. Product-relevant: synapse (baseline), domain_curriculum_prior. "
            "Experimental only: domain_prior_edge_classifier. "
            "Historical (eval-only): concept_first, baseline_coverage_recovery. "
            "Aliases: linear, direct, concept-first, coverage_recovery, curriculum_prior, edge_classifier."
        ),
    )
    p.add_argument("--model", default=None, help="Override configured model for the active provider")
    p.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Run the same quality dataset across multiple models (e.g. gpt-4o-mini gpt-4o gpt-4.1-mini)",
    )
    p.add_argument("--repetitions", type=int, default=1, help="Repeats per example (use >=3 for stabler aggregates)")
    p.add_argument(
        "--generations",
        type=int,
        default=None,
        help="Alias for --repetitions (preferred name for stability runs)",
    )
    p.add_argument("--seed", type=int, default=42, help="LLM seed where the provider supports it")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--limit", type=int, default=None, help="Evaluate only the first N examples")
    p.add_argument("--ids", default=None, help="Comma-separated example ids")
    p.add_argument("--output-dir", type=Path, default=default_out)
    p.add_argument("--skip-ops-latency", action="store_true", help="Skip ingest/expand/audit/… latency suite")
    p.add_argument("--ops-latency-samples", type=int, default=5)
    p.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM calls (reliability is always deterministic; audit structural-only; quality needs --rescore/--analyze)",
    )
    p.add_argument(
        "--analyze",
        type=Path,
        default=None,
        help="Write a deterministic failure-analysis artifact from an existing quality JSON",
    )
    p.add_argument(
        "--rescore",
        type=Path,
        default=None,
        help="Recompute quality metrics on stored generations (no new LLM calls)",
    )
    p.add_argument(
        "--matching-modes",
        nargs="+",
        default=None,
        choices=["strict", "fair", "curated_alias", "curated", "curated_aliases"],
        help="With --rescore: compare strict/fair/curated_alias on the same generations",
    )
    p.add_argument(
        "--matching-mode",
        default=None,
        choices=["strict", "fair", "curated_alias", "curated", "curated_aliases"],
        help="Single matching mode for --rescore (default: fair / current behavior)",
    )
    p.add_argument(
        "--topic-equivalence-review",
        type=Path,
        default=None,
        help="Write deterministic unmatched-topic equivalence review from a quality artifact",
    )
    p.add_argument(
        "--gold-edge-ambiguity-review",
        type=Path,
        default=None,
        help="Write EXTRA_DEPENDENCY review inventory from a quality artifact (no auto-accept)",
    )
    p.add_argument(
        "--node-edge-attribution",
        type=Path,
        default=None,
        help="Attribute missing/invalid edges to node vs relationship failures (diagnostic; no rescore)",
    )
    p.add_argument(
        "--compare-concept-first",
        type=Path,
        default=None,
        help="Write Concept-First vs baseline comparison from a quality artifact containing both systems",
    )
    p.add_argument(
        "--inventory-attribution",
        type=Path,
        default=None,
        help="Stage-1 inventory quality + causal edge attribution (diagnostic; no LLM if Stage-1 meta present)",
    )
    p.add_argument(
        "--inventory-pruning",
        nargs="?",
        const="DEFAULT",
        default=None,
        metavar="ARTIFACT",
        help=(
            "Offline Stage-1 pruning calibration/replay (no LLM). "
            "Optional path to a quality artifact; default uses the latest CF baseline artifact."
        ),
    )
    p.add_argument(
        "--stability-analysis",
        nargs="?",
        const="DEFAULT",
        default=None,
        metavar="ARTIFACT",
        help=(
            "Baseline multi-generation stability analysis (opt-in; no quality intervention). "
            "Optional path to a multi-generation quality artifact; with a live quality run, "
            "analysis runs after the benchmark when --generations/--repetitions > 1."
        ),
    )
    p.add_argument(
        "--persistent-failure-attribution",
        nargs="?",
        const="DEFAULT",
        default=None,
        metavar="ARTIFACT",
        help=(
            "Offline attribution of stable missing edges to endpoint vs relationship root causes "
            "(no LLM). Optional path to a quality_stability artifact; default uses the latest."
        ),
    )
    p.add_argument(
        "--compare-coverage-recovery",
        type=Path,
        default=None,
        help="Offline baseline vs baseline_coverage_recovery comparison from a quality artifact (no LLM)",
    )
    p.add_argument(
        "--representation-alignment",
        nargs="?",
        const="DEFAULT",
        default=None,
        metavar="ARTIFACT",
        help=(
            "Offline constrained representation alignment replay on a stored baseline artifact "
            "(no LLM, no new concepts). Optional path; default uses latest synapse quality artifact."
        ),
    )
    p.add_argument(
        "--pure-relationship-analysis",
        nargs="?",
        const="DEFAULT",
        default=None,
        metavar="ARTIFACT",
        help=(
            "Offline pure relationship failure analysis on a quality_stability artifact "
            "(no LLM). Isolates BOTH_ENDPOINTS_PRESENT edge omissions with strict EXACT/ALIAS gates."
        ),
    )
    p.add_argument(
        "--missing-concept-information",
        nargs="?",
        const="DEFAULT",
        default=None,
        metavar="ARTIFACT",
        help=(
            "Offline missing-concept information availability analysis on a quality_stability "
            "artifact (no LLM / no LLM judge). Classifies NEVER_PRESENT gold concepts by "
            "whether evidence exists in runtime input (goal + input_notes)."
        ),
    )
    p.add_argument(
        "--curriculum-inventory-check",
        action="store_true",
        help="Offline curriculum inventory validation + gold coverage gate (no LLM)",
    )
    p.add_argument(
        "--domain-coverage-report",
        action="store_true",
        help="Offline mapped/unmapped domain coverage report (no LLM; no inventory mutation)",
    )
    p.add_argument(
        "--expansion-benchmark-analysis",
        type=Path,
        default=None,
        help="Offline expansion analysis (curated_alias+edge_calibrated) for a quality artifact",
    )
    p.add_argument(
        "--inventory-v2-comparison",
        nargs=2,
        metavar=("V1_ARTIFACT", "V2_ARTIFACT"),
        default=None,
        help="Offline databases/data_engineering prior v1 vs v2 comparison (no LLM)",
    )
    p.add_argument(
        "--curriculum-prior-analysis",
        type=Path,
        default=None,
        help="Offline baseline vs domain_curriculum_prior analysis from a quality artifact (no LLM)",
    )
    p.add_argument(
        "--constrained-dependency-analysis",
        type=Path,
        default=None,
        help=(
            "Offline domain_curriculum_prior vs domain_prior_edge_classifier analysis "
            "from a quality artifact (no LLM)"
        ),
    )
    p.add_argument(
        "--edge-classifier-prompt",
        default=None,
        choices=[
            "edge_classifier_baseline",
            "edge_classifier_fewshot_directness",
            "baseline",
            "fewshot",
            "fewshot_directness",
        ],
        help=(
            "Prompt variant for domain_prior_edge_classifier "
            "(default: edge_classifier_baseline)"
        ),
    )
    p.add_argument(
        "--edge-classifier-prompt-ab",
        action="store_true",
        help=(
            "Final few-shot directness A/B for the edge classifier "
            "(shared selection; writes edge_classifier_prompt_ab artifacts)"
        ),
    )
    p.add_argument(
        "--full-dataset",
        action="store_true",
        help=(
            "Do not restrict domain_curriculum_prior / edge_classifier to mapped cases. "
            "Used by the final 40-case benchmark."
        ),
    )
    p.add_argument(
        "--final-40-case",
        nargs="?",
        const="RUN",
        default=None,
        metavar="ARTIFACT",
        help=(
            "Final 40-case system evaluation. Omit ARTIFACT to run live "
            "(synapse + domain_curriculum_prior + domain_prior_edge_classifier). "
            "Pass a quality JSON to analyze offline."
        ),
    )
    p.add_argument(
        "--curriculum-domains",
        nargs="+",
        default=None,
        help="Optional domain filter for --curriculum-inventory-check",
    )
    p.add_argument(
        "--aligned-artifact",
        type=Path,
        default=None,
        help="Optional representation-alignment replay artifact (reference note only; no scoring change)",
    )
    p.add_argument(
        "--edge-matching-modes",
        nargs="+",
        default=None,
        choices=["fair", "current_fair", "edge_calibrated", "edge_ambiguity_calibrated", "calibrated"],
        help="With --rescore: compare fair vs edge_calibrated gold-edge interpretation",
    )
    p.add_argument("--analyze-system", default="synapse", help="System key inside --analyze JSON (default: synapse)")
    p.add_argument(
        "--prompt-variants",
        nargs="+",
        default=None,
        choices=["baseline", "concept_direct_prerequisite", "concept", "concept-direct"],
        help="Quality A/B: run each ingest prompt variant (default: single baseline)",
    )
    p.add_argument(
        "--prompt-variant",
        default=None,
        choices=["baseline", "concept_direct_prerequisite", "concept", "concept-direct"],
        help="Single ingest prompt variant for quality (default: baseline)",
    )
    p.add_argument(
        "--compare-prompts",
        nargs=2,
        metavar=("BASELINE_JSON", "CONCEPT_JSON"),
        default=None,
        help="Write pairwise prompt A/B failure comparison from two quality artifacts",
    )
    p.add_argument(
        "--proposal-metrics-only",
        action="store_true",
        help="Print proposal/human-feedback metrics from the event log and exit",
    )
    return p


def _normalize_systems(raw: list[str]) -> list[str]:
    alias = {
        "linear": "linear_baseline",
        "direct": "direct_llm_graph",
        "synapse": "synapse",
        "concept_first": "concept_first",
        "concept-first": "concept_first",
        "linear_baseline": "linear_baseline",
        "direct_llm_graph": "direct_llm_graph",
        "baseline_coverage_recovery": "baseline_coverage_recovery",
        "coverage_recovery": "baseline_coverage_recovery",
        "baseline-coverage-recovery": "baseline_coverage_recovery",
        "domain_curriculum_prior": "domain_curriculum_prior",
        "curriculum_prior": "domain_curriculum_prior",
        "domain_prior": "domain_curriculum_prior",
        "domain_prior_edge_classifier": "domain_prior_edge_classifier",
        "edge_classifier": "domain_prior_edge_classifier",
        "constrained_dependency": "domain_prior_edge_classifier",
    }
    out: list[str] = []
    for s in raw:
        key = alias[s]
        if key not in out:
            out.append(key)
    return out


def _apply_model_override(model: str | None) -> None:
    if not model:
        return
    provider = (os.environ.get("LLM_PROVIDER") or "openai").strip().lower()
    if provider == "gemini":
        os.environ["GEMINI_MODEL"] = model
    elif provider == "openai_compatible":
        os.environ["OPENAI_COMPATIBLE_MODEL"] = model
    else:
        os.environ["OPENAI_MODEL"] = model
    from app.services.llm import reset_llm_provider

    reset_llm_provider()


def _quality_dataset(args: argparse.Namespace) -> Path:
    if args.dataset is not None:
        return Path(args.dataset)
    quality = _repo_root() / "data" / "eval" / "learning_graph_quality_v1.jsonl"
    legacy = _repo_root() / "data" / "eval" / "learning_graph_eval_v1.jsonl"
    return quality if quality.is_file() else legacy


async def _run_quality(args: argparse.Namespace, *, model: str | None, prompt_variant: str | None = None) -> dict:
    from app.evaluation.benchmark import run_benchmark
    from app.evaluation.dataset import filter_examples, load_dataset
    from app.prompts.ingest import resolve_prompt_variant

    _apply_model_override(model)
    dataset = _quality_dataset(args)
    examples = load_dataset(dataset)
    ids = {x.strip() for x in args.ids.split(",")} if args.ids else None
    systems = _normalize_systems(list(args.systems))
    if (
        ("domain_curriculum_prior" in systems or "domain_prior_edge_classifier" in systems)
        and ids is None
        and not getattr(args, "full_dataset", False)
    ):
        from app.curriculum.inventory import load_case_domain_map

        mapped = set(load_case_domain_map())
        if args.curriculum_domains:
            cmap = load_case_domain_map()
            allowed = set(args.curriculum_domains)
            mapped = {cid for cid, d in cmap.items() if d in allowed}
        ids = mapped
    examples = filter_examples(examples, ids=ids, limit=args.limit)
    if not examples:
        raise SystemExit("No examples selected.")
    variant = resolve_prompt_variant(prompt_variant)
    repetitions = _effective_repetitions(args)
    stability = args.stability_analysis is not None
    btype = "quality_stability" if (stability and repetitions > 1) else "quality"
    print(
        f"Running quality evaluation on {len(examples)} example(s), "
        f"systems={systems}, model={model or 'configured'}, "
        f"prompt={variant}, repetitions={repetitions}"
        + (f", benchmark_type={btype}" if btype != "quality" else "")
        + "…",
        flush=True,
    )
    return await run_benchmark(
        examples,
        systems=systems,  # type: ignore[arg-type]
        repetitions=repetitions,
        temperature=args.temperature,
        seed=args.seed,
        include_ops_latency=not args.skip_ops_latency,
        ops_latency_samples=max(0, args.ops_latency_samples),
        dataset_name=dataset.stem,
        model=model,
        prompt_variant=variant,
        edge_classifier_prompt_variant=getattr(args, "edge_classifier_prompt", None),
        benchmark_type=btype,
    )


def _effective_repetitions(args: argparse.Namespace) -> int:
    if getattr(args, "generations", None) is not None:
        return max(1, int(args.generations))
    return max(1, int(args.repetitions))


def _synapse_quality_row(result: dict) -> dict:
    block = (result.get("systems") or {}).get("synapse") or {}
    m = block.get("metrics") or {}
    lat = block.get("latency") or {}
    cost = block.get("cost") or {}
    return {
        "model": result.get("model"),
        "provider": result.get("provider"),
        "topic_f1": m.get("topic_f1"),
        "dependency_f1": m.get("dependency_f1"),
        "missing_prerequisite_rate": m.get("missing_prerequisite_rate"),
        "dependency_direction_error_rate": m.get("dependency_direction_error_rate"),
        "cycle_attempt_rate": m.get("cycle_attempt_rate"),
        "p50_ms": lat.get("p50_ms"),
        "avg_cost_usd": cost.get("average_cost_usd"),
    }


async def _async_main(args: argparse.Namespace) -> int:
    if args.proposal_metrics_only:
        import json

        from app.evaluation.proposal_metrics import collect_proposal_metrics

        print(json.dumps(collect_proposal_metrics(), indent=2))
        return 0

    if args.analyze:
        from app.evaluation.inspect import analyze_benchmark

        dataset = args.dataset or _quality_dataset(args)
        path = analyze_benchmark(
            args.analyze,
            dataset_path=dataset,
            system=args.analyze_system,
        )
        print(f"Wrote {path}")
        return 0

    if args.topic_equivalence_review:
        from app.evaluation.topic_equivalence import build_topic_equivalence_review

        dataset = args.dataset or _quality_dataset(args)
        path = build_topic_equivalence_review(
            args.topic_equivalence_review,
            dataset_path=dataset,
            system=args.analyze_system,
        )
        print(f"Wrote {path}")
        print(f"Wrote {path.with_suffix('.md')}")
        return 0

    if args.gold_edge_ambiguity_review:
        from app.evaluation.edge_ambiguity import build_gold_edge_ambiguity_review

        dataset = args.dataset or _quality_dataset(args)
        path = build_gold_edge_ambiguity_review(
            args.gold_edge_ambiguity_review,
            dataset_path=dataset,
            system=args.analyze_system,
        )
        print(f"Wrote {path}")
        print(f"Wrote {path.with_suffix('.md')}")
        return 0

    if args.node_edge_attribution:
        from app.evaluation.node_edge_attribution import run_node_edge_attribution

        dataset = args.dataset or _quality_dataset(args)
        path = run_node_edge_attribution(
            args.node_edge_attribution,
            dataset_path=dataset,
            system=args.analyze_system,
        )
        print(f"Wrote {path}")
        md = path.parent / path.name.replace("_node_edge_attribution.json", "_node_vs_edge_error_analysis.md")
        if md.is_file():
            print(f"Wrote {md}")
        return 0

    if args.compare_concept_first:
        from app.evaluation.concept_first_compare import compare_concept_first_runs

        md_path, json_path = compare_concept_first_runs(args.compare_concept_first)
        print(f"Wrote {md_path}")
        print(f"Wrote {json_path}")
        return 0

    if args.inventory_attribution:
        from app.evaluation.inventory_attribution import run_inventory_attribution

        dataset = args.dataset or _quality_dataset(args)
        md_path, json_path = run_inventory_attribution(
            args.inventory_attribution,
            dataset_path=dataset,
        )
        print(f"Wrote {md_path}")
        print(f"Wrote {json_path}")
        return 0

    if args.inventory_pruning is not None:
        from app.evaluation.inventory_pruning_analysis import run_inventory_pruning_analysis

        artifact = None if args.inventory_pruning == "DEFAULT" else Path(args.inventory_pruning)
        dataset = args.dataset or _quality_dataset(args)
        md_path, json_path = run_inventory_pruning_analysis(
            artifact,
            dataset_path=dataset,
        )
        print(f"Wrote {md_path}")
        print(f"Wrote {json_path}")
        return 0

    if args.stability_analysis is not None and args.stability_analysis != "DEFAULT":
        from app.evaluation.stability_analysis import run_baseline_stability_analysis

        dataset = args.dataset or _quality_dataset(args)
        md_path, json_path = run_baseline_stability_analysis(
            args.stability_analysis,
            dataset_path=dataset,
            system=args.analyze_system,
        )
        print(f"Wrote {md_path}")
        print(f"Wrote {json_path}")
        return 0

    if args.stability_analysis == "DEFAULT" and args.no_llm:
        from app.evaluation.stability_analysis import run_baseline_stability_analysis

        bench_dir = _repo_root() / "results" / "benchmarks"
        candidates = sorted(bench_dir.glob("*_quality_stability_*.json"), reverse=True)
        if not candidates:
            candidates = sorted(bench_dir.glob("*_quality_*.json"), reverse=True)
        if not candidates:
            print("No benchmark artifact found for --stability-analysis.", file=sys.stderr)
            return 2
        dataset = args.dataset or _quality_dataset(args)
        md_path, json_path = run_baseline_stability_analysis(
            candidates[0],
            dataset_path=dataset,
            system=args.analyze_system,
        )
        print(f"Wrote {md_path}")
        print(f"Wrote {json_path}")
        return 0

    if args.persistent_failure_attribution is not None:
        from app.evaluation.persistent_failure_attribution import (
            find_latest_stability_artifact,
            run_persistent_failure_attribution,
        )

        if args.persistent_failure_attribution == "DEFAULT":
            artifact = find_latest_stability_artifact()
        else:
            artifact = Path(args.persistent_failure_attribution)
        dataset = args.dataset or _quality_dataset(args)
        md_path, json_path, pareto_path = run_persistent_failure_attribution(
            artifact,
            dataset_path=dataset,
            system=args.analyze_system,
        )
        print(f"Wrote {md_path}")
        print(f"Wrote {json_path}")
        print(f"Wrote {pareto_path}")
        return 0

    if args.compare_coverage_recovery:
        from app.evaluation.coverage_recovery_compare import compare_coverage_recovery_runs

        dataset = args.dataset or _quality_dataset(args)
        md_path, json_path = compare_coverage_recovery_runs(
            args.compare_coverage_recovery,
            dataset_path=dataset,
        )
        print(f"Wrote {md_path}")
        print(f"Wrote {json_path}")
        return 0

    if args.curriculum_inventory_check:
        from app.evaluation.curriculum_inventory_check import run_curriculum_inventory_check
        from app.evaluation.domain_coverage_report import run_inventory_health_only

        dataset = args.dataset or _quality_dataset(args)
        md_path, json_path = run_curriculum_inventory_check(
            dataset_path=dataset,
            domains=args.curriculum_domains,
        )
        print(f"Wrote {md_path}")
        print(f"Wrote {json_path}")
        health_md, health_js = run_inventory_health_only()
        print(f"Wrote {health_md}")
        print(f"Wrote {health_js}")
        return 0

    if args.domain_coverage_report:
        from app.evaluation.domain_coverage_report import run_domain_coverage_report

        dataset = args.dataset or _quality_dataset(args)
        md_path, json_path = run_domain_coverage_report(dataset_path=dataset)
        print(f"Wrote {md_path}")
        print(f"Wrote {json_path}")
        return 0

    if args.expansion_benchmark_analysis is not None:
        from app.evaluation.expansion_benchmark_analysis import run_expansion_benchmark_analysis

        dataset = args.dataset or _quality_dataset(args)
        md_path, json_path = run_expansion_benchmark_analysis(
            args.expansion_benchmark_analysis,
            dataset_path=dataset,
        )
        print(f"Wrote {md_path}")
        print(f"Wrote {json_path}")
        return 0

    if args.inventory_v2_comparison is not None:
        from app.evaluation.inventory_v2_comparison import run_inventory_v2_comparison

        dataset = args.dataset or _quality_dataset(args)
        v1_art, v2_art = args.inventory_v2_comparison
        md_path, json_path = run_inventory_v2_comparison(
            v1_artifact=v1_art,
            v2_artifact=v2_art,
            dataset_path=dataset,
        )
        print(f"Wrote {md_path}")
        print(f"Wrote {json_path}")
        return 0

    if args.curriculum_prior_analysis is not None:
        from app.evaluation.curriculum_prior_analysis import run_curriculum_prior_analysis

        dataset = args.dataset or _quality_dataset(args)
        md_path, json_path = run_curriculum_prior_analysis(
            args.curriculum_prior_analysis,
            dataset_path=dataset,
        )
        print(f"Wrote {md_path}")
        print(f"Wrote {json_path}")
        return 0

    if args.constrained_dependency_analysis is not None:
        from app.evaluation.constrained_dependency_analysis import (
            run_constrained_dependency_analysis,
        )

        dataset = args.dataset or _quality_dataset(args)
        json_path, md_path = run_constrained_dependency_analysis(
            args.constrained_dependency_analysis,
            dataset_path=dataset,
        )
        print(f"Wrote {md_path}")
        print(f"Wrote {json_path}")
        return 0

    if args.edge_classifier_prompt_ab:
        from app.evaluation.edge_classifier_prompt_ab import run_edge_classifier_prompt_ab

        _apply_model_override(args.model)
        bench_path, json_path, md_path = await run_edge_classifier_prompt_ab(
            dataset_path=args.dataset or _quality_dataset(args),
            model=args.model or "gpt-4o-mini",
            temperature=args.temperature,
            seed=args.seed,
        )
        print(f"Wrote {bench_path}")
        print(f"Wrote {md_path}")
        print(f"Wrote {json_path}")
        return 0

    if getattr(args, "final_40_case", None) is not None:
        from app.evaluation.final_40_case_comparison import (
            build_final_comparison,
            run_final_40_case_live,
        )
        from app.evaluation.reliability import run_reliability_benchmark

        if args.final_40_case != "RUN":
            reliability = run_reliability_benchmark()
            json_path, md_path = build_final_comparison(
                Path(args.final_40_case),
                dataset_path=args.dataset or _quality_dataset(args),
                reliability=reliability,
            )
            print(f"Wrote {json_path}")
            print(f"Wrote {md_path}")
            return 0

        _apply_model_override(args.model)
        gens = _effective_repetitions(args)
        if getattr(args, "generations", None) is None and args.repetitions == 1:
            gens = 3
        quality_path, json_path, md_path = await run_final_40_case_live(
            model=args.model or "gpt-4o-mini",
            generations=gens,
            temperature=args.temperature,
            seed=args.seed,
            output_dir=args.output_dir,
        )
        print(f"Wrote {quality_path}")
        print(f"Wrote {md_path}")
        print(f"Wrote {json_path}")
        return 0

    if args.missing_concept_information is not None:
        from app.evaluation.persistent_failure_attribution import find_latest_stability_artifact
        from app.evaluation.missing_concept_information import (
            run_missing_concept_information_analysis,
        )

        artifact = (
            find_latest_stability_artifact()
            if args.missing_concept_information == "DEFAULT"
            else Path(args.missing_concept_information)
        )
        dataset = args.dataset or _quality_dataset(args)
        md_path, json_path, pareto_path = run_missing_concept_information_analysis(
            artifact,
            dataset_path=dataset,
            system=args.analyze_system,
        )
        print(f"Wrote {md_path}")
        print(f"Wrote {json_path}")
        print(f"Wrote {pareto_path}")
        return 0

    if args.pure_relationship_analysis is not None:
        from app.evaluation.persistent_failure_attribution import find_latest_stability_artifact
        from app.evaluation.pure_relationship_analysis import run_pure_relationship_analysis

        artifact = (
            find_latest_stability_artifact()
            if args.pure_relationship_analysis == "DEFAULT"
            else Path(args.pure_relationship_analysis)
        )
        dataset = args.dataset or _quality_dataset(args)
        md_path, json_path, pareto_path = run_pure_relationship_analysis(
            artifact,
            dataset_path=dataset,
            system=args.analyze_system,
            aligned_artifact_path=args.aligned_artifact,
        )
        print(f"Wrote {md_path}")
        print(f"Wrote {json_path}")
        print(f"Wrote {pareto_path}")
        return 0

    if args.representation_alignment is not None:
        from app.evaluation.representation_alignment_analysis import (
            find_baseline_artifact,
            run_representation_alignment_replay,
        )

        artifact = (
            find_baseline_artifact()
            if args.representation_alignment == "DEFAULT"
            else Path(args.representation_alignment)
        )
        dataset = args.dataset or _quality_dataset(args)
        md_path, json_path, bench_path = run_representation_alignment_replay(
            artifact,
            dataset_path=dataset,
            system=args.analyze_system,
        )
        print(f"Wrote {md_path}")
        print(f"Wrote {json_path}")
        print(f"Wrote {bench_path}")
        return 0

    if args.rescore:
        dataset = args.dataset or _quality_dataset(args)
        if args.edge_matching_modes:
            from app.evaluation.edge_ambiguity import rescore_edge_ambiguity_modes

            path = rescore_edge_ambiguity_modes(
                args.rescore,
                modes=list(args.edge_matching_modes),
                dataset_path=dataset,
                system=args.analyze_system,
                output_dir=args.output_dir,
                topic_matching_mode="curated_alias",
            )
            print(f"Wrote {path}")
            print(f"Wrote {path.with_suffix('.md')}")
            return 0
        if args.matching_modes:
            from app.evaluation.matching_calibration import rescore_matching_modes

            path = rescore_matching_modes(
                args.rescore,
                modes=list(args.matching_modes),
                dataset_path=dataset,
                system=args.analyze_system,
                output_dir=args.output_dir,
            )
            print(f"Wrote {path}")
            print(f"Wrote {path.with_suffix('.md')}")
            return 0
        if args.matching_mode:
            from app.evaluation.matching_calibration import rescore_benchmark_with_mode

            path = rescore_benchmark_with_mode(
                args.rescore,
                matching_mode=args.matching_mode,
                dataset_path=dataset,
                output_dir=args.output_dir,
            )
            print(f"Wrote {path}")
            print(f"Wrote {path.with_suffix('.md')}")
            return 0
        from app.evaluation.inspect import rescore_benchmark

        path = rescore_benchmark(args.rescore, dataset_path=dataset, output_dir=args.output_dir)
        print(f"Wrote {path}")
        print(f"Wrote {path.with_suffix('.md')}")
        return 0

    if args.compare_prompts:
        from app.evaluation.prompt_compare import compare_prompt_variant_runs

        path = compare_prompt_variant_runs(
            args.compare_prompts[0],
            args.compare_prompts[1],
            dataset_path=args.dataset or _quality_dataset(args),
            system=args.analyze_system,
        )
        print(f"Wrote {path}")
        print(f"Wrote {path.with_suffix('.md')}")
        return 0

    from app.evaluation.reporting import write_benchmark_result

    if args.benchmark == "reliability":
        from app.evaluation.isolation import isolate_eval_runtime

        isolate_eval_runtime(force=True)
        from app.evaluation.reliability import run_reliability_benchmark

        print("Running deterministic reliability benchmark (no LLM)…", flush=True)
        result = run_reliability_benchmark()
        path = write_benchmark_result(result, args.output_dir)
        print(f"Wrote {path}")
        print(f"Wrote {path.with_suffix('.md')}")
        return 0

    if args.benchmark == "audit":
        from app.evaluation.isolation import isolate_eval_runtime

        isolate_eval_runtime(force=True)
        from app.evaluation.audit_eval import run_audit_benchmark

        no_llm = bool(args.no_llm)
        print(
            "Running audit benchmark ({})…".format(
                "structural only" if no_llm else "structural + semantic LLM",
            ),
            flush=True,
        )
        result = await run_audit_benchmark(no_llm=no_llm)
        path = write_benchmark_result(result, args.output_dir)
        print(f"Wrote {path}")
        print(f"Wrote {path.with_suffix('.md')}")
        return 0

    if args.no_llm:
        print("quality benchmark requires an LLM unless you pass --rescore or --analyze.", file=sys.stderr)
        return 2

    from app.prompts.ingest import resolve_prompt_variant

    models = list(args.models) if args.models else [args.model]
    if args.prompt_variants:
        variants = [resolve_prompt_variant(v) for v in args.prompt_variants]
    elif args.prompt_variant:
        variants = [resolve_prompt_variant(args.prompt_variant)]
    else:
        variants = ["baseline"]

    deduped: list[str] = []
    for v in variants:
        if v not in deduped:
            deduped.append(v)
    variants = deduped

    if len(models) == 1 and len(variants) == 1:
        result = await _run_quality(args, model=models[0], prompt_variant=variants[0])
        path = write_benchmark_result(result, args.output_dir)
        print(f"Wrote {path}")
        print(f"Wrote {path.with_suffix('.md')}")
        if args.stability_analysis is not None:
            from app.evaluation.stability_analysis import run_baseline_stability_analysis

            md_path, json_path = run_baseline_stability_analysis(
                path,
                dataset_path=_quality_dataset(args),
                system=args.analyze_system,
            )
            print(f"Wrote {md_path}")
            print(f"Wrote {json_path}")
        return 0

    if len(models) == 1 and len(variants) > 1:
        # Keep prompt as the only controlled variable (ops suite uses production default prompt).
        if not args.skip_ops_latency:
            print("Note: skipping ops-latency suite for prompt A/B (prompt must be the only variable).", flush=True)
            args.skip_ops_latency = True
        by_variant: dict[str, Any] = {}
        paths: list[str] = []
        for variant in variants:
            result = await _run_quality(args, model=models[0], prompt_variant=variant)
            path = write_benchmark_result(result, args.output_dir)
            paths.append(str(path))
            print(f"Wrote {path}")
            by_variant[variant] = {
                "artifact": str(path),
                "prompt_variant": result.get("prompt_variant"),
                "prompt_version": result.get("prompt_version"),
                "prompt_hash": result.get("prompt_hash"),
                "metrics": result.get("metrics"),
                "latency": result.get("latency"),
                "cost": result.get("cost"),
            }
        if "baseline" in by_variant and "concept_direct_prerequisite" in by_variant:
            from app.evaluation.prompt_compare import compare_prompt_variant_runs

            cmp_path = compare_prompt_variant_runs(
                by_variant["baseline"]["artifact"],
                by_variant["concept_direct_prerequisite"]["artifact"],
                dataset_path=_quality_dataset(args),
                system="synapse" if "synapse" in _normalize_systems(list(args.systems)) else "direct_llm_graph",
            )
            print(f"Wrote {cmp_path}")
            print(f"Wrote {cmp_path.with_suffix('.md')}")
        combined = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "benchmark_type": "quality",
            "dataset": _quality_dataset(args).stem,
            "dataset_version": _quality_dataset(args).stem,
            "model": f"prompt-ab-{(models[0] or 'model')}",
            "prompt_variants": variants,
            "per_variant": by_variant,
            "per_variant_artifacts": paths,
            "metrics": {},
            "failures": {},
            "latency": {},
            "cost": {},
            "notes": [
                "A/B ingest prompt experiment: same model/dataset/settings; only prompt_variant differs.",
                "Do not claim DAG validation caused quality changes.",
            ],
        }
        path = write_benchmark_result(combined, args.output_dir)
        print(f"Wrote prompt A/B summary {path}")
        return 0

    comparison_rows = []
    child_paths = []
    last = None
    for model in models:
        result = await _run_quality(args, model=model, prompt_variant=variants[0])
        last = result
        path = write_benchmark_result(result, args.output_dir)
        child_paths.append(str(path))
        print(f"Wrote {path}")
        comparison_rows.append(_synapse_quality_row(result))

    combined = {
        "timestamp": last["timestamp"] if last else "",
        "benchmark_type": "quality",
        "dataset": last.get("dataset") if last else "",
        "dataset_version": last.get("dataset_version") if last else "",
        "model": "comparison",
        "provider": last.get("provider") if last else "",
        "prompt_variant": last.get("prompt_variant") if last else variants[0],
        "seed": args.seed,
        "repetitions": args.repetitions,
        "example_count": last.get("example_count") if last else 0,
        "metrics": {row["model"]: row for row in comparison_rows},
        "failures": {},
        "latency": {},
        "cost": {},
        "model_comparison": comparison_rows,
        "per_model_artifacts": child_paths,
        "notes": [
            "Each model is a separate quality run with identical dataset/seed/systems.",
            "Synapse vs Direct quality still differs only when validation drops invalid edges.",
        ],
    }
    path = write_benchmark_result(combined, args.output_dir)
    print(f"Wrote comparison {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return asyncio.run(_async_main(args))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
