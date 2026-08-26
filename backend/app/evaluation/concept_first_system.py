"""Evaluation adapter for the experimental Concept-First generation system."""

from __future__ import annotations

from typing import Any

from app.evaluation.baselines import build_source_text
from app.evaluation.failure_analysis import classify_llm_exception
from app.evaluation.schemas import EvalExample, GeneratedGraph
from app.services.concept_first import run_concept_first_pipeline


async def run_concept_first(
    example: EvalExample,
    *,
    temperature: float = 0.0,
    seed: int | None = 42,
    enable_pruning: bool = False,
    prune_config_name: str = "combined_conservative",
) -> tuple[GeneratedGraph, dict[str, Any]]:
    """Run Concept-First pipeline and map to GeneratedGraph + cost/latency meta."""
    strategy = "concept_first_pruned" if enable_pruning else "concept_first"
    try:
        cf = await run_concept_first_pipeline(
            build_source_text(example),
            temperature=temperature,
            seed=seed,
            enable_pruning=enable_pruning,
            prune_config_name=prune_config_name,
        )
    except Exception as exc:
        category = classify_llm_exception(exc)
        graph = GeneratedGraph(
            topics=[],
            dependencies=[],
            parse_ok=False,
            error=str(exc),
            error_category=category,
            generation_meta={"generation_strategy": strategy},
        )
        return graph, {
            "llm_latency_ms": 0.0,
            "input_tokens": None,
            "output_tokens": None,
            "tokens_estimated": True,
            "cost_usd": None,
            "error": str(exc),
            "error_category": category,
            "generation_strategy": strategy,
        }

    meta = cf.to_meta()
    if not cf.parse_ok or not cf.topics:
        graph = GeneratedGraph(
            topics=[],
            dependencies=[],
            parse_ok=False,
            error="; ".join(cf.errors) or "concept-first failed",
            error_category="parse_error",
            generation_meta=meta,
            raw_response=cf.concept_raw or cf.dependency_raw,
        )
        return graph, meta

    deps = [(d["from"], d["to"]) for d in cf.dependencies]
    skipped = []
    for s in cf.skipped_dependencies:
        if hasattr(s, "from_title"):
            skipped.append(
                {"from_title": s.from_title, "to_title": s.to_title, "reason": s.reason}
            )
        elif isinstance(s, dict):
            skipped.append(s)

    graph = GeneratedGraph(
        topics=[t["title"] for t in cf.topics],
        dependencies=deps,
        skipped_dependencies=skipped,
        topic_confidences=[float(t.get("confidence", 0.7)) for t in cf.topics],
        raw_response=cf.dependency_raw or cf.concept_raw,
        parse_ok=True,
        error="; ".join(cf.errors) if cf.errors else None,
        generation_meta=meta,
    )
    return graph, meta
