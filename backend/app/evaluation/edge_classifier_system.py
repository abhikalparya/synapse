"""Evaluation adapter for domain_prior_edge_classifier (opt-in)."""

from __future__ import annotations

from typing import Any

from app.curriculum.inventory import resolve_domain_for_case
from app.evaluation.baselines import build_source_text
from app.evaluation.failure_analysis import classify_llm_exception
from app.evaluation.schemas import EvalExample, GeneratedGraph
from app.services.domain_prior_edge_classifier import run_domain_prior_edge_classifier_pipeline


async def run_domain_prior_edge_classifier(
    example: EvalExample,
    *,
    domain: str | None = None,
    temperature: float = 0.0,
    seed: int | None = 42,
    prompt_variant: str | None = None,
) -> tuple[GeneratedGraph, dict[str, Any]]:
    try:
        resolved = resolve_domain_for_case(example.id, domain_override=domain)
    except ValueError as exc:
        graph = GeneratedGraph(
            topics=[],
            dependencies=[],
            parse_ok=False,
            error=str(exc),
            error_category="config_error",
            generation_meta={"generation_strategy": "domain_prior_edge_classifier"},
        )
        return graph, {
            "llm_latency_ms": 0.0,
            "error": str(exc),
            "error_category": "config_error",
            "generation_strategy": "domain_prior_edge_classifier",
            "domain_status": "DOMAIN_UNRESOLVED",
        }

    try:
        result = await run_domain_prior_edge_classifier_pipeline(
            build_source_text(example),
            domain=resolved,
            temperature=temperature,
            seed=seed,
            prompt_variant=prompt_variant,
        )
    except Exception as exc:
        category = classify_llm_exception(exc)
        graph = GeneratedGraph(
            topics=[],
            dependencies=[],
            parse_ok=False,
            error=str(exc),
            error_category=category,
            generation_meta={
                "generation_strategy": "domain_prior_edge_classifier",
                "domain": resolved,
            },
        )
        return graph, {
            "llm_latency_ms": 0.0,
            "error": str(exc),
            "error_category": category,
            "generation_strategy": "domain_prior_edge_classifier",
            "domain": resolved,
        }

    meta = result.to_meta()
    raw = "\n---\n".join(result.classification_raw_batches) or result.selection_raw
    if not result.parse_ok or not result.topics:
        graph = GeneratedGraph(
            topics=[],
            dependencies=[],
            parse_ok=False,
            error="; ".join(result.errors) or "domain prior edge classifier failed",
            error_category="parse_error",
            generation_meta=meta,
            raw_response=raw,
        )
        return graph, meta

    deps = [(d["from"], d["to"]) for d in result.dependencies]
    skipped = []
    for s in result.skipped_dependencies:
        if hasattr(s, "from_title"):
            skipped.append(
                {"from_title": s.from_title, "to_title": s.to_title, "reason": s.reason}
            )
        elif isinstance(s, dict):
            skipped.append(s)

    graph = GeneratedGraph(
        topics=[t["title"] for t in result.topics],
        dependencies=deps,
        skipped_dependencies=skipped,
        topic_confidences=[float(t.get("confidence", 0.7)) for t in result.topics],
        raw_response=raw,
        parse_ok=True,
        generation_meta=meta,
    )
    return graph, meta
