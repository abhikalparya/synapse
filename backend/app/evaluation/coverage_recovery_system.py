"""Eval adapter: baseline Synapse graph + one targeted coverage-recovery pass."""

from __future__ import annotations

from typing import Any

from app.evaluation.schemas import GeneratedGraph
from app.evaluation.synapse_system import run_synapse_from_raw
from app.services.coverage_recovery import run_coverage_recovery_pass


async def run_baseline_coverage_recovery_from_raw(
    raw: str,
    *,
    learning_objective: str,
    temperature: float = 0.0,
    seed: int | None = None,
    baseline_meta: dict[str, Any] | None = None,
) -> tuple[GeneratedGraph, dict[str, Any]]:
    """Parse baseline raw → Synapse DAG → coverage recovery → in-memory graph.

    Does not write the database. Recovery is a single non-recursive pass.
    """
    baseline = run_synapse_from_raw(raw)
    meta: dict[str, Any] = dict(baseline_meta or {})
    meta["generation_strategy"] = "baseline_coverage_recovery"
    meta["baseline_topics"] = list(baseline.topics)
    meta["baseline_dependencies"] = [list(d) for d in baseline.dependencies]

    if not baseline.parse_ok:
        baseline.generation_meta = {**(baseline.generation_meta or {}), **meta}
        return baseline, meta

    baseline_topics = [
        {
            "title": t,
            "summary": "",
            "confidence": (
                float(baseline.topic_confidences[i])
                if baseline.topic_confidences and i < len(baseline.topic_confidences)
                else 0.7
            ),
        }
        for i, t in enumerate(baseline.topics)
    ]
    baseline_deps = [{"from": a, "to": b} for a, b in baseline.dependencies]

    recovery = await run_coverage_recovery_pass(
        learning_objective=learning_objective,
        baseline_topics=baseline_topics,
        baseline_dependencies=baseline_deps,
        temperature=temperature,
        seed=None if seed is None else seed + 17,
    )
    rmeta = recovery.to_meta()
    meta.update(rmeta)
    meta["baseline_llm_latency_ms"] = float((baseline_meta or {}).get("llm_latency_ms") or 0.0)
    meta["llm_latency_ms"] = float(meta.get("baseline_llm_latency_ms") or 0.0) + float(
        recovery.llm_latency_ms or 0.0
    )
    # Cost: sum baseline + recovery when available
    bcost = (baseline_meta or {}).get("cost_usd")
    rcost = recovery.cost_usd
    if bcost is not None or rcost is not None:
        meta["cost_usd"] = float(bcost or 0.0) + float(rcost or 0.0)
        meta["tokens_estimated"] = bool(
            (baseline_meta or {}).get("tokens_estimated", True) or recovery.tokens_estimated
        )
    bin_tok = (baseline_meta or {}).get("input_tokens")
    bout_tok = (baseline_meta or {}).get("output_tokens")
    if bin_tok is not None or recovery.input_tokens is not None:
        meta["input_tokens"] = int(bin_tok or 0) + int(recovery.input_tokens or 0)
    if bout_tok is not None or recovery.output_tokens is not None:
        meta["output_tokens"] = int(bout_tok or 0) + int(recovery.output_tokens or 0)

    graph = GeneratedGraph(
        topics=[str(t["title"]) for t in recovery.topics_after],
        dependencies=[(str(d["from"]), str(d["to"])) for d in recovery.dependencies_after],
        skipped_dependencies=[
            {"from_title": s.from_title, "to_title": s.to_title, "reason": s.reason}
            for s in recovery.skipped_dependencies
        ]
        + list(baseline.skipped_dependencies or []),
        topic_confidences=[float(t.get("confidence", 0.7)) for t in recovery.topics_after],
        raw_response=recovery.raw_response or raw,
        parse_ok=True,
        generation_meta=meta,
    )
    return graph, meta
