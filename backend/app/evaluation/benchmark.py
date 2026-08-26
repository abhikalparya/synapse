"""Benchmark orchestration: golden-dataset graph quality + optional operation latency."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Sequence

from app.evaluation.baselines import (
    generate_direct_llm_raw,
    run_direct_from_raw,
    run_linear_baseline,
)
from app.evaluation.concept_first_system import run_concept_first
from app.evaluation.cost import pricing_metadata
from app.evaluation.failure_analysis import summarize_failures
from app.evaluation.latency import summarize_latencies_ms
from app.evaluation.metrics import aggregate_scores, score_graph
from app.evaluation.proposal_metrics import collect_proposal_metrics
from app.evaluation.schemas import EvalExample, GeneratedGraph, GraphQualityScores, SystemExampleResult, SystemName
from app.evaluation.synapse_system import run_synapse_from_raw
from app.prompts.ask import build_ask_prompt
from app.prompts.audit import build_audit_prompt
from app.prompts.expand import build_expand_prompt
from app.prompts.quiz import build_quiz_prompt
from app.prompts.reshape import build_reshape_prompt
from app.services.llm import call_llm_detailed, capture_llm_calls, llm_operation


def _mean_or_none(values: list[float | None]) -> float | None:
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def _system_block(results: list[SystemExampleResult]) -> dict[str, Any]:
    scores = [r.scores for r in results if r.scores is not None]
    metrics = aggregate_scores(scores)
    total_latencies = [r.total_latency_ms for r in results]
    llm_latencies = [r.llm_latency_ms for r in results]
    det_latencies = [r.deterministic_latency_ms for r in results]
    costs = [r.cost_usd for r in results]
    failures = summarize_failures(r.failures for r in results)

    successful = [r for r in results if r.scores is not None and r.graph.parse_ok]
    avg_cost = _mean_or_none([r.cost_usd for r in successful])
    total_cost = sum(c for c in costs if c is not None) if any(c is not None for c in costs) else None

    return {
        "metrics": metrics,
        "latency": {
            **summarize_latencies_ms(total_latencies),
            "llm": summarize_latencies_ms(llm_latencies),
            "deterministic": summarize_latencies_ms(det_latencies),
        },
        "cost": {
            "average_cost_usd": avg_cost,
            "total_cost_usd": total_cost,
            "average_cost_per_successful_operation": avg_cost,
            "note": "USD estimates from versioned pricing table; null when model/tokens unavailable.",
        },
        "failures": failures,
        "example_results": [
            {
                "example_id": r.example_id,
                "repetition": r.repetition,
                "generation_index": r.repetition,
                "seed": (r.graph.generation_meta or {}).get("seed"),
                "seed_supported": (r.graph.generation_meta or {}).get("seed_supported"),
                "parse_ok": r.graph.parse_ok,
                "error": r.graph.error,
                "failures": r.failures,
                "scores": None
                if r.scores is None
                else {
                    "topic_precision": r.scores.topic_precision,
                    "topic_recall": r.scores.topic_recall,
                    "topic_f1": r.scores.topic_f1,
                    "dependency_precision": r.scores.dependency_precision,
                    "dependency_recall": r.scores.dependency_recall,
                    "dependency_f1": r.scores.dependency_f1,
                    "required_edge_precision": r.scores.required_edge_precision,
                    "required_edge_recall": r.scores.required_edge_recall,
                    "required_edge_f1": r.scores.required_edge_f1,
                    "missing_required_edge_rate": r.scores.missing_required_edge_rate,
                    "invalid_extra_edge_rate": r.scores.invalid_extra_edge_rate,
                    "graph_valid": r.scores.graph_valid,
                    "cycle_attempt": r.scores.cycle_attempt,
                    "missing_prerequisite_rate": r.scores.missing_prerequisite_rate,
                    "hallucinated_topic_rate": r.scores.hallucinated_topic_rate,
                    "extra_dependency_rate": r.scores.extra_dependency_rate,
                    "dependency_direction_error_rate": r.scores.dependency_direction_error_rate,
                    "redundant_transitive_edge_rate": r.scores.redundant_transitive_edge_rate,
                    "redundant_transitive_edge_count": r.scores.redundant_transitive_edge_count,
                },
                "total_latency_ms": r.total_latency_ms,
                "llm_latency_ms": r.llm_latency_ms,
                "deterministic_latency_ms": r.deterministic_latency_ms,
                "cost_usd": r.cost_usd,
                "cost_estimated": r.cost_estimated,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "generated_topics": r.graph.topics,
                "generated_dependencies": [list(d) for d in r.graph.dependencies],
                "skipped_dependencies": r.graph.skipped_dependencies,
                "generation_meta": r.graph.generation_meta or None,
            }
            for r in results
        ],
    }


def _pack_result(
    *,
    example: EvalExample,
    system: SystemName,
    repetition: int,
    graph: GeneratedGraph,
    scores: GraphQualityScores | None,
    total_ms: float,
    llm_ms: float,
    meta: dict[str, Any],
) -> SystemExampleResult:
    det_ms = max(0.0, total_ms - llm_ms)
    failures = list(scores.failures) if scores else []
    if graph.error_category and graph.error_category not in failures:
        failures = [graph.error_category, *failures]
    return SystemExampleResult(
        example_id=example.id,
        system=system,
        repetition=repetition,
        scores=scores,
        graph=graph,
        total_latency_ms=total_ms,
        llm_latency_ms=llm_ms,
        deterministic_latency_ms=det_ms,
        cost_usd=meta.get("cost_usd"),
        cost_estimated=bool(meta.get("tokens_estimated", True)),
        input_tokens=meta.get("input_tokens"),
        output_tokens=meta.get("output_tokens"),
        failures=failures,
    )


def _provider_seed_supported() -> bool:
    """True when the active provider accepts a seed parameter (not a guarantee of bit-identity)."""
    try:
        from app.services.llm import _get_provider

        name = (getattr(_get_provider(), "provider_name", None) or "").strip().lower()
        return name in {"openai", "gemini", "openai_compatible"}
    except Exception:
        return False


def _attach_generation_meta(
    graph: GeneratedGraph,
    *,
    seed_used: int | None,
    seed_supported: bool,
    generation_index: int,
    prompt_variant: str | None,
) -> None:
    meta = dict(graph.generation_meta or {})
    meta["seed"] = seed_used
    meta["seed_supported"] = seed_supported
    meta["generation_index"] = generation_index
    meta["prompt_variant"] = prompt_variant or "baseline"
    graph.generation_meta = meta


async def evaluate_example(
    example: EvalExample,
    *,
    systems: Sequence[SystemName],
    repetition: int,
    temperature: float,
    seed: int | None,
    prompt_variant: str | None = None,
    edge_classifier_prompt_variant: str | None = None,
) -> dict[SystemName, SystemExampleResult]:
    """Run selected systems for one example.

    Direct and Synapse share one joint LLM graph call. Concept-First always uses its
    own staged LLM calls and never shares that baseline call.
    """
    out: dict[SystemName, SystemExampleResult] = {}
    seed_supported = _provider_seed_supported() and seed is not None

    if "linear_baseline" in systems:
        seed_used = None if seed is None else seed + 1000 + repetition
        t0 = time.perf_counter()
        graph, meta = await run_linear_baseline(
            example,
            temperature=temperature,
            seed=seed_used,
        )
        total_ms = (time.perf_counter() - t0) * 1000.0
        _attach_generation_meta(
            graph,
            seed_used=seed_used,
            seed_supported=seed_supported,
            generation_index=repetition,
            prompt_variant=prompt_variant,
        )
        scores = score_graph(example, graph) if graph.parse_ok else None
        out["linear_baseline"] = _pack_result(
            example=example,
            system="linear_baseline",
            repetition=repetition,
            graph=graph,
            scores=scores,
            total_ms=total_ms,
            llm_ms=float(meta.get("llm_latency_ms") or 0.0),
            meta=meta,
        )

    if "concept_first" in systems:
        seed_used = None if seed is None else seed + 3000 + repetition
        t0 = time.perf_counter()
        graph, meta = await run_concept_first(
            example,
            temperature=temperature,
            seed=seed_used,
        )
        total_ms = (time.perf_counter() - t0) * 1000.0
        _attach_generation_meta(
            graph,
            seed_used=seed_used,
            seed_supported=seed_supported,
            generation_index=repetition,
            prompt_variant=prompt_variant,
        )
        scores = score_graph(example, graph) if graph.parse_ok else None
        out["concept_first"] = _pack_result(
            example=example,
            system="concept_first",
            repetition=repetition,
            graph=graph,
            scores=scores,
            total_ms=total_ms,
            llm_ms=float(meta.get("llm_latency_ms") or 0.0),
            meta=meta,
        )

    if "domain_curriculum_prior" in systems:
        from app.evaluation.curriculum_prior_system import run_domain_curriculum_prior

        seed_used = None if seed is None else seed + 5000 + repetition
        t0 = time.perf_counter()
        graph, meta = await run_domain_curriculum_prior(
            example,
            temperature=temperature,
            seed=seed_used,
        )
        total_ms = (time.perf_counter() - t0) * 1000.0
        _attach_generation_meta(
            graph,
            seed_used=seed_used,
            seed_supported=seed_supported,
            generation_index=repetition,
            prompt_variant=prompt_variant,
        )
        scores = score_graph(example, graph) if graph.parse_ok else None
        out["domain_curriculum_prior"] = _pack_result(
            example=example,
            system="domain_curriculum_prior",  # type: ignore[arg-type]
            repetition=repetition,
            graph=graph,
            scores=scores,
            total_ms=total_ms,
            llm_ms=float(meta.get("llm_latency_ms") or 0.0),
            meta=meta,
        )

    if "domain_prior_edge_classifier" in systems:
        from app.evaluation.edge_classifier_system import run_domain_prior_edge_classifier

        seed_used = None if seed is None else seed + 7000 + repetition
        t0 = time.perf_counter()
        graph, meta = await run_domain_prior_edge_classifier(
            example,
            temperature=temperature,
            seed=seed_used,
            prompt_variant=edge_classifier_prompt_variant,
        )
        total_ms = (time.perf_counter() - t0) * 1000.0
        _attach_generation_meta(
            graph,
            seed_used=seed_used,
            seed_supported=seed_supported,
            generation_index=repetition,
            prompt_variant=prompt_variant,
        )
        # Preserve edge-classifier prompt identity (ingest prompt_variant is unrelated).
        gmeta = dict(graph.generation_meta or {})
        if meta.get("edge_classifier_prompt_variant") or meta.get("prompt_variant"):
            ecv = meta.get("edge_classifier_prompt_variant") or meta.get("prompt_variant")
            gmeta["prompt_variant"] = ecv
            gmeta["edge_classifier_prompt_variant"] = ecv
            if meta.get("prompt_version"):
                gmeta["prompt_version"] = meta["prompt_version"]
            if meta.get("prompt_hash"):
                gmeta["prompt_hash"] = meta["prompt_hash"]
            graph.generation_meta = gmeta
        scores = score_graph(example, graph) if graph.parse_ok else None
        out["domain_prior_edge_classifier"] = _pack_result(
            example=example,
            system="domain_prior_edge_classifier",  # type: ignore[arg-type]
            repetition=repetition,
            graph=graph,
            scores=scores,
            total_ms=total_ms,
            llm_ms=float(meta.get("llm_latency_ms") or 0.0),
            meta=meta,
        )

    need_graph = (
        "direct_llm_graph" in systems
        or "synapse" in systems
        or "baseline_coverage_recovery" in systems
    )
    if not need_graph:
        return out

    seed_used = None if seed is None else seed + repetition
    try:
        shared_raw, shared_meta = await generate_direct_llm_raw(
            example,
            temperature=temperature,
            seed=seed_used,
            prompt_variant=prompt_variant,
        )
    except Exception as exc:
        from app.evaluation.failure_analysis import classify_llm_exception

        category = classify_llm_exception(exc)
        failed = GeneratedGraph(
            topics=[],
            dependencies=[],
            parse_ok=False,
            error=str(exc),
            error_category=category,
        )
        _attach_generation_meta(
            failed,
            seed_used=seed_used,
            seed_supported=seed_supported,
            generation_index=repetition,
            prompt_variant=prompt_variant,
        )
        meta = {"llm_latency_ms": 0.0, "cost_usd": None, "tokens_estimated": True}
        for sys in ("direct_llm_graph", "synapse", "baseline_coverage_recovery"):
            if sys in systems:
                out[sys] = _pack_result(  # type: ignore[arg-type]
                    example=example,
                    system=sys,  # type: ignore[arg-type]
                    repetition=repetition,
                    graph=failed,
                    scores=None,
                    total_ms=0.0,
                    llm_ms=0.0,
                    meta=meta,
                )
        return out

    llm_ms = float(shared_meta.get("llm_latency_ms") or 0.0)

    if "direct_llm_graph" in systems:
        t_det = time.perf_counter()
        graph = run_direct_from_raw(shared_raw)
        det_ms = (time.perf_counter() - t_det) * 1000.0
        _attach_generation_meta(
            graph,
            seed_used=seed_used,
            seed_supported=seed_supported,
            generation_index=repetition,
            prompt_variant=prompt_variant,
        )
        scores = score_graph(example, graph) if graph.parse_ok else None
        out["direct_llm_graph"] = _pack_result(
            example=example,
            system="direct_llm_graph",
            repetition=repetition,
            graph=graph,
            scores=scores,
            total_ms=llm_ms + det_ms,
            llm_ms=llm_ms,
            meta=shared_meta,
        )

    baseline_synapse_graph = None
    if "synapse" in systems or "baseline_coverage_recovery" in systems:
        t_det = time.perf_counter()
        baseline_synapse_graph = run_synapse_from_raw(shared_raw)
        det_ms = (time.perf_counter() - t_det) * 1000.0
        if "synapse" in systems:
            graph = baseline_synapse_graph
            _attach_generation_meta(
                graph,
                seed_used=seed_used,
                seed_supported=seed_supported,
                generation_index=repetition,
                prompt_variant=prompt_variant,
            )
            scores = score_graph(example, graph) if graph.parse_ok else None
            out["synapse"] = _pack_result(
                example=example,
                system="synapse",
                repetition=repetition,
                graph=graph,
                scores=scores,
                total_ms=llm_ms + det_ms,
                llm_ms=llm_ms,
                meta=shared_meta,
            )

    if "baseline_coverage_recovery" in systems:
        from app.evaluation.coverage_recovery_system import run_baseline_coverage_recovery_from_raw

        t0 = time.perf_counter()
        graph, meta = await run_baseline_coverage_recovery_from_raw(
            shared_raw,
            learning_objective=example.goal,
            temperature=temperature,
            seed=seed_used,
            baseline_meta=shared_meta,
        )
        total_ms = (time.perf_counter() - t0) * 1000.0
        # Prefer measured combined LLM latency from meta when present
        cov_llm_ms = float(meta.get("llm_latency_ms") or llm_ms)
        _attach_generation_meta(
            graph,
            seed_used=seed_used,
            seed_supported=seed_supported,
            generation_index=repetition,
            prompt_variant=prompt_variant,
        )
        # Preserve recovery fields attached by the adapter
        if graph.generation_meta is None:
            graph.generation_meta = {}
        graph.generation_meta.update({k: v for k, v in meta.items() if k not in graph.generation_meta or k.startswith("recovery_") or k == "generation_strategy"})
        scores = score_graph(example, graph) if graph.parse_ok else None
        out["baseline_coverage_recovery"] = _pack_result(
            example=example,
            system="baseline_coverage_recovery",  # type: ignore[arg-type]
            repetition=repetition,
            graph=graph,
            scores=scores,
            total_ms=total_ms,
            llm_ms=cov_llm_ms,
            meta=meta,
        )

    return out


async def run_operation_latency_suite(
    examples: Sequence[EvalExample],
    *,
    samples_per_op: int = 5,
    temperature: float = 0.0,
    seed: int | None = 42,
) -> dict[str, Any]:
    """Latency for ingest-shaped / expand / audit / reshape / quiz / ask prompts.

    Uses golden examples as stand-ins for graph content so the live user DB is untouched.
    Measures LLM latency separately from local post-processing where separable.
    """
    if samples_per_op <= 0 or not examples:
        return {}

    ops: dict[str, list[dict[str, float]]] = {
        "ingest": [],
        "expand": [],
        "audit": [],
        "reshape": [],
        "quiz": [],
        "ask": [],
    }

    for i in range(samples_per_op):
        ex = examples[i % len(examples)]
        topics = [{"title": t, "summary": ex.gold_topic_summaries.get(t, f"Overview of {t}.")} for t in ex.gold_topics]
        edges = list(ex.gold_dependencies)
        leaf = ex.gold_topics[-1]
        leaf_summary = ex.gold_topic_summaries.get(leaf, f"Overview of {leaf}.")
        prereqs = [b for a, b in edges if a == leaf]

        # ingest (graph JSON generation — same prompt as production ingest)
        from app.evaluation.baselines import build_source_text
        from app.prompts.ingest import build_ingest_prompt

        prompt = build_ingest_prompt(build_source_text(ex), known_topic_titles=[])
        t0 = time.perf_counter()
        with capture_llm_calls() as records:
            with llm_operation("ingest"):
                await call_llm_detailed(prompt, temperature=temperature, seed=None if seed is None else seed + i)
        total = (time.perf_counter() - t0) * 1000.0
        llm_ms = records[0].latency_ms if records else total
        ops["ingest"].append({"total_ms": total, "llm_ms": llm_ms, "deterministic_ms": max(0.0, total - llm_ms)})

        # expand
        prompt = build_expand_prompt(leaf, leaf_summary, prereqs, None)
        t0 = time.perf_counter()
        with capture_llm_calls() as records:
            with llm_operation("expand"):
                await call_llm_detailed(prompt, temperature=temperature, seed=None if seed is None else seed + 50 + i)
        total = (time.perf_counter() - t0) * 1000.0
        llm_ms = records[0].latency_ms if records else total
        ops["expand"].append({"total_ms": total, "llm_ms": llm_ms, "deterministic_ms": max(0.0, total - llm_ms)})

        # audit
        prompt = build_audit_prompt(topics, edges)
        t0 = time.perf_counter()
        with capture_llm_calls() as records:
            with llm_operation("audit"):
                await call_llm_detailed(prompt, temperature=temperature, seed=None if seed is None else seed + 100 + i)
        total = (time.perf_counter() - t0) * 1000.0
        llm_ms = records[0].latency_ms if records else total
        ops["audit"].append({"total_ms": total, "llm_ms": llm_ms, "deterministic_ms": max(0.0, total - llm_ms)})

        # reshape
        prompt = build_reshape_prompt(topics, edges, [], None)
        t0 = time.perf_counter()
        with capture_llm_calls() as records:
            with llm_operation("reshape"):
                await call_llm_detailed(prompt, temperature=temperature, seed=None if seed is None else seed + 150 + i)
        total = (time.perf_counter() - t0) * 1000.0
        llm_ms = records[0].latency_ms if records else total
        ops["reshape"].append({"total_ms": total, "llm_ms": llm_ms, "deterministic_ms": max(0.0, total - llm_ms)})

        # quiz
        prompt = build_quiz_prompt(leaf, leaf_summary, [])
        t0 = time.perf_counter()
        with capture_llm_calls() as records:
            with llm_operation("quiz"):
                await call_llm_detailed(prompt, temperature=temperature, seed=None if seed is None else seed + 200 + i)
        total = (time.perf_counter() - t0) * 1000.0
        llm_ms = records[0].latency_ms if records else total
        ops["quiz"].append({"total_ms": total, "llm_ms": llm_ms, "deterministic_ms": max(0.0, total - llm_ms)})

        # ask (scoped Q&A)
        prompt = build_ask_prompt(
            topic_title=leaf,
            topic_summary=leaf_summary,
            resources=[],
            artifacts=[],
            question=f"What should I learn before studying {leaf}?",
            history=None,
        )
        t0 = time.perf_counter()
        with capture_llm_calls() as records:
            with llm_operation("ask"):
                await call_llm_detailed(prompt, temperature=temperature, seed=None if seed is None else seed + 250 + i)
        total = (time.perf_counter() - t0) * 1000.0
        llm_ms = records[0].latency_ms if records else total
        ops["ask"].append({"total_ms": total, "llm_ms": llm_ms, "deterministic_ms": max(0.0, total - llm_ms)})

    summary: dict[str, Any] = {}
    for op, samples in ops.items():
        summary[op] = {
            **summarize_latencies_ms([s["total_ms"] for s in samples]),
            "llm": summarize_latencies_ms([s["llm_ms"] for s in samples]),
            "deterministic": summarize_latencies_ms([s["deterministic_ms"] for s in samples]),
        }
    return summary


async def run_benchmark(
    examples: Sequence[EvalExample],
    *,
    systems: Sequence[SystemName] = ("linear_baseline", "direct_llm_graph", "synapse"),
    repetitions: int = 1,
    temperature: float = 0.0,
    seed: int | None = 42,
    include_ops_latency: bool = True,
    ops_latency_samples: int = 5,
    dataset_name: str = "learning_graph_eval_v1",
    model: str | None = None,
    provider: str | None = None,
    prompt_variant: str | None = None,
    edge_classifier_prompt_variant: str | None = None,
    benchmark_type: str = "quality",
) -> dict[str, Any]:
    from app.prompts.ingest import prompt_metadata

    per_system: dict[str, list[SystemExampleResult]] = {s: [] for s in systems}
    resolved_model = model
    resolved_provider = provider
    prompt_meta = prompt_metadata(prompt_variant)

    for rep in range(repetitions):
        for example in examples:
            results = await evaluate_example(
                example,
                systems=systems,
                repetition=rep,
                temperature=temperature,
                seed=seed,
                prompt_variant=prompt_variant,
                edge_classifier_prompt_variant=edge_classifier_prompt_variant,
            )
            for sys, result in results.items():
                per_system[sys].append(result)

    if resolved_model is None or resolved_provider is None:
        for results in per_system.values():
            for r in results:
                if r.input_tokens is not None or r.cost_usd is not None or r.llm_latency_ms > 0:
                    break

    system_blocks = {name: _system_block(results) for name, results in per_system.items()}

    try:
        from app.services.llm import _get_provider

        prov = _get_provider()
        resolved_model = resolved_model or getattr(prov, "model", None) or "unknown"
        resolved_provider = resolved_provider or getattr(prov, "provider_name", None) or "unknown"
    except Exception:
        resolved_model = resolved_model or "unknown"
        resolved_provider = resolved_provider or "unknown"

    all_failures = summarize_failures(
        r.failures for results in per_system.values() for r in results
    )

    operation_latency: dict[str, Any] = {}
    operation_cost: dict[str, Any] = {}
    if include_ops_latency and ops_latency_samples > 0:
        with capture_llm_calls() as op_records:
            operation_latency = await run_operation_latency_suite(
                examples,
                samples_per_op=ops_latency_samples,
                temperature=temperature,
                seed=seed,
            )
        by_op: dict[str, list[float]] = {}
        for rec in op_records:
            if rec.operation and rec.estimated_cost_usd is not None:
                by_op.setdefault(rec.operation, []).append(rec.estimated_cost_usd)
        operation_cost = {
            op: {
                "average_cost_usd": (sum(vals) / len(vals)) if vals else None,
                "samples": len(vals),
            }
            for op, vals in by_op.items()
        }

    notes = [
        "Graph validity for Synapse uses production build_topics_and_dependencies.",
        "Topic matching is deterministic (normalize, aliases, token containment / Jaccard); no LLM judge.",
        "Proposal metrics reflect real event-log data only; empty means nothing recorded yet.",
        "The benchmark measures agreement with curated reference structures and does not claim that there is only one universally correct learning graph.",
        f"Ingest prompt variant: {prompt_meta['prompt_variant']} ({prompt_meta['prompt_version']}).",
    ]
    if repetitions > 1:
        notes.append(
            f"Multi-generation run: {repetitions} independent generations per case "
            f"(base seed={seed}; per-gen seed=base+repetition when supported). "
            "Per-generation rows are stored independently; measurement only (not best-of-N)."
        )

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmark_type": benchmark_type,
        "dataset": dataset_name,
        "dataset_version": dataset_name,
        "model": resolved_model,
        "provider": resolved_provider,
        "prompt_variant": prompt_meta["prompt_variant"],
        "prompt_version": prompt_meta["prompt_version"],
        "prompt_hash": prompt_meta["prompt_hash"],
        "example_count": len(examples),
        "repetitions": repetitions,
        "generations": repetitions,
        "temperature": temperature,
        "seed": seed,
        "seed_supported": _provider_seed_supported() and seed is not None,
        "systems": system_blocks,
        "metrics": {name: block.get("metrics") for name, block in system_blocks.items()},
        "latency": {name: block.get("latency") for name, block in system_blocks.items()},
        "cost": {
            "pricing": pricing_metadata(),
            "by_system": {name: block.get("cost") for name, block in system_blocks.items()},
            "by_operation": operation_cost,
            "note": (
                "Direct and Synapse share one graph-JSON LLM call per example; "
                "their quality differs only by deterministic validation. "
                "Linear uses a separate roadmap call."
            ),
        },
        "failures": all_failures,
        "operation_latency": operation_latency,
        "proposal_metrics": collect_proposal_metrics(),
        "notes": notes,
    }
