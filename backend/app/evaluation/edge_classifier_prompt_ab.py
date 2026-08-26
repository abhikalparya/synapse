"""Shared-selection A/B: edge_classifier_baseline vs fewshot_directness."""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.curriculum.inventory import load_case_domain_map, resolve_domain_for_case
from app.evaluation.baselines import build_source_text
from app.evaluation.cost import estimate_cost_usd
from app.evaluation.dataset import filter_examples, load_dataset
from app.evaluation.edge_ambiguity import adapt_example_for_edge_mode
from app.evaluation.metrics import (
    aggregate_scores,
    find_redundant_transitive_edges,
    normalize_topic,
    score_graph,
)
from app.evaluation.schemas import EvalExample, GeneratedGraph
from app.prompts.domain_prior_edge_classifier import (
    edge_classifier_prompt_hash,
    resolve_edge_classifier_prompt_variant,
)
from app.services.domain_prior_edge_classifier import (
    EdgeClassifierResult,
    classify_with_frozen_selection,
    run_domain_prior_edge_classifier_pipeline,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BENCH = _REPO_ROOT / "results" / "benchmarks"
DEFAULT_FAIL = _REPO_ROOT / "results" / "failure_analysis"

BASELINE_VARIANT = "edge_classifier_baseline"
FEWSHOT_VARIANT = "edge_classifier_fewshot_directness"


def _safe_rate(n: float, d: float) -> float:
    return (n / d) if d else 0.0


def _norm_edge(a: str, b: str) -> tuple[str, str]:
    return (normalize_topic(a), normalize_topic(b))


def _result_to_graph(result: EdgeClassifierResult) -> GeneratedGraph:
    deps = [(d["from"], d["to"]) for d in result.dependencies]
    return GeneratedGraph(
        topics=[t["title"] for t in result.topics],
        dependencies=deps,
        parse_ok=result.parse_ok and bool(result.topics),
        error="; ".join(result.errors) if result.errors else None,
        generation_meta=result.to_meta(),
        raw_response="\n---\n".join(result.classification_raw_batches) or result.selection_raw,
    )


def _pack_case(
    example: EvalExample,
    *,
    variant: str,
    result: EdgeClassifierResult,
    total_ms: float,
) -> dict[str, Any]:
    from dataclasses import asdict

    adapted = adapt_example_for_edge_mode(
        example, "edge_calibrated", topic_matching_mode="curated_alias"
    )
    graph = _result_to_graph(result)
    scores = score_graph(adapted, graph) if graph.parse_ok else None
    meta = result.to_meta()
    return {
        "example_id": example.id,
        "domain": result.domain,
        "prompt_variant": variant,
        "prompt_hash": meta.get("prompt_hash"),
        "prompt_version": meta.get("prompt_version"),
        "parse_ok": graph.parse_ok,
        "scores": asdict(scores) if scores else None,
        "total_latency_ms": total_ms,
        "stage_latency_ms": meta.get("stage_latency_ms"),
        "input_tokens": meta.get("input_tokens"),
        "output_tokens": meta.get("output_tokens"),
        "selection_input_tokens": meta.get("selection_input_tokens"),
        "selection_output_tokens": meta.get("selection_output_tokens"),
        "classification_input_tokens": meta.get("classification_input_tokens"),
        "classification_output_tokens": meta.get("classification_output_tokens"),
        "estimated_cost_usd": meta.get("estimated_cost_usd")
        or estimate_cost_usd(
            meta.get("model") or "gpt-4o-mini",
            int(meta.get("input_tokens") or 0),
            int(meta.get("output_tokens") or 0),
        ),
        "selected_titles": list(result.selected_titles),
        "selected_concept_ids": list(result.selected_ids),
        "candidate_meta": dict(result.candidate_meta),
        "pair_decisions": list(result.pair_decisions),
        "generated_topics": list(graph.topics),
        "generated_dependencies": [list(e) for e in graph.dependencies],
        "new_concept_count": result.new_concept_count,
        "uncertain_count": meta.get("uncertain_count", 0),
        "unknown_id_rate_inputs": meta.get("unknown_id_rate_inputs", 0),
        "rejected_non_candidate_count": meta.get("rejected_non_candidate_count", 0),
        "cycle_rejected_edges": result.cycle_rejected_edges,
        "generation_meta": meta,
    }


def _fn_label(
    edge: tuple[str, str],
    *,
    selected: set[str],
    predicted: set[tuple[str, str]],
    decisions: dict[tuple[str, str], str],
    cand_space: set[tuple[str, str]],
) -> str:
    if edge[0] not in selected or edge[1] not in selected:
        return "NOT_SELECTED"
    if edge not in cand_space:
        return "MISSING_FROM_CANDIDATE_SPACE"
    dec = decisions.get(edge)
    if dec == "UNCERTAIN":
        return "UNCERTAIN"
    if dec == "NOT_REQUIRED" or edge not in predicted:
        return "PREDICTED_NOT_REQUIRED"
    return "PREDICTED_NOT_REQUIRED"


def _fp_label(
    edge: tuple[str, str],
    *,
    gold: set[tuple[str, str]],
    predicted: set[tuple[str, str]],
) -> str:
    if (edge[1], edge[0]) in gold:
        return "WRONG_DIRECTION"
    redundant = {
        _norm_edge(a, b) for a, b in find_redundant_transitive_edges(list(predicted))
    }
    if edge in redundant:
        return "TRANSITIVE_REDUNDANCY"
    return "INVALID_DIRECT_EDGE"


def _analyze_variant_row(example: EvalExample, row: dict[str, Any]) -> dict[str, Any]:
    adapted = adapt_example_for_edge_mode(
        example, "edge_calibrated", topic_matching_mode="curated_alias"
    )
    gold = {_norm_edge(a, b) for a, b in adapted.required_dependencies}
    selected = {normalize_topic(t) for t in row.get("selected_titles") or []}
    predicted = {_norm_edge(a, b) for a, b in (row.get("generated_dependencies") or [])}
    decisions: dict[tuple[str, str], str] = {}
    for d in row.get("pair_decisions") or []:
        key = _norm_edge(d.get("from_title") or "", d.get("to_title") or "")
        decisions[key] = str(d.get("decision") or "").upper()
    cand_space = set(decisions.keys())
    # Also include all directed pairs among selected if decisions incomplete
    for a in selected:
        for b in selected:
            if a != b:
                cand_space.add((a, b))
    gold_in = {e for e in gold if e in cand_space or (e[0] in selected and e[1] in selected)}
    fn = gold_in - predicted
    fp = predicted - gold
    fn_cats = Counter(
        _fn_label(
            e,
            selected=selected,
            predicted=predicted,
            decisions=decisions,
            cand_space=cand_space,
        )
        for e in fn
    )
    fp_cats = Counter(_fp_label(e, gold=gold, predicted=predicted) for e in fp)
    return {
        "gold": gold,
        "gold_in_space": gold_in,
        "predicted": predicted,
        "decisions": decisions,
        "fn": fn,
        "fp": fp,
        "fn_cats": fn_cats,
        "fp_cats": fp_cats,
        "scores": row.get("scores") or {},
    }


async def run_edge_classifier_prompt_ab(
    *,
    dataset_path: str | Path | None = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    seed: int = 42,
    output_dir: str | Path | None = None,
    failure_dir: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    """Live A/B with shared selection. Writes benchmark JSON + analysis JSON/MD."""
    import os

    os.environ["OPENAI_MODEL"] = model
    from app.services.llm import reset_llm_provider

    reset_llm_provider()

    ds = (
        Path(dataset_path)
        if dataset_path
        else _REPO_ROOT / "data" / "eval" / "learning_graph_quality_v1.jsonl"
    )
    case_map = load_case_domain_map()
    examples = filter_examples(load_dataset(ds), ids=set(case_map))
    bench_out = Path(output_dir) if output_dir else DEFAULT_BENCH
    fail_out = Path(failure_dir) if failure_dir else DEFAULT_FAIL
    bench_out.mkdir(parents=True, exist_ok=True)
    fail_out.mkdir(parents=True, exist_ok=True)

    case_rows: list[dict[str, Any]] = []
    baseline_scores = []
    fewshot_scores = []
    domain_scores: dict[str, dict[str, list]] = defaultdict(
        lambda: {"baseline": [], "fewshot": []}
    )
    pair_diffs: list[dict[str, Any]] = []
    fn_base = Counter()
    fn_few = Counter()
    fp_base = Counter()
    fp_few = Counter()
    compiler_cases: list[dict[str, Any]] = []

    for ex in examples:
        domain = resolve_domain_for_case(ex.id)
        source = build_source_text(ex)
        seed_used = seed + 7000

        t0 = time.perf_counter()
        base_result = await run_domain_prior_edge_classifier_pipeline(
            source,
            domain=domain,
            temperature=temperature,
            seed=seed_used,
            prompt_variant=BASELINE_VARIANT,
        )
        base_ms = (time.perf_counter() - t0) * 1000.0
        base_row = _pack_case(ex, variant=BASELINE_VARIANT, result=base_result, total_ms=base_ms)

        t1 = time.perf_counter()
        few_result = await classify_with_frozen_selection(
            source,
            domain=domain,
            selected_ids=list(base_result.selected_ids),
            selection_raw=base_result.selection_raw,
            selection_ms=base_result.timings.selection_ms,
            selection_input_tokens=base_result.selection_input_tokens,
            selection_output_tokens=base_result.selection_output_tokens,
            model=base_result.model,
            provider=base_result.provider,
            temperature=temperature,
            seed=seed_used,
            prompt_variant=FEWSHOT_VARIANT,
        )
        few_ms = (time.perf_counter() - t1) * 1000.0 + base_result.timings.selection_ms
        few_row = _pack_case(ex, variant=FEWSHOT_VARIANT, result=few_result, total_ms=few_ms)

        base_an = _analyze_variant_row(ex, base_row)
        few_an = _analyze_variant_row(ex, few_row)
        fn_base.update(base_an["fn_cats"])
        fn_few.update(few_an["fn_cats"])
        fp_base.update(base_an["fp_cats"])
        fp_few.update(few_an["fp_cats"])

        if base_row.get("scores"):
            # rebuild score object-like via score_graph already stored as dict
            from app.evaluation.schemas import GraphQualityScores

            bs = GraphQualityScores(**{k: v for k, v in base_row["scores"].items() if k in GraphQualityScores.__dataclass_fields__})
            baseline_scores.append(bs)
            domain_scores[domain]["baseline"].append(bs)
        if few_row.get("scores"):
            from app.evaluation.schemas import GraphQualityScores

            fs = GraphQualityScores(**{k: v for k, v in few_row["scores"].items() if k in GraphQualityScores.__dataclass_fields__})
            fewshot_scores.append(fs)
            domain_scores[domain]["fewshot"].append(fs)

        # Pair-level diffs
        all_keys = set(base_an["decisions"]) | set(few_an["decisions"]) | base_an["gold_in_space"]
        for key in sorted(all_keys):
            bd = base_an["decisions"].get(key, "MISSING")
            fd = few_an["decisions"].get(key, "MISSING")
            in_gold = key in base_an["gold"]
            gold_status = "REQUIRED" if in_gold else "NOT_GOLD"
            before = None
            after = None
            if in_gold and key not in base_an["predicted"]:
                before = _fn_label(
                    key,
                    selected={normalize_topic(t) for t in base_row["selected_titles"]},
                    predicted=base_an["predicted"],
                    decisions=base_an["decisions"],
                    cand_space=set(base_an["decisions"]) | set(few_an["decisions"]),
                )
            if in_gold and key not in few_an["predicted"]:
                after = _fn_label(
                    key,
                    selected={normalize_topic(t) for t in few_row["selected_titles"]},
                    predicted=few_an["predicted"],
                    decisions=few_an["decisions"],
                    cand_space=set(base_an["decisions"]) | set(few_an["decisions"]),
                )
            if not in_gold and key in base_an["predicted"]:
                before = _fp_label(key, gold=base_an["gold"], predicted=base_an["predicted"])
            if not in_gold and key in few_an["predicted"]:
                after = _fp_label(key, gold=few_an["gold"], predicted=few_an["predicted"])
            if bd != fd or before or after:
                pair_diffs.append(
                    {
                        "case_id": ex.id,
                        "from_id": key[0],
                        "to_id": key[1],
                        "baseline_decision": bd,
                        "fewshot_decision": fd,
                        "gold_status": gold_status,
                        "failure_category_before": before,
                        "failure_category_after": after,
                    }
                )

        case_entry = {
            "example_id": ex.id,
            "domain": domain,
            "learning_goal": ex.goal,
            "selected_concepts": base_row["selected_titles"],
            "candidate_pair_count": (base_row.get("candidate_meta") or {}).get(
                "candidate_pairs_evaluated"
            ),
            "baseline": base_row,
            "fewshot": few_row,
            "required_edge_recall": {
                "baseline": (base_row.get("scores") or {}).get("required_edge_recall"),
                "fewshot": (few_row.get("scores") or {}).get("required_edge_recall"),
            },
            "required_edge_f1": {
                "baseline": (base_row.get("scores") or {}).get("required_edge_f1"),
                "fewshot": (few_row.get("scores") or {}).get("required_edge_f1"),
            },
            "fn_before": sorted(base_an["fn"]),
            "fn_after": sorted(few_an["fn"]),
            "transitive_before": base_an["fp_cats"].get("TRANSITIVE_REDUNDANCY", 0),
            "transitive_after": few_an["fp_cats"].get("TRANSITIVE_REDUNDANCY", 0),
            "predicted_not_required_before": base_an["fn_cats"].get(
                "PREDICTED_NOT_REQUIRED", 0
            ),
            "predicted_not_required_after": few_an["fn_cats"].get(
                "PREDICTED_NOT_REQUIRED", 0
            ),
        }
        case_rows.append(case_entry)
        if domain == "compiler_construction":
            compiler_cases.append(case_entry)

    base_agg = aggregate_scores(baseline_scores) if baseline_scores else {}
    few_agg = aggregate_scores(fewshot_scores) if fewshot_scores else {}

    def delta(key: str) -> float | None:
        if key not in base_agg or key not in few_agg:
            return None
        return float(few_agg[key]) - float(base_agg[key])

    metric_keys = [
        "required_edge_precision",
        "required_edge_recall",
        "required_edge_f1",
        "missing_required_edge_rate",
        "invalid_extra_edge_rate",
        "redundant_transitive_edge_rate",
        "dependency_direction_error_rate",
        "topic_f1",
        "hallucinated_topic_rate",
    ]
    deltas = {k: delta(k) for k in metric_keys}

    # Compiler verdict
    if not compiler_cases:
        compiler_verdict = "DO_NOT_CHANGE_COMPILER"
    else:
        br = sum(
            (c["required_edge_recall"]["baseline"] or 0) for c in compiler_cases
        ) / len(compiler_cases)
        fr = sum(
            (c["required_edge_recall"]["fewshot"] or 0) for c in compiler_cases
        ) / len(compiler_cases)
        if fr > br + 0.02:
            compiler_verdict = "IMPROVE_COMPILER"
        elif fr < br - 0.02:
            compiler_verdict = "REGRESS_COMPILER"
        else:
            compiler_verdict = "DO_NOT_CHANGE_COMPILER"

    pn_before = fn_base.get("PREDICTED_NOT_REQUIRED", 0)
    pn_after = fn_few.get("PREDICTED_NOT_REQUIRED", 0)
    tr_before = fp_base.get("TRANSITIVE_REDUNDANCY", 0)
    tr_after = fp_few.get("TRANSITIVE_REDUNDANCY", 0)
    recall_d = deltas.get("required_edge_recall") or 0.0
    prec_d = deltas.get("required_edge_precision") or 0.0
    f1_d = deltas.get("required_edge_f1") or 0.0

    def avg_num(vals: list[float]) -> float | None:
        return (sum(vals) / len(vals)) if vals else None

    def stage_avg(rows: list[dict], stage: str) -> float | None:
        vals = []
        for r in rows:
            v = (r.get("stage_latency_ms") or {}).get(stage)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        return avg_num(vals)

    base_rows = [c["baseline"] for c in case_rows]
    few_rows = [c["fewshot"] for c in case_rows]
    cost_latency = {
        "baseline_avg_total_ms": avg_num(
            [float(r["total_latency_ms"]) for r in base_rows if r.get("total_latency_ms") is not None]
        ),
        "fewshot_avg_total_ms": avg_num(
            [float(r["total_latency_ms"]) for r in few_rows if r.get("total_latency_ms") is not None]
        ),
        "baseline_avg_selection_ms": stage_avg(base_rows, "selection"),
        "baseline_avg_classification_ms": stage_avg(base_rows, "edge_classification"),
        "fewshot_avg_classification_ms": stage_avg(few_rows, "edge_classification"),
        "baseline_avg_estimated_cost_usd": avg_num(
            [float(r["estimated_cost_usd"]) for r in base_rows if r.get("estimated_cost_usd") is not None]
        ),
        "fewshot_avg_estimated_cost_usd": avg_num(
            [float(r["estimated_cost_usd"]) for r in few_rows if r.get("estimated_cost_usd") is not None]
        ),
        "avg_additional_classification_input_tokens": (
            avg_num([float(r.get("classification_input_tokens") or 0) for r in few_rows]) or 0
        )
        - (avg_num([float(r.get("classification_input_tokens") or 0) for r in base_rows]) or 0),
        "avg_additional_estimated_cost_usd": (
            avg_num(
                [float(r["estimated_cost_usd"]) for r in few_rows if r.get("estimated_cost_usd") is not None]
            )
            or 0
        )
        - (
            avg_num(
                [float(r["estimated_cost_usd"]) for r in base_rows if r.get("estimated_cost_usd") is not None]
            )
            or 0
        ),
        "per_case_cost": [
            {
                "example_id": c["example_id"],
                "baseline_estimated_cost_usd": c["baseline"].get("estimated_cost_usd"),
                "fewshot_estimated_cost_usd": c["fewshot"].get("estimated_cost_usd"),
                "baseline_classification_input_tokens": c["baseline"].get(
                    "classification_input_tokens"
                ),
                "fewshot_classification_input_tokens": c["fewshot"].get(
                    "classification_input_tokens"
                ),
                "baseline_total_ms": c["baseline"].get("total_latency_ms"),
                "fewshot_total_ms": c["fewshot"].get("total_latency_ms"),
            }
            for c in case_rows
        ],
    }

    pn_improved = pn_after < pn_before
    tr_improved = tr_after < tr_before
    precision_ok = prec_d >= -0.03
    recall_ok = recall_d >= -0.02
    cost_ok = (cost_latency["avg_additional_estimated_cost_usd"] or 0) < 0.001
    if (
        pn_improved
        and tr_improved
        and precision_ok
        and recall_ok
        and (recall_d > 0.02 or f1_d > 0.02 or (pn_before - pn_after) >= 3)
        and cost_ok
        and compiler_verdict != "REGRESS_COMPILER"
    ):
        diagnosis = "SUPPORTED"
    elif (pn_improved or tr_improved) and precision_ok and recall_ok:
        diagnosis = "PARTIALLY_SUPPORTED"
    else:
        diagnosis = "NOT_SUPPORTED"

    domain_summary = {}
    for domain, buckets in domain_scores.items():
        domain_summary[domain] = {
            "baseline": aggregate_scores(buckets["baseline"]) if buckets["baseline"] else {},
            "fewshot": aggregate_scores(buckets["fewshot"]) if buckets["fewshot"] else {},
        }

    safety = {
        "new_concept_count_total": sum(
            int(c["baseline"].get("new_concept_count") or 0)
            + int(c["fewshot"].get("new_concept_count") or 0)
            for c in case_rows
        ),
        "unknown_id_total": sum(
            int(c["baseline"].get("unknown_id_rate_inputs") or 0)
            + int(c["fewshot"].get("unknown_id_rate_inputs") or 0)
            for c in case_rows
        ),
        "invalid_pair_total": sum(
            int(c["baseline"].get("rejected_non_candidate_count") or 0)
            + int(c["fewshot"].get("rejected_non_candidate_count") or 0)
            for c in case_rows
        ),
        "selection_identical": all(
            c["baseline"]["selected_concept_ids"] == c["fewshot"]["selected_concept_ids"]
            for c in case_rows
        ),
    }

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    report = {
        "timestamp": ts,
        "benchmark_type": "edge_classifier_prompt_ab",
        "model": model,
        "temperature": temperature,
        "seed": seed,
        "cases": [c["example_id"] for c in case_rows],
        "prompt_variants": {
            "baseline": {
                "name": BASELINE_VARIANT,
                "hash": edge_classifier_prompt_hash(BASELINE_VARIANT),
            },
            "fewshot": {
                "name": FEWSHOT_VARIANT,
                "hash": edge_classifier_prompt_hash(FEWSHOT_VARIANT),
            },
        },
        "matching": "curated_alias + edge_calibrated",
        "selection_shared": True,
        "aggregate": {
            "baseline": base_agg,
            "fewshot": few_agg,
            "delta_fewshot_minus_baseline": deltas,
        },
        "failure_metrics": {
            "PREDICTED_NOT_REQUIRED": {"baseline": pn_before, "fewshot": pn_after},
            "TRANSITIVE_REDUNDANCY": {"baseline": tr_before, "fewshot": tr_after},
            "fn_baseline": dict(fn_base),
            "fn_fewshot": dict(fn_few),
            "fp_baseline": dict(fp_base),
            "fp_fewshot": dict(fp_few),
        },
        "domain_summary": domain_summary,
        "compiler_analysis": {
            "verdict": compiler_verdict,
            "cases": compiler_cases,
        },
        "cost_latency": cost_latency,
        "pair_level_differences": pair_diffs,
        "case_results": case_rows,
        "safety": safety,
        "final_diagnosis": diagnosis,
        "next_step": "Proceed to final 40-case benchmark",
    }

    bench_path = bench_out / f"{ts}_edge_classifier_prompt_ab.json"
    json_path = fail_out / f"{ts}_edge_classifier_prompt_ab.json"
    md_path = fail_out / f"{ts}_edge_classifier_prompt_ab.md"
    bench_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    def fmt(v: Any) -> str:
        if v is None:
            return "n/a"
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    lines = [
        f"# Edge Classifier Prompt A/B — {ts}",
        "",
        f"Diagnosis: **{diagnosis}**",
        f"Compiler: **{compiler_verdict}**",
        f"Model: `{model}` · Cases: {len(case_rows)} · Selection shared: yes",
        f"Variants: `{BASELINE_VARIANT}` vs `{FEWSHOT_VARIANT}`",
        "",
        "## Aggregate metrics",
        "",
        "| Metric | Baseline Edge Classifier | Few-Shot Edge Classifier | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for k in metric_keys:
        lines.append(
            f"| {k} | {fmt(base_agg.get(k))} | {fmt(few_agg.get(k))} | {fmt(deltas.get(k))} |"
        )
    lines += [
        "",
        "## Failure metrics",
        "",
        f"- PREDICTED_NOT_REQUIRED: {pn_before} → {pn_after}",
        f"- TRANSITIVE_REDUNDANCY: {tr_before} → {tr_after}",
        "",
        "## Domain Required Edge Recall",
        "",
    ]
    for domain, block in sorted(domain_summary.items()):
        lines.append(
            f"- **{domain}**: baseline=`{fmt(block['baseline'].get('required_edge_recall'))}` "
            f"fewshot=`{fmt(block['fewshot'].get('required_edge_recall'))}`"
        )
    lines += [
        "",
        "## Cost / latency",
        "",
        f"- Baseline avg total ms: `{fmt(cost_latency['baseline_avg_total_ms'])}`",
        f"- Few-shot avg total ms: `{fmt(cost_latency['fewshot_avg_total_ms'])}`",
        f"- Baseline avg est. cost: `{fmt(cost_latency['baseline_avg_estimated_cost_usd'])}`",
        f"- Few-shot avg est. cost: `{fmt(cost_latency['fewshot_avg_estimated_cost_usd'])}`",
        f"- Avg additional classification input tokens: "
        f"`{fmt(cost_latency['avg_additional_classification_input_tokens'])}`",
        f"- Avg additional estimated cost: `{fmt(cost_latency['avg_additional_estimated_cost_usd'])}`",
        "",
        "## Compiler-specific",
        "",
    ]
    for c in compiler_cases:
        lines += [
            f"### {c['example_id']} — recall "
            f"{fmt(c['required_edge_recall']['baseline'])} → "
            f"{fmt(c['required_edge_recall']['fewshot'])}",
            f"- Selected: {', '.join(c['selected_concepts'])}",
            f"- Candidate pairs: {c['candidate_pair_count']}",
            f"- FN before/after: {c['fn_before']} / {c['fn_after']}",
            f"- Transitive extras before/after: "
            f"{c['transitive_before']} / {c['transitive_after']}",
            f"- PREDICTED_NOT_REQUIRED before/after: "
            f"{c['predicted_not_required_before']} / {c['predicted_not_required_after']}",
            "",
        ]
    lines += [
        "## Safety",
        "",
        f"- NEW_CONCEPT_COUNT total: `{safety['new_concept_count_total']}`",
        f"- Unknown IDs: `{safety['unknown_id_total']}`",
        f"- Invalid pairs: `{safety['invalid_pair_total']}`",
        f"- Selection identical: `{safety['selection_identical']}`",
        "",
        "## Next step",
        "",
        "Proceed to final 40-case benchmark",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return bench_path, json_path, md_path
