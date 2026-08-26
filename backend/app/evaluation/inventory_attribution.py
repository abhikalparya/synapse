"""Stage-1 inventory quality + causal graph-error attribution (evaluation only).

Distinguishes Concept-First inventory failure from Stage-2 relationship failure using
stored benchmark artifacts. No new LLM calls when Stage-1 inventory fields are present.

Reuses curated_alias matching, edge_calibrated gold edges, and node/topic classifiers.
Does not change scores, aliases, gold, prompts, or production generation.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from app.evaluation.dataset import load_dataset
from app.evaluation.edge_ambiguity import adapt_example_for_edge_mode
from app.evaluation.inspect import _graph_from_row
from app.evaluation.metrics import compare_graphs, normalize_topic, score_graph
from app.evaluation.node_edge_attribution import (
    classify_generated_topic,
    classify_gold_topic_representation,
)
from app.evaluation.schemas import EvalExample, GeneratedGraph

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = _REPO_ROOT / "results" / "failure_analysis"

InventoryClass = Literal[
    "EXACT",
    "ALIAS",
    "GRANULARITY",
    "DECOMPOSITION",
    "ABSTRACTION",
    "RELATED_BUT_DISTINCT",
    "OUT_OF_SCOPE",
    "HALLUCINATION",
    "UNKNOWN",
]

MISSING_CAUSAL = (
    "SOURCE_ABSENT_FROM_INVENTORY",
    "TARGET_ABSENT_FROM_INVENTORY",
    "BOTH_ENDPOINTS_ABSENT_FROM_INVENTORY",
    "ENDPOINT_GRANULARITY_MISMATCH",
    "ENDPOINT_ABSTRACTION_MISMATCH",
    "ENDPOINT_DECOMPOSITION",
    "BOTH_ENDPOINTS_AVAILABLE_EDGE_OMITTED",
    "UNKNOWN",
)

INVALID_CAUSAL = (
    "SOURCE_OUT_OF_SCOPE_INVENTORY",
    "TARGET_OUT_OF_SCOPE_INVENTORY",
    "BOTH_ENDPOINTS_OUT_OF_SCOPE_INVENTORY",
    "ENDPOINT_GRANULARITY_DRIFT",
    "ENDPOINT_ABSTRACTION_DRIFT",
    "ENDPOINT_DECOMPOSITION_DRIFT",
    "BOTH_ENDPOINTS_VALID_EDGE_INVALID",
    "CURRICULUM_SCOPE_DRIFT",
    "UNKNOWN",
)

_PRESENT = frozenset({"EXACT_MATCH", "ALIAS_MATCH"})
_ABSENT = frozenset({"MISSING"})
_GEN_VALID = frozenset({"MATCHED_GOLD_TOPIC", "ALIAS_OF_GOLD_TOPIC"})
_GEN_OOS = frozenset({"OUT_OF_SCOPE", "GENUINE_HALLUCINATION", "RELATED_BUT_DISTINCT", "UNKNOWN"})
_GEN_OPTIONALISH = frozenset({"ABSTRACTION_VARIANT", "RELATED_BUT_DISTINCT", "OUT_OF_SCOPE"})

_GEN_TO_INV: dict[str, InventoryClass] = {
    "MATCHED_GOLD_TOPIC": "EXACT",
    "ALIAS_OF_GOLD_TOPIC": "ALIAS",
    "GRANULARITY_VARIANT": "GRANULARITY",
    "DECOMPOSITION_COMPONENT": "DECOMPOSITION",
    "ABSTRACTION_VARIANT": "ABSTRACTION",
    "RELATED_BUT_DISTINCT": "RELATED_BUT_DISTINCT",
    "OUT_OF_SCOPE": "OUT_OF_SCOPE",
    "GENUINE_HALLUCINATION": "HALLUCINATION",
    "UNKNOWN": "UNKNOWN",
}


def _rows_by_id(system_block: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in system_block.get("example_results") or []:
        eid = str(row.get("example_id") or "")
        if eid and int(row.get("repetition") or 0) == 0:
            out[eid] = row
    return out


def _safe_rate(num: float, den: float) -> float:
    return (num / den) if den else 0.0


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def inventory_graph_from_titles(titles: list[str]) -> GeneratedGraph:
    """Inventory-only graph (no dependencies) for independent Stage-1 evaluation."""
    return GeneratedGraph(topics=list(titles), dependencies=[], parse_ok=True)


def extract_stage1_inventory(row: dict[str, Any]) -> list[str]:
    """Prefer Stage-1 candidate titles; fall back to normalized inventory / final topics."""
    meta = row.get("generation_meta") or {}
    candidates = meta.get("candidate_concepts")
    if isinstance(candidates, list) and candidates:
        titles: list[str] = []
        for c in candidates:
            if isinstance(c, dict):
                t = str(c.get("title") or "").strip()
            else:
                t = str(c).strip()
            if t:
                titles.append(t)
        if titles:
            return titles
    norm = meta.get("normalized_inventory")
    if isinstance(norm, list) and norm:
        return [str(t).strip() for t in norm if str(t).strip()]
    return [str(t).strip() for t in (row.get("generated_topics") or []) if str(t).strip()]


def extract_normalized_inventory(row: dict[str, Any]) -> list[str]:
    meta = row.get("generation_meta") or {}
    norm = meta.get("normalized_inventory")
    if isinstance(norm, list) and norm:
        return [str(t).strip() for t in norm if str(t).strip()]
    # Normalization rarely changed titles; Stage-1 ≈ normalized when absent.
    return extract_stage1_inventory(row)


def stage1_data_available(row: dict[str, Any]) -> bool:
    meta = row.get("generation_meta") or {}
    return bool(meta.get("candidate_concepts") or meta.get("normalized_inventory"))


def map_generated_status_to_inventory_class(status: str) -> InventoryClass:
    return _GEN_TO_INV.get(status, "UNKNOWN")


def classify_inventory_concepts(
    titles: list[str],
    example: EvalExample,
) -> list[dict[str, Any]]:
    rows = []
    for t in titles:
        st = classify_generated_topic(t, example)
        rows.append(
            {
                "title": t,
                "status": st["status"],
                "inventory_class": map_generated_status_to_inventory_class(st["status"]),
                "canonical": st.get("canonical"),
                "reason": st.get("reason"),
            }
        )
    return rows


def evaluate_inventory(
    titles: list[str],
    example: EvalExample,
) -> dict[str, Any]:
    """Score an inventory independently of dependencies (curated_alias via adapted example)."""
    graph = inventory_graph_from_titles(titles)
    scores = score_graph(example, graph)
    classifications = classify_inventory_concepts(titles, example)
    by_class = Counter(str(c["inventory_class"]) for c in classifications)
    n = len(titles) or 0

    gold_statuses = {
        g: classify_gold_topic_representation(g, example, graph) for g in example.required_topic_list()
    }
    missing_foundational = sum(1 for s in gold_statuses.values() if s["status"] in _ABSENT)
    # Also count related/unknown as not cleanly present for foundational completeness
    missing_or_unmatched = sum(
        1 for s in gold_statuses.values() if s["status"] not in _PRESENT
    )

    return {
        "topics": list(titles),
        "n_topics": len(titles),
        "topic_precision": scores.topic_precision,
        "topic_recall": scores.topic_recall,
        "topic_f1": scores.topic_f1,
        "hallucinated_topic_rate": scores.hallucinated_topic_rate,
        "out_of_scope_rate": _safe_rate(by_class.get("OUT_OF_SCOPE", 0), n),
        "granularity_mismatch_rate": _safe_rate(by_class.get("GRANULARITY", 0), n),
        "abstraction_mismatch_rate": _safe_rate(by_class.get("ABSTRACTION", 0), n),
        "decomposition_rate": _safe_rate(by_class.get("DECOMPOSITION", 0), n),
        "related_but_distinct_rate": _safe_rate(by_class.get("RELATED_BUT_DISTINCT", 0), n),
        "hallucination_rate": _safe_rate(by_class.get("HALLUCINATION", 0), n),
        "missing_foundational_concept_rate": _safe_rate(
            missing_foundational, len(example.required_topic_list())
        ),
        "missing_or_unmatched_foundational_rate": _safe_rate(
            missing_or_unmatched, len(example.required_topic_list())
        ),
        "class_counts": dict(by_class),
        "concept_classifications": classifications,
        "gold_representation": {
            g: {"status": gold_statuses[g]["status"], "candidates": gold_statuses[g].get("candidates")}
            for g in gold_statuses
        },
    }


def gold_endpoint_present(status: str) -> bool:
    return status in _PRESENT


def attribute_missing_edge_vs_inventory(
    frm: str,
    to: str,
    example: EvalExample,
    inventory: GeneratedGraph,
) -> dict[str, Any]:
    """Causal attribution of a missing required edge relative to an inventory."""
    src = classify_gold_topic_representation(frm, example, inventory)
    tgt = classify_gold_topic_representation(to, example, inventory)
    ss, ts = src["status"], tgt["status"]
    sp, tp = gold_endpoint_present(ss), gold_endpoint_present(ts)

    if not sp and not tp and ss in _ABSENT and ts in _ABSENT:
        primary = "BOTH_ENDPOINTS_ABSENT_FROM_INVENTORY"
    elif not sp and ss in _ABSENT and tp:
        primary = "SOURCE_ABSENT_FROM_INVENTORY"
    elif not tp and ts in _ABSENT and sp:
        primary = "TARGET_ABSENT_FROM_INVENTORY"
    elif not sp and not tp:
        # both not exact/alias — prefer representation mismatch over "absent" if not MISSING
        if ss == "GRANULARITY_VARIANT" or ts == "GRANULARITY_VARIANT":
            primary = "ENDPOINT_GRANULARITY_MISMATCH"
        elif ss == "ABSTRACTED" or ts == "ABSTRACTED":
            primary = "ENDPOINT_ABSTRACTION_MISMATCH"
        elif ss == "DECOMPOSED" or ts == "DECOMPOSED":
            primary = "ENDPOINT_DECOMPOSITION"
        elif ss in _ABSENT or ts in _ABSENT:
            primary = "BOTH_ENDPOINTS_ABSENT_FROM_INVENTORY"
        else:
            primary = "UNKNOWN"
    elif not sp and ss in _ABSENT:
        primary = "SOURCE_ABSENT_FROM_INVENTORY"
    elif not tp and ts in _ABSENT:
        primary = "TARGET_ABSENT_FROM_INVENTORY"
    elif ss == "GRANULARITY_VARIANT" or ts == "GRANULARITY_VARIANT":
        primary = "ENDPOINT_GRANULARITY_MISMATCH"
    elif ss == "ABSTRACTED" or ts == "ABSTRACTED":
        primary = "ENDPOINT_ABSTRACTION_MISMATCH"
    elif ss == "DECOMPOSED" or ts == "DECOMPOSED":
        primary = "ENDPOINT_DECOMPOSITION"
    elif sp and tp:
        primary = "BOTH_ENDPOINTS_AVAILABLE_EDGE_OMITTED"
    else:
        primary = "UNKNOWN"

    assert primary in MISSING_CAUSAL
    return {
        "edge": [frm, to],
        "source_status": ss,
        "target_status": ts,
        "primary_attribution": primary,
        "both_endpoints_available": bool(sp and tp),
    }


def attribute_invalid_edge_vs_inventory(
    frm: str,
    to: str,
    example: EvalExample,
) -> dict[str, Any]:
    """Causal attribution of an invalid extra edge relative to Stage-1 concept validity."""
    src = classify_generated_topic(frm, example)
    tgt = classify_generated_topic(to, example)
    ss, ts = src["status"], tgt["status"]

    required_norms = {normalize_topic(t) for t in example.required_topic_list()}
    optional_norms = {normalize_topic(t) for t in example.optional_topic_list()}

    def _bucket(st: dict[str, Any]) -> str:
        can = st.get("canonical")
        if can and normalize_topic(str(can)) in required_norms:
            return "required"
        if can and normalize_topic(str(can)) in optional_norms:
            return "optional"
        return "none"

    sb, tb = _bucket(src), _bucket(tgt)
    src_valid = ss in _GEN_VALID
    tgt_valid = ts in _GEN_VALID
    src_oos = ss in {"OUT_OF_SCOPE", "GENUINE_HALLUCINATION"}
    tgt_oos = ts in {"OUT_OF_SCOPE", "GENUINE_HALLUCINATION"}

    if ss == "GRANULARITY_VARIANT" or ts == "GRANULARITY_VARIANT":
        primary = "ENDPOINT_GRANULARITY_DRIFT"
    elif ss == "ABSTRACTION_VARIANT" or ts == "ABSTRACTION_VARIANT":
        primary = "ENDPOINT_ABSTRACTION_DRIFT"
    elif ss == "DECOMPOSITION_COMPONENT" or ts == "DECOMPOSITION_COMPONENT":
        primary = "ENDPOINT_DECOMPOSITION_DRIFT"
    elif src_oos and tgt_oos:
        primary = "BOTH_ENDPOINTS_OUT_OF_SCOPE_INVENTORY"
    elif src_oos and not tgt_oos:
        primary = "SOURCE_OUT_OF_SCOPE_INVENTORY"
    elif tgt_oos and not src_oos:
        primary = "TARGET_OUT_OF_SCOPE_INVENTORY"
    elif src_valid and tgt_valid:
        if sb == "optional" or tb == "optional":
            primary = "CURRICULUM_SCOPE_DRIFT"
        else:
            primary = "BOTH_ENDPOINTS_VALID_EDGE_INVALID"
    elif ss in _GEN_OPTIONALISH or ts in _GEN_OPTIONALISH:
        primary = "CURRICULUM_SCOPE_DRIFT"
    else:
        primary = "UNKNOWN"

    assert primary in INVALID_CAUSAL
    return {
        "edge": [frm, to],
        "source_status": ss,
        "target_status": ts,
        "primary_attribution": primary,
        "both_endpoints_valid": bool(src_valid and tgt_valid and sb == "required" and tb == "required"),
    }


def edge_opportunity_and_conditional_recall(
    example: EvalExample,
    inventory: GeneratedGraph,
    generated_deps: list[tuple[str, str]],
) -> dict[str, Any]:
    """EDGE_OPPORTUNITY_RATE and CONDITIONAL_EDGE_RECALL for an inventory + dep set."""
    required = example.required_dependency_list()
    opportunity_edges: list[tuple[str, str]] = []
    for frm, to in required:
        src = classify_gold_topic_representation(frm, example, inventory)
        tgt = classify_gold_topic_representation(to, example, inventory)
        if gold_endpoint_present(src["status"]) and gold_endpoint_present(tgt["status"]):
            opportunity_edges.append((frm, to))

    full = GeneratedGraph(topics=list(inventory.topics), dependencies=list(generated_deps), parse_ok=True)
    cmp = compare_graphs(example, full)
    missing_norms = {
        (normalize_topic(e[0]), normalize_topic(e[1])) for e in cmp["missing_dependencies"]
    }
    reversed_norms: set[tuple[str, str]] = set()
    for item in cmp.get("reversed_dependencies") or []:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            gold = item[1]
            if isinstance(gold, (list, tuple)) and len(gold) == 2:
                reversed_norms.add((normalize_topic(gold[0]), normalize_topic(gold[1])))

    def _required_ok(frm: str, to: str) -> bool:
        key = (normalize_topic(frm), normalize_topic(to))
        return key not in missing_norms and key not in reversed_norms

    correctly_generated = sum(1 for frm, to in required if _required_ok(frm, to))
    opportunity_correct = sum(1 for frm, to in opportunity_edges if _required_ok(frm, to))

    n_req = len(required)
    n_opp = len(opportunity_edges)
    return {
        "required_edge_count": n_req,
        "opportunity_edge_count": n_opp,
        "correctly_generated_required": correctly_generated,
        "opportunity_correct": opportunity_correct,
        "EDGE_OPPORTUNITY_RATE": _safe_rate(n_opp, n_req),
        "CONDITIONAL_EDGE_RECALL": _safe_rate(opportunity_correct, n_opp),
        "opportunity_edges": [list(e) for e in opportunity_edges],
    }



def inventory_delta(
    example: EvalExample,
    baseline_inv: GeneratedGraph,
    cf_inv: GeneratedGraph,
) -> dict[str, Any]:
    """Compare which required gold concepts are present (exact/alias) in each inventory."""
    baseline_only: list[str] = []
    cf_only: list[str] = []
    both: list[str] = []
    neither: list[str] = []
    for g in example.required_topic_list():
        b = classify_gold_topic_representation(g, example, baseline_inv)["status"] in _PRESENT
        c = classify_gold_topic_representation(g, example, cf_inv)["status"] in _PRESENT
        if b and c:
            both.append(g)
        elif b and not c:
            baseline_only.append(g)
        elif c and not b:
            cf_only.append(g)
        else:
            neither.append(g)
    return {
        "PRESENT_IN_BASELINE_ONLY": baseline_only,
        "PRESENT_IN_CONCEPT_FIRST_ONLY": cf_only,
        "PRESENT_IN_BOTH": both,
        "MISSING_FROM_BOTH": neither,
    }


def _bucket_missing(attrs: list[dict[str, Any]]) -> dict[str, Any]:
    c = Counter(a["primary_attribution"] for a in attrs)
    n = sum(c.values())
    endpoint_absent = (
        c.get("SOURCE_ABSENT_FROM_INVENTORY", 0)
        + c.get("TARGET_ABSENT_FROM_INVENTORY", 0)
        + c.get("BOTH_ENDPOINTS_ABSENT_FROM_INVENTORY", 0)
    )
    representation = (
        c.get("ENDPOINT_GRANULARITY_MISMATCH", 0)
        + c.get("ENDPOINT_ABSTRACTION_MISMATCH", 0)
        + c.get("ENDPOINT_DECOMPOSITION", 0)
    )
    omitted = c.get("BOTH_ENDPOINTS_AVAILABLE_EDGE_OMITTED", 0)
    return {
        "counts": dict(c),
        "total": n,
        "endpoint_absent_count": endpoint_absent,
        "endpoint_absent_rate": _safe_rate(endpoint_absent, n),
        "representation_mismatch_count": representation,
        "representation_mismatch_rate": _safe_rate(representation, n),
        "both_available_omitted_count": omitted,
        "both_available_omitted_rate": _safe_rate(omitted, n),
    }


def _bucket_invalid(attrs: list[dict[str, Any]]) -> dict[str, Any]:
    c = Counter(a["primary_attribution"] for a in attrs)
    n = sum(c.values())
    invalid_ep = (
        c.get("SOURCE_OUT_OF_SCOPE_INVENTORY", 0)
        + c.get("TARGET_OUT_OF_SCOPE_INVENTORY", 0)
        + c.get("BOTH_ENDPOINTS_OUT_OF_SCOPE_INVENTORY", 0)
    )
    drift = (
        c.get("ENDPOINT_GRANULARITY_DRIFT", 0)
        + c.get("ENDPOINT_ABSTRACTION_DRIFT", 0)
        + c.get("ENDPOINT_DECOMPOSITION_DRIFT", 0)
        + c.get("CURRICULUM_SCOPE_DRIFT", 0)
    )
    valid_invalid = c.get("BOTH_ENDPOINTS_VALID_EDGE_INVALID", 0)
    return {
        "counts": dict(c),
        "total": n,
        "invalid_endpoint_count": invalid_ep,
        "invalid_endpoint_rate": _safe_rate(invalid_ep, n),
        "representation_drift_count": drift,
        "representation_drift_rate": _safe_rate(drift, n),
        "both_valid_edge_invalid_count": valid_invalid,
        "both_valid_edge_invalid_rate": _safe_rate(valid_invalid, n),
    }


def analyze_case(
    example: EvalExample,
    baseline_row: dict[str, Any],
    cf_row: dict[str, Any],
) -> dict[str, Any]:
    adapted = adapt_example_for_edge_mode(
        example, "edge_calibrated", topic_matching_mode="curated_alias"
    )
    baseline_topics = [str(t) for t in (baseline_row.get("generated_topics") or [])]
    cf_stage1 = extract_stage1_inventory(cf_row)
    cf_norm = extract_normalized_inventory(cf_row)
    # Causal opportunity uses the inventory Stage 2 actually saw (= normalized)
    inv_for_stage2 = cf_norm or cf_stage1

    baseline_inv = inventory_graph_from_titles(baseline_topics)
    cf_inv = inventory_graph_from_titles(inv_for_stage2)
    cf_stage1_graph = inventory_graph_from_titles(cf_stage1)

    baseline_inv_metrics = evaluate_inventory(baseline_topics, adapted)
    cf_inv_metrics = evaluate_inventory(cf_stage1, adapted)

    baseline_graph = _graph_from_row(baseline_row)
    cf_graph = _graph_from_row(cf_row)

    baseline_cmp = compare_graphs(adapted, baseline_graph)
    cf_cmp = compare_graphs(adapted, cf_graph)

    baseline_missing_attrs = [
        attribute_missing_edge_vs_inventory(str(e[0]), str(e[1]), adapted, baseline_inv)
        for e in baseline_cmp["missing_dependencies"]
    ]
    cf_missing_attrs = [
        attribute_missing_edge_vs_inventory(str(e[0]), str(e[1]), adapted, cf_inv)
        for e in cf_cmp["missing_dependencies"]
    ]
    baseline_invalid_attrs = [
        attribute_invalid_edge_vs_inventory(str(e[0]), str(e[1]), adapted)
        for e in baseline_cmp["extra_dependencies"]
    ]
    cf_invalid_attrs = [
        attribute_invalid_edge_vs_inventory(str(e[0]), str(e[1]), adapted)
        for e in cf_cmp["extra_dependencies"]
    ]

    baseline_opp = edge_opportunity_and_conditional_recall(
        adapted, baseline_inv, list(baseline_graph.dependencies)
    )
    cf_opp = edge_opportunity_and_conditional_recall(
        adapted, cf_inv, list(cf_graph.dependencies)
    )

    delta = inventory_delta(adapted, baseline_inv, cf_inv)

    # Downstream impact of delta categories
    def _edges_touching(concepts: list[str], missing_edges: list) -> int:
        norms = {normalize_topic(c) for c in concepts}
        n = 0
        for e in missing_edges:
            if normalize_topic(str(e[0])) in norms or normalize_topic(str(e[1])) in norms:
                n += 1
        return n

    delta_impact = {
        "PRESENT_IN_BASELINE_ONLY_missing_edge_touch_count": _edges_touching(
            delta["PRESENT_IN_BASELINE_ONLY"], cf_cmp["missing_dependencies"]
        ),
        "PRESENT_IN_CONCEPT_FIRST_ONLY_missing_edge_touch_count": _edges_touching(
            delta["PRESENT_IN_CONCEPT_FIRST_ONLY"], baseline_cmp["missing_dependencies"]
        ),
        "MISSING_FROM_BOTH_missing_edge_touch_count_cf": _edges_touching(
            delta["MISSING_FROM_BOTH"], cf_cmp["missing_dependencies"]
        ),
    }

    b_miss = _bucket_missing(baseline_missing_attrs)
    c_miss = _bucket_missing(cf_missing_attrs)
    b_inv = _bucket_invalid(baseline_invalid_attrs)
    c_inv = _bucket_invalid(cf_invalid_attrs)

    # Case conclusion
    inv_f1_delta = cf_inv_metrics["topic_f1"] - baseline_inv_metrics["topic_f1"]
    omitted_delta = c_miss["both_available_omitted_count"] - b_miss["both_available_omitted_count"]
    absent_delta = c_miss["endpoint_absent_count"] - b_miss["endpoint_absent_count"]
    cond_delta = cf_opp["CONDITIONAL_EDGE_RECALL"] - baseline_opp["CONDITIONAL_EDGE_RECALL"]

    if abs(inv_f1_delta) < 0.03 and abs(cond_delta) < 0.05 and absent_delta == 0 and omitted_delta == 0:
        case_conclusion = "NO_MEANINGFUL_DIFFERENCE"
    elif absent_delta > 0 and cond_delta >= -0.05:
        case_conclusion = "INVENTORY_FAILURE"
    elif omitted_delta > 0 and inv_f1_delta >= -0.03:
        case_conclusion = "RELATIONSHIP_FAILURE"
    elif absent_delta > 0 and (omitted_delta > 0 or cond_delta < -0.05):
        case_conclusion = "MIXED"
    elif inv_f1_delta < -0.05:
        case_conclusion = "INVENTORY_FAILURE"
    elif cond_delta < -0.08:
        case_conclusion = "RELATIONSHIP_FAILURE"
    else:
        case_conclusion = "MIXED"

    return {
        "case_id": example.id,
        "learning_objective": example.goal,
        "gold_topics": list(adapted.required_topic_list()),
        "gold_dependencies": [list(d) for d in adapted.required_dependency_list()],
        "baseline_inventory": baseline_topics,
        "concept_first_stage1_inventory": cf_stage1,
        "concept_first_normalized_inventory": cf_norm,
        "baseline_dependencies": [list(d) for d in baseline_graph.dependencies],
        "concept_first_dependencies": [list(d) for d in cf_graph.dependencies],
        "inventory_metrics": {
            "baseline": {
                k: baseline_inv_metrics[k]
                for k in (
                    "topic_precision",
                    "topic_recall",
                    "topic_f1",
                    "hallucinated_topic_rate",
                    "out_of_scope_rate",
                    "granularity_mismatch_rate",
                    "abstraction_mismatch_rate",
                    "decomposition_rate",
                    "related_but_distinct_rate",
                    "missing_foundational_concept_rate",
                    "n_topics",
                    "class_counts",
                )
            },
            "concept_first_stage1": {
                k: cf_inv_metrics[k]
                for k in (
                    "topic_precision",
                    "topic_recall",
                    "topic_f1",
                    "hallucinated_topic_rate",
                    "out_of_scope_rate",
                    "granularity_mismatch_rate",
                    "abstraction_mismatch_rate",
                    "decomposition_rate",
                    "related_but_distinct_rate",
                    "missing_foundational_concept_rate",
                    "n_topics",
                    "class_counts",
                )
            },
        },
        "missing_required_edge_attributions": {
            "baseline": baseline_missing_attrs,
            "concept_first": cf_missing_attrs,
            "baseline_buckets": b_miss,
            "concept_first_buckets": c_miss,
        },
        "invalid_extra_edge_attributions": {
            "baseline": baseline_invalid_attrs,
            "concept_first": cf_invalid_attrs,
            "baseline_buckets": b_inv,
            "concept_first_buckets": c_inv,
        },
        "edge_opportunity": {"baseline": baseline_opp, "concept_first": cf_opp},
        "conditional_edge_recall": {
            "baseline": baseline_opp["CONDITIONAL_EDGE_RECALL"],
            "concept_first": cf_opp["CONDITIONAL_EDGE_RECALL"],
        },
        "baseline_vs_cf_inventory_delta": delta,
        "delta_impact": delta_impact,
        "case_conclusion": case_conclusion,
        # keep full inventory detail for machine readers (not duplicated in MD)
        "inventory_detail": {
            "baseline_gold_representation": baseline_inv_metrics["gold_representation"],
            "cf_gold_representation": cf_inv_metrics["gold_representation"],
            "cf_concept_classifications": cf_inv_metrics["concept_classifications"],
        },
    }


def _aggregate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    def avg_inv(side: str, key: str) -> float:
        return _mean([float(c["inventory_metrics"][side][key]) for c in cases])

    def sum_bucket(side: str, which: str, field: str) -> int:
        # which: missing_required_edge_attributions | invalid_extra_edge_attributions
        key = "baseline_buckets" if side == "baseline" else "concept_first_buckets"
        return sum(int(c[which][key].get(field, 0)) for c in cases)

    def sum_total(side: str, which: str) -> int:
        key = "baseline_buckets" if side == "baseline" else "concept_first_buckets"
        return sum(int(c[which][key].get("total", 0)) for c in cases)

    b_miss_n = sum_total("baseline", "missing_required_edge_attributions")
    c_miss_n = sum_total("concept_first", "missing_required_edge_attributions")
    b_inv_n = sum_total("baseline", "invalid_extra_edge_attributions")
    c_inv_n = sum_total("concept_first", "invalid_extra_edge_attributions")

    b_opp_num = sum(int(c["edge_opportunity"]["baseline"]["opportunity_edge_count"]) for c in cases)
    b_req = sum(int(c["edge_opportunity"]["baseline"]["required_edge_count"]) for c in cases)
    c_opp_num = sum(int(c["edge_opportunity"]["concept_first"]["opportunity_edge_count"]) for c in cases)
    c_req = sum(int(c["edge_opportunity"]["concept_first"]["required_edge_count"]) for c in cases)
    b_opp_ok = sum(int(c["edge_opportunity"]["baseline"]["opportunity_correct"]) for c in cases)
    c_opp_ok = sum(int(c["edge_opportunity"]["concept_first"]["opportunity_correct"]) for c in cases)

    delta_counts = Counter()
    for c in cases:
        d = c["baseline_vs_cf_inventory_delta"]
        for k in (
            "PRESENT_IN_BASELINE_ONLY",
            "PRESENT_IN_CONCEPT_FIRST_ONLY",
            "PRESENT_IN_BOTH",
            "MISSING_FROM_BOTH",
        ):
            delta_counts[k] += len(d[k])

    conclusions = Counter(c["case_conclusion"] for c in cases)

    inv_keys = (
        "topic_precision",
        "topic_recall",
        "topic_f1",
        "missing_foundational_concept_rate",
        "out_of_scope_rate",
        "granularity_mismatch_rate",
        "abstraction_mismatch_rate",
        "decomposition_rate",
        "hallucinated_topic_rate",
    )
    inventory_comparison = {}
    for k in inv_keys:
        b = avg_inv("baseline", k)
        cf = avg_inv("concept_first_stage1", k)
        inventory_comparison[k] = {"baseline": b, "concept_first_stage1": cf, "delta": cf - b}

    return {
        "n_cases": len(cases),
        "inventory_comparison": inventory_comparison,
        "causal_missing": {
            "baseline": {
                "endpoint_absent": sum_bucket("baseline", "missing_required_edge_attributions", "endpoint_absent_count"),
                "representation_mismatch": sum_bucket(
                    "baseline", "missing_required_edge_attributions", "representation_mismatch_count"
                ),
                "both_available_omitted": sum_bucket(
                    "baseline", "missing_required_edge_attributions", "both_available_omitted_count"
                ),
                "total": b_miss_n,
            },
            "concept_first": {
                "endpoint_absent": sum_bucket(
                    "concept_first", "missing_required_edge_attributions", "endpoint_absent_count"
                ),
                "representation_mismatch": sum_bucket(
                    "concept_first", "missing_required_edge_attributions", "representation_mismatch_count"
                ),
                "both_available_omitted": sum_bucket(
                    "concept_first", "missing_required_edge_attributions", "both_available_omitted_count"
                ),
                "total": c_miss_n,
            },
        },
        "causal_invalid": {
            "baseline": {
                "invalid_endpoint": sum_bucket(
                    "baseline", "invalid_extra_edge_attributions", "invalid_endpoint_count"
                ),
                "representation_drift": sum_bucket(
                    "baseline", "invalid_extra_edge_attributions", "representation_drift_count"
                ),
                "both_valid_edge_invalid": sum_bucket(
                    "baseline", "invalid_extra_edge_attributions", "both_valid_edge_invalid_count"
                ),
                "total": b_inv_n,
            },
            "concept_first": {
                "invalid_endpoint": sum_bucket(
                    "concept_first", "invalid_extra_edge_attributions", "invalid_endpoint_count"
                ),
                "representation_drift": sum_bucket(
                    "concept_first", "invalid_extra_edge_attributions", "representation_drift_count"
                ),
                "both_valid_edge_invalid": sum_bucket(
                    "concept_first", "invalid_extra_edge_attributions", "both_valid_edge_invalid_count"
                ),
                "total": c_inv_n,
            },
        },
        "opportunity": {
            "baseline": {
                "EDGE_OPPORTUNITY_RATE": _safe_rate(b_opp_num, b_req),
                "CONDITIONAL_EDGE_RECALL": _safe_rate(b_opp_ok, b_opp_num),
                "opportunity_edges": b_opp_num,
                "required_edges": b_req,
                "opportunity_correct": b_opp_ok,
            },
            "concept_first": {
                "EDGE_OPPORTUNITY_RATE": _safe_rate(c_opp_num, c_req),
                "CONDITIONAL_EDGE_RECALL": _safe_rate(c_opp_ok, c_opp_num),
                "opportunity_edges": c_opp_num,
                "required_edges": c_req,
                "opportunity_correct": c_opp_ok,
            },
        },
        "inventory_delta_totals": dict(delta_counts),
        "case_conclusions": dict(conclusions),
    }


def choose_diagnosis(aggregate: dict[str, Any]) -> tuple[str, str]:
    """Return (code, explanation) using measured aggregates only."""
    inv = aggregate["inventory_comparison"]
    miss = aggregate["causal_missing"]
    invld = aggregate["causal_invalid"]
    opp = aggregate["opportunity"]

    f1_delta = inv["topic_f1"]["delta"]
    recall_delta = inv["topic_recall"]["delta"]
    miss_found_delta = inv["missing_foundational_concept_rate"]["delta"]
    oos_delta = inv["out_of_scope_rate"]["delta"]

    b_abs = miss["baseline"]["endpoint_absent"]
    c_abs = miss["concept_first"]["endpoint_absent"]
    b_omit = miss["baseline"]["both_available_omitted"]
    c_omit = miss["concept_first"]["both_available_omitted"]
    b_inv_ep = invld["baseline"]["invalid_endpoint"]
    c_inv_ep = invld["concept_first"]["invalid_endpoint"]
    b_valid_bad = invld["baseline"]["both_valid_edge_invalid"]
    c_valid_bad = invld["concept_first"]["both_valid_edge_invalid"]

    opp_delta = (
        opp["concept_first"]["EDGE_OPPORTUNITY_RATE"] - opp["baseline"]["EDGE_OPPORTUNITY_RATE"]
    )
    cond_delta = (
        opp["concept_first"]["CONDITIONAL_EDGE_RECALL"] - opp["baseline"]["CONDITIONAL_EDGE_RECALL"]
    )

    inventory_signals = 0
    relationship_signals = 0

    if f1_delta < -0.04 or recall_delta < -0.04 or miss_found_delta > 0.04:
        inventory_signals += 2
    if oos_delta > 0.05:
        inventory_signals += 1
    if c_abs - b_abs >= 5:
        inventory_signals += 2
    if opp_delta < -0.05:
        inventory_signals += 2
    if c_inv_ep - b_inv_ep >= 5:
        inventory_signals += 1

    if cond_delta < -0.05:
        relationship_signals += 2
    if c_omit - b_omit >= 5:
        relationship_signals += 2
    if c_valid_bad - b_valid_bad >= 3:
        relationship_signals += 1
    # Similar inventory F1 but worse conditional recall
    if abs(f1_delta) < 0.03 and cond_delta < -0.05:
        relationship_signals += 2

    if inventory_signals >= 3 and relationship_signals >= 3:
        return (
            "MIXED_FAILURE",
            f"Inventory signals={inventory_signals}, relationship signals={relationship_signals}.",
        )
    if inventory_signals >= 3 and relationship_signals < 3:
        return (
            "STAGE_1_INVENTORY_FAILURE_PRIMARY",
            f"Inventory signals={inventory_signals} dominate relationship signals={relationship_signals} "
            f"(inv F1 delta={f1_delta:+.3f}, opportunity delta={opp_delta:+.3f}, "
            f"absent edges {b_abs}→{c_abs}, conditional recall delta={cond_delta:+.3f}).",
        )
    if relationship_signals >= 3 and inventory_signals < 3:
        return (
            "STAGE_2_RELATIONSHIP_FAILURE_PRIMARY",
            f"Relationship signals={relationship_signals} dominate inventory signals={inventory_signals} "
            f"(conditional recall delta={cond_delta:+.3f}, omitted {b_omit}→{c_omit}).",
        )
    if inventory_signals == 0 and relationship_signals == 0:
        return ("NO_CLEAR_DIAGNOSIS", "No strong attribution signals above thresholds.")
    if inventory_signals > relationship_signals:
        return (
            "STAGE_1_INVENTORY_FAILURE_PRIMARY",
            f"Weak-majority inventory signals={inventory_signals} vs relationship={relationship_signals}.",
        )
    if relationship_signals > inventory_signals:
        return (
            "STAGE_2_RELATIONSHIP_FAILURE_PRIMARY",
            f"Weak-majority relationship signals={relationship_signals} vs inventory={inventory_signals}.",
        )
    return ("MIXED_FAILURE", f"Tied signals inventory={inventory_signals} relationship={relationship_signals}.")


def _pick_representative(cases: list[dict[str, Any]], n: int = 12) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {
        "inv_worse": [],
        "inv_better_rel_worse": [],
        "similar_edge_diff": [],
        "missing_foundational": [],
        "granularity": [],
        "oos_invalid": [],
        "both_available_omitted": [],
        "both_valid_invalid": [],
        "cf_improved": [],
        "baseline_better": [],
    }
    for c in cases:
        bi = c["inventory_metrics"]["baseline"]
        ci = c["inventory_metrics"]["concept_first_stage1"]
        b_omit = c["missing_required_edge_attributions"]["baseline_buckets"]["both_available_omitted_count"]
        c_omit = c["missing_required_edge_attributions"]["concept_first_buckets"]["both_available_omitted_count"]
        c_abs = c["missing_required_edge_attributions"]["concept_first_buckets"]["endpoint_absent_count"]
        c_gran = c["missing_required_edge_attributions"]["concept_first_buckets"]["counts"].get(
            "ENDPOINT_GRANULARITY_MISMATCH", 0
        )
        c_oos = c["invalid_extra_edge_attributions"]["concept_first_buckets"]["invalid_endpoint_count"]
        c_valid_bad = c["invalid_extra_edge_attributions"]["concept_first_buckets"][
            "both_valid_edge_invalid_count"
        ]
        if ci["topic_f1"] + 0.03 < bi["topic_f1"]:
            buckets["inv_worse"].append(c)
        if ci["topic_f1"] + 0.02 >= bi["topic_f1"] and c_omit > b_omit:
            buckets["inv_better_rel_worse"].append(c)
        if abs(ci["topic_f1"] - bi["topic_f1"]) < 0.05 and (c_omit != b_omit):
            buckets["similar_edge_diff"].append(c)
        if ci["missing_foundational_concept_rate"] > 0.2 or c_abs > 0:
            buckets["missing_foundational"].append(c)
        if c_gran > 0:
            buckets["granularity"].append(c)
        if c_oos > 0:
            buckets["oos_invalid"].append(c)
        if c_omit > 0:
            buckets["both_available_omitted"].append(c)
        if c_valid_bad > 0:
            buckets["both_valid_invalid"].append(c)
        if c["case_conclusion"] == "NO_MEANINGFUL_DIFFERENCE" or (
            ci["topic_f1"] > bi["topic_f1"] + 0.05
        ):
            buckets["cf_improved"].append(c)
        if bi["topic_f1"] > ci["topic_f1"] + 0.05:
            buckets["baseline_better"].append(c)

    picked: list[dict[str, Any]] = []
    seen: set[str] = set()
    order = [
        "inv_worse",
        "inv_better_rel_worse",
        "similar_edge_diff",
        "missing_foundational",
        "granularity",
        "oos_invalid",
        "both_available_omitted",
        "both_valid_invalid",
        "cf_improved",
        "baseline_better",
    ]
    for key in order:
        for c in buckets[key]:
            if c["case_id"] in seen:
                continue
            picked.append(c)
            seen.add(c["case_id"])
            break
    for c in cases:
        if len(picked) >= n:
            break
        if c["case_id"] not in seen:
            picked.append(c)
            seen.add(c["case_id"])
    return picked[:n]


def _render_md(payload: dict[str, Any], cases: list[dict[str, Any]]) -> str:
    agg = payload["aggregate"]
    inv = agg["inventory_comparison"]
    miss = agg["causal_missing"]
    invld = agg["causal_invalid"]
    opp = agg["opportunity"]
    diag = payload["diagnosis"]
    lines = [
        "# Inventory vs Relationship Attribution",
        "",
        f"- Source artifact: `{payload['source_artifact']}`",
        f"- LLM calls: **{payload['llm_calls']}**",
        f"- Matching: `curated_alias` + `edge_calibrated`",
        f"- Cases: {agg['n_cases']}",
        f"- Diagnosis: **{diag['code']}**",
        f"- Rationale: {diag['rationale']}",
        "",
        "## Inventory comparison",
        "",
        "| Metric | Baseline | Concept-First Stage 1 | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for k, row in inv.items():
        lines.append(
            f"| {k} | {row['baseline']:.3f} | {row['concept_first_stage1']:.3f} | {row['delta']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "## Causal missing-edge attribution",
            "",
            "| Attribution | Baseline | Concept-First |",
            "| --- | ---: | ---: |",
            f"| Endpoint absent | {miss['baseline']['endpoint_absent']} / {miss['baseline']['total']} | "
            f"{miss['concept_first']['endpoint_absent']} / {miss['concept_first']['total']} |",
            f"| Representation mismatch | {miss['baseline']['representation_mismatch']} / {miss['baseline']['total']} | "
            f"{miss['concept_first']['representation_mismatch']} / {miss['concept_first']['total']} |",
            f"| Both endpoints available, edge omitted | {miss['baseline']['both_available_omitted']} / {miss['baseline']['total']} | "
            f"{miss['concept_first']['both_available_omitted']} / {miss['concept_first']['total']} |",
            "",
            "## Causal invalid-extra attribution",
            "",
            "| Attribution | Baseline | Concept-First |",
            "| --- | ---: | ---: |",
            f"| Invalid / OOS endpoint | {invld['baseline']['invalid_endpoint']} / {invld['baseline']['total']} | "
            f"{invld['concept_first']['invalid_endpoint']} / {invld['concept_first']['total']} |",
            f"| Representation drift | {invld['baseline']['representation_drift']} / {invld['baseline']['total']} | "
            f"{invld['concept_first']['representation_drift']} / {invld['concept_first']['total']} |",
            f"| Both endpoints valid, edge invalid | {invld['baseline']['both_valid_edge_invalid']} / {invld['baseline']['total']} | "
            f"{invld['concept_first']['both_valid_edge_invalid']} / {invld['concept_first']['total']} |",
            "",
            "## Opportunity analysis",
            "",
            f"- Baseline EDGE_OPPORTUNITY_RATE: **{opp['baseline']['EDGE_OPPORTUNITY_RATE']:.3f}** "
            f"({opp['baseline']['opportunity_edges']}/{opp['baseline']['required_edges']})",
            f"- Concept-First EDGE_OPPORTUNITY_RATE: **{opp['concept_first']['EDGE_OPPORTUNITY_RATE']:.3f}** "
            f"({opp['concept_first']['opportunity_edges']}/{opp['concept_first']['required_edges']})",
            f"- Baseline CONDITIONAL_EDGE_RECALL: **{opp['baseline']['CONDITIONAL_EDGE_RECALL']:.3f}** "
            f"({opp['baseline']['opportunity_correct']}/{opp['baseline']['opportunity_edges']})",
            f"- Concept-First CONDITIONAL_EDGE_RECALL: **{opp['concept_first']['CONDITIONAL_EDGE_RECALL']:.3f}** "
            f"({opp['concept_first']['opportunity_correct']}/{opp['concept_first']['opportunity_edges']})",
            "",
            "## Inventory delta (required gold concepts)",
            "",
            f"- PRESENT_IN_BASELINE_ONLY: {agg['inventory_delta_totals'].get('PRESENT_IN_BASELINE_ONLY', 0)}",
            f"- PRESENT_IN_CONCEPT_FIRST_ONLY: {agg['inventory_delta_totals'].get('PRESENT_IN_CONCEPT_FIRST_ONLY', 0)}",
            f"- PRESENT_IN_BOTH: {agg['inventory_delta_totals'].get('PRESENT_IN_BOTH', 0)}",
            f"- MISSING_FROM_BOTH: {agg['inventory_delta_totals'].get('MISSING_FROM_BOTH', 0)}",
            "",
            "## Representative cases",
            "",
        ]
    )
    for c in cases:
        lines.extend(
            [
                f"### {c['case_id']} — {c['case_conclusion']}",
                "",
                f"**Learning objective:** {c['learning_objective']}",
                "",
                f"- Gold topics: {c['gold_topics']}",
                f"- Baseline topics: {c['baseline_inventory']}",
                f"- Concept-First Stage-1: {c['concept_first_stage1_inventory']}",
                f"- Normalized inventory: {c['concept_first_normalized_inventory']}",
                f"- Gold deps: {c['gold_dependencies']}",
                f"- Baseline deps: {c['baseline_dependencies']}",
                f"- Concept-First deps: {c['concept_first_dependencies']}",
                f"- Inventory F1: baseline={c['inventory_metrics']['baseline']['topic_f1']:.3f} "
                f"CF={c['inventory_metrics']['concept_first_stage1']['topic_f1']:.3f}",
                f"- Missing attr (CF): {c['missing_required_edge_attributions']['concept_first_buckets']}",
                f"- Invalid attr (CF): {c['invalid_extra_edge_attributions']['concept_first_buckets']}",
                f"- Opportunity/conditional: "
                f"opp={c['edge_opportunity']['concept_first']['EDGE_OPPORTUNITY_RATE']:.3f} "
                f"cond={c['conditional_edge_recall']['concept_first']:.3f}",
                f"- Inventory delta: {c['baseline_vs_cf_inventory_delta']}",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def run_inventory_attribution(
    result_path: str | Path,
    *,
    dataset_path: str | Path | None = None,
    baseline_system: str = "synapse",
    concept_system: str = "concept_first",
    output_dir: str | Path | None = None,
    max_cases: int = 12,
) -> tuple[Path, Path]:
    """Analyze stored baseline + Concept-First artifact; write JSON + Markdown."""
    target = Path(result_path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    systems = payload.get("systems") or {}
    if baseline_system not in systems or concept_system not in systems:
        raise ValueError(
            f"Artifact must contain systems {baseline_system!r} and {concept_system!r}; "
            f"found {list(systems)}"
        )

    ds_stem = payload.get("dataset") or payload.get("dataset_version") or "learning_graph_quality_v1"
    ds_path = Path(dataset_path) if dataset_path else _REPO_ROOT / "data" / "eval" / f"{ds_stem}.jsonl"
    if not ds_path.is_file():
        ds_path = _REPO_ROOT / "data" / "eval" / "learning_graph_quality_v1.jsonl"
    examples = {ex.id: ex for ex in load_dataset(ds_path)}

    base_rows = _rows_by_id(systems[baseline_system])
    cf_rows = _rows_by_id(systems[concept_system])

    stage1_ok = 0
    stage1_missing = 0
    for eid, row in cf_rows.items():
        if stage1_data_available(row):
            stage1_ok += 1
        else:
            stage1_missing += 1

    if stage1_missing and stage1_ok == 0:
        raise RuntimeError(
            "NEW_GENERATION_REQUIRED: Concept-First rows lack Stage-1 inventory fields "
            "(candidate_concepts / normalized_inventory). Re-run quality with concept_first "
            "after ensuring generation_meta persistence."
        )

    llm_calls = "NO_NEW_LLM_CALLS" if stage1_ok else "NEW_GENERATION_REQUIRED"

    cases: list[dict[str, Any]] = []
    for eid, ex in examples.items():
        if eid not in base_rows or eid not in cf_rows:
            continue
        if not base_rows[eid].get("parse_ok", True) or not cf_rows[eid].get("parse_ok", True):
            continue
        cases.append(analyze_case(ex, base_rows[eid], cf_rows[eid]))

    aggregate = _aggregate(cases)
    code, rationale = choose_diagnosis(aggregate)
    representative = _pick_representative(cases, n=max_cases)

    out_dir = Path(output_dir) if output_dir else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    json_path = out_dir / f"{ts}_inventory_attribution.json"
    md_path = out_dir / f"{ts}_inventory_attribution_comparison.md"

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_artifact": str(target),
        "dataset": str(ds_path),
        "llm_calls": llm_calls,
        "stage1_rows_with_inventory": stage1_ok,
        "stage1_rows_missing_inventory": stage1_missing,
        "matching_mode": "curated_alias",
        "edge_mode": "edge_calibrated",
        "baseline_system": baseline_system,
        "concept_system": concept_system,
        "benchmark_config": {
            "model": payload.get("model"),
            "seed": payload.get("seed"),
            "repetitions": payload.get("repetitions"),
            "example_count": payload.get("example_count"),
        },
        "definitions": {
            "EDGE_OPPORTUNITY_RATE": (
                "required edges whose both gold endpoints have EXACT/ALIAS representation "
                "in the available inventory / total required edges"
            ),
            "CONDITIONAL_EDGE_RECALL": (
                "correctly generated required edges among opportunity edges "
                "(both endpoints available in inventory)"
            ),
        },
        "diagnosis": {"code": code, "rationale": rationale},
        "aggregate": aggregate,
        "cases": cases,
        "representative_case_ids": [c["case_id"] for c in representative],
    }
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(_render_md(result, representative), encoding="utf-8")
    return md_path, json_path
