"""Offline analysis: domain_curriculum_prior vs domain_prior_edge_classifier."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.curriculum.edge_candidates import generate_candidate_pairs
from app.curriculum.inventory import load_case_domain_map, load_domain_inventory
from app.curriculum.selection import SelectedConcept
from app.evaluation.dataset import load_dataset
from app.evaluation.edge_ambiguity import adapt_example_for_edge_mode
from app.evaluation.inspect import _graph_from_row
from app.evaluation.metrics import (
    aggregate_scores,
    compare_graphs,
    find_redundant_transitive_edges,
    normalize_topic,
    score_graph,
)
from app.evaluation.schemas import GeneratedGraph

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = _REPO_ROOT / "results" / "failure_analysis"


def _safe_rate(n: float, d: float) -> float:
    return (n / d) if d else 0.0


def _row_by_id(block: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in block.get("example_results") or []:
        eid = str(row.get("example_id") or "")
        if eid and eid not in out:
            out[eid] = row
    return out


def _norm_edge(frm: str, to: str) -> tuple[str, str]:
    return (normalize_topic(frm), normalize_topic(to))


def _edges_from_graph(graph: GeneratedGraph) -> set[tuple[str, str]]:
    return {_norm_edge(a, b) for a, b in graph.dependencies}


def _classify_fp(
    edge: tuple[str, str],
    *,
    gold_req: set[tuple[str, str]],
    acceptable: set[tuple[str, str]],
    generated_all: set[tuple[str, str]],
) -> str:
    rev = (edge[1], edge[0])
    if rev in gold_req:
        return "WRONG_DIRECTION"
    if edge in acceptable:
        return "ACCEPTABLE_ALTERNATIVE"
    # Transitive redundancy among generated edges
    redundant = {
        _norm_edge(a, b) for a, b in find_redundant_transitive_edges(list(generated_all))
    }
    if edge in redundant:
        return "TRANSITIVE_REDUNDANCY"
    return "INVALID_DIRECT_EDGE"


def _classify_fn(
    edge: tuple[str, str],
    *,
    selected: set[str],
    predicted_required: set[tuple[str, str]],
    uncertain: set[tuple[str, str]],
    predicted_not_required: set[tuple[str, str]],
    candidate_space: set[tuple[str, str]],
) -> str:
    frm, to = edge
    if frm not in selected or to not in selected:
        return "NOT_SELECTED"
    if edge not in candidate_space:
        return "MISSING_FROM_CANDIDATE_SPACE"
    if edge in uncertain:
        return "UNCERTAIN"
    if edge in predicted_not_required or edge not in predicted_required:
        return "PREDICTED_NOT_REQUIRED"
    return "PREDICTED_NOT_REQUIRED"


def _meta(row: dict[str, Any]) -> dict[str, Any]:
    return dict(row.get("generation_meta") or row.get("meta") or {})


def run_constrained_dependency_analysis(
    artifact_path: str | Path,
    *,
    dataset_path: str | Path | None = None,
    prior_system: str = "domain_curriculum_prior",
    classifier_system: str = "domain_prior_edge_classifier",
    baseline_system: str = "synapse",
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    target = Path(artifact_path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    systems = payload.get("systems") or {}
    if prior_system not in systems or classifier_system not in systems:
        raise ValueError(
            f"Artifact must contain {prior_system!r} and {classifier_system!r}; "
            f"found {list(systems)}"
        )

    ds_path = (
        Path(dataset_path)
        if dataset_path
        else _REPO_ROOT / "data" / "eval" / "learning_graph_quality_v1.jsonl"
    )
    examples = {ex.id: ex for ex in load_dataset(ds_path)}
    case_map = load_case_domain_map()
    prior_rows = _row_by_id(systems[prior_system])
    clf_rows = _row_by_id(systems[classifier_system])
    base_rows = _row_by_id(systems[baseline_system]) if baseline_system in systems else {}
    shared = sorted(set(prior_rows) & set(clf_rows) & set(case_map))

    prior_scores = []
    clf_scores = []
    base_scores = []
    case_reports: list[dict[str, Any]] = []
    fn_counts: Counter[str] = Counter()
    fp_counts: Counter[str] = Counter()
    pair_stats = {
        "candidate_pair_count": 0,
        "required_pair_count": 0,
        "predicted_required_pair_count": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "uncertain": 0,
        "invalid_pair_outputs": 0,
        "unknown_ids": 0,
        "duplicates": 0,
    }
    source_centered: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"gold": 0, "prior_hit": 0, "clf_hit": 0, "prior_fp": 0, "clf_fp": 0}
    )
    domain_aggs: dict[str, dict[str, list]] = defaultdict(
        lambda: {"prior": [], "clf": [], "base": []}
    )

    for eid in shared:
        ex = examples.get(eid)
        if ex is None:
            continue
        domain = case_map[eid]
        inventory = load_domain_inventory(domain)
        adapted = adapt_example_for_edge_mode(
            ex, "edge_calibrated", topic_matching_mode="curated_alias"
        )
        prior_row = prior_rows[eid]
        clf_row = clf_rows[eid]
        prior_g = _graph_from_row(prior_row)
        clf_g = _graph_from_row(clf_row)
        prior_sc = score_graph(adapted, prior_g) if prior_g.parse_ok else None
        clf_sc = score_graph(adapted, clf_g) if clf_g.parse_ok else None
        if prior_sc:
            prior_scores.append(prior_sc)
            domain_aggs[domain]["prior"].append(prior_sc)
        if clf_sc:
            clf_scores.append(clf_sc)
            domain_aggs[domain]["clf"].append(clf_sc)

        base_g = None
        base_sc = None
        if eid in base_rows:
            base_g = _graph_from_row(base_rows[eid])
            base_sc = score_graph(adapted, base_g) if base_g.parse_ok else None
            if base_sc:
                base_scores.append(base_sc)
                domain_aggs[domain]["base"].append(base_sc)

        clf_meta = _meta(clf_row)
        prior_meta = _meta(prior_row)
        selected_titles = list(clf_meta.get("selected_titles") or prior_meta.get("selected_titles") or [])
        selected_ids = list(clf_meta.get("selected_concept_ids") or [])
        selected_n = {normalize_topic(t) for t in selected_titles}

        # Reconstruct candidate space from selected titles via inventory
        concepts: list[SelectedConcept] = []
        by_title = {c.title.casefold(): c for c in inventory.concepts}
        for title in selected_titles:
            c = by_title.get(title.casefold())
            if c:
                concepts.append(
                    SelectedConcept(concept_id=c.id, title=c.title, kind="REQUIRED")
                )
            elif selected_ids:
                pass
        if selected_ids and not concepts:
            by_id = inventory.by_id()
            for cid in selected_ids:
                if cid in by_id:
                    concepts.append(
                        SelectedConcept(
                            concept_id=cid, title=by_id[cid].title, kind="REQUIRED"
                        )
                    )
        pairs, cand_meta = generate_candidate_pairs(concepts)
        # Candidate space in title-normalized form
        cand_space = {_norm_edge(p.from_title, p.to_title) for p in pairs}

        gold_req = {_norm_edge(a, b) for a, b in adapted.required_dependencies}
        acceptable = {_norm_edge(a, b) for a, b in adapted.acceptable_dependencies}
        prior_edges = _edges_from_graph(prior_g) if prior_g.parse_ok else set()
        clf_edges = _edges_from_graph(clf_g) if clf_g.parse_ok else set()

        # Pair-level: gold required among selected endpoints only
        gold_in_space = {e for e in gold_req if e in cand_space}
        pred_req = clf_edges  # post-validation accepted edges
        tp = gold_in_space & pred_req
        fp = pred_req - gold_req - acceptable
        fn = gold_in_space - pred_req

        pair_stats["candidate_pair_count"] += int(
            clf_meta.get("candidate_meta", {}).get("candidate_pairs_evaluated")
            or cand_meta.get("candidate_pairs_evaluated")
            or len(pairs)
        )
        pair_stats["required_pair_count"] += len(gold_in_space)
        pair_stats["predicted_required_pair_count"] += int(
            clf_meta.get("predicted_required_pair_count") or len(pred_req)
        )
        pair_stats["tp"] += len(tp)
        pair_stats["fp"] += len(fp)
        pair_stats["fn"] += len(fn)
        pair_stats["uncertain"] += int(clf_meta.get("uncertain_count") or 0)
        pair_stats["invalid_pair_outputs"] += int(
            clf_meta.get("rejected_non_candidate_count") or 0
        )
        pair_stats["unknown_ids"] += int(clf_meta.get("unknown_id_rate_inputs") or 0)
        pair_stats["duplicates"] += int(clf_meta.get("duplicate_decision_count") or 0)

        uncertain_set: set[tuple[str, str]] = set()  # not stored per-pair in meta
        for e in fn:
            label = _classify_fn(
                e,
                selected=selected_n,
                predicted_required=pred_req,
                uncertain=uncertain_set,
                predicted_not_required=cand_space - pred_req,
                candidate_space=cand_space,
            )
            fn_counts[label] += 1
        for e in pred_req - gold_req:
            label = _classify_fp(
                e, gold_req=gold_req, acceptable=acceptable, generated_all=pred_req
            )
            fp_counts[label] += 1

        # Source-centered
        gold_by_src: dict[str, set[str]] = defaultdict(set)
        for a, b in gold_req:
            gold_by_src[a].add(b)
        for src, targets in gold_by_src.items():
            if src not in selected_n:
                continue
            key = src
            source_centered[key]["gold"] += len(targets)
            prior_t = {b for a, b in prior_edges if a == src}
            clf_t = {b for a, b in clf_edges if a == src}
            source_centered[key]["prior_hit"] += len(targets & prior_t)
            source_centered[key]["clf_hit"] += len(targets & clf_t)
            source_centered[key]["prior_fp"] += len(prior_t - targets - {normalize_topic(x) for x, _ in acceptable if normalize_topic(x) == src})
            source_centered[key]["clf_fp"] += len(clf_t - targets)

        prior_cmp = compare_graphs(adapted, prior_g) if prior_g.parse_ok else {}
        clf_cmp = compare_graphs(adapted, clf_g) if clf_g.parse_ok else {}

        def _recall(sc: Any) -> float | None:
            return None if sc is None else float(sc.required_edge_recall)

        pr = _recall(prior_sc)
        cr = _recall(clf_sc)
        if pr is None or cr is None:
            verdict = "UNCHANGED"
        elif cr > pr + 0.02:
            verdict = "IMPROVED"
        elif cr < pr - 0.02:
            verdict = "REGRESSED"
        else:
            verdict = "UNCHANGED"

        case_reports.append(
            {
                "example_id": eid,
                "domain": domain,
                "learning_goal": ex.goal,
                "selected_concepts": selected_titles,
                "candidate_pair_count": cand_meta.get("candidate_pairs_evaluated", len(pairs)),
                "gold_edges": [list(e) for e in sorted(gold_req)],
                "domain_prior_edges": [list(e) for e in sorted(prior_edges)],
                "pair_classifier_edges": [list(e) for e in sorted(clf_edges)],
                "true_positives": [list(e) for e in sorted(tp)],
                "false_positives": [list(e) for e in sorted(pred_req - gold_req)],
                "false_negatives": [list(e) for e in sorted(fn)],
                "uncertain_count": clf_meta.get("uncertain_count", 0),
                "cycle_rejected_edges": clf_meta.get("cycle_rejected_edges", 0),
                "latency_ms": {
                    "prior_total": prior_row.get("total_latency_ms") or prior_row.get("total_ms"),
                    "classifier_total": clf_row.get("total_latency_ms") or clf_row.get("total_ms"),
                    "classifier_stages": clf_meta.get("stage_latency_ms"),
                    "prior_stages": prior_meta.get("stage_latency_ms"),
                },
                "cost": {
                    "prior": prior_meta.get("estimated_cost_usd")
                    or prior_meta.get("cost_usd")
                    or prior_row.get("cost_usd"),
                    "classifier": clf_meta.get("estimated_cost_usd")
                    or clf_meta.get("cost_usd")
                    or clf_row.get("cost_usd"),
                    "classifier_estimated_cost_usd": clf_meta.get("estimated_cost_usd"),
                },
                "required_edge_recall": {"prior": pr, "classifier": cr},
                "required_edge_f1": {
                    "prior": None if prior_sc is None else prior_sc.required_edge_f1,
                    "classifier": None if clf_sc is None else clf_sc.required_edge_f1,
                },
                "verdict": verdict,
                "comparison_detail": {
                    "prior_missing": prior_cmp.get("missing_dependencies"),
                    "clf_missing": clf_cmp.get("missing_dependencies"),
                },
            }
        )

    prior_agg = aggregate_scores(prior_scores) if prior_scores else {}
    clf_agg = aggregate_scores(clf_scores) if clf_scores else {}
    base_agg = aggregate_scores(base_scores) if base_scores else {}

    def _delta(key: str) -> float | None:
        if key not in prior_agg or key not in clf_agg:
            return None
        return float(clf_agg[key]) - float(prior_agg[key])

    metric_keys = [
        "topic_f1",
        "required_edge_precision",
        "required_edge_recall",
        "required_edge_f1",
        "missing_required_edge_rate",
        "invalid_extra_edge_rate",
        "dependency_direction_error_rate",
        "hallucinated_topic_rate",
        "redundant_transitive_edge_rate",
    ]
    delta = {k: _delta(k) for k in metric_keys}

    pair_precision = _safe_rate(pair_stats["tp"], pair_stats["tp"] + pair_stats["fp"])
    pair_recall = _safe_rate(pair_stats["tp"], pair_stats["tp"] + pair_stats["fn"])
    pair_f1 = (
        _safe_rate(2 * pair_precision * pair_recall, pair_precision + pair_recall)
        if (pair_precision + pair_recall)
        else 0.0
    )

    # Diagnosis (exploratory; domain heterogeneity and cost matter)
    recall_delta = delta.get("required_edge_recall")
    f1_delta = delta.get("required_edge_f1")
    invalid_delta = delta.get("invalid_extra_edge_rate")
    domain_regressions = 0
    for _domain, buckets in domain_aggs.items():
        p_scores = buckets["prior"]
        c_scores = buckets["clf"]
        if not p_scores or not c_scores:
            continue
        p_agg = aggregate_scores(p_scores)
        c_agg = aggregate_scores(c_scores)
        if float(c_agg.get("required_edge_recall") or 0) + 0.02 < float(
            p_agg.get("required_edge_recall") or 0
        ):
            domain_regressions += 1
    if recall_delta is None:
        diagnosis = "INSUFFICIENT_EVIDENCE"
    elif (
        recall_delta >= 0.05
        and (f1_delta or 0) >= 0.03
        and (invalid_delta or 0) <= 0.10
        and domain_regressions == 0
    ):
        diagnosis = "SUPPORTED"
    elif recall_delta >= 0.03 or (f1_delta or 0) >= 0.03:
        # Material aggregate gain, but precision/cost/domain trade-offs remain
        diagnosis = "PARTIALLY_SUPPORTED"
    elif abs(recall_delta) < 0.02 and abs(f1_delta or 0) < 0.02:
        diagnosis = "NOT_SUPPORTED"
    else:
        diagnosis = "NOT_SUPPORTED"

    domain_summary = {}
    for domain, buckets in domain_aggs.items():
        domain_summary[domain] = {
            "prior": aggregate_scores(buckets["prior"]) if buckets["prior"] else {},
            "classifier": aggregate_scores(buckets["clf"]) if buckets["clf"] else {},
            "baseline": aggregate_scores(buckets["base"]) if buckets["base"] else {},
        }

    def _row_total_ms(row: dict[str, Any]) -> float:
        for key in ("total_latency_ms", "total_ms"):
            v = row.get(key)
            if isinstance(v, (int, float)):
                return float(v)
        return 0.0

    def _avg_meta(rows: dict[str, dict], key_path: list[str]) -> float | None:
        vals = []
        for eid in shared:
            row = rows.get(eid)
            if not row:
                continue
            meta = _meta(row)
            cur: Any = meta
            ok = True
            for k in key_path:
                if not isinstance(cur, dict) or k not in cur:
                    ok = False
                    break
                cur = cur[k]
            if ok and isinstance(cur, (int, float)):
                vals.append(float(cur))
        return (sum(vals) / len(vals)) if vals else None

    def _avg_row_cost(rows: dict[str, dict]) -> float | None:
        from app.evaluation.cost import estimate_cost_usd

        vals = []
        for eid in shared:
            row = rows.get(eid)
            if not row:
                continue
            meta = _meta(row)
            for key in ("estimated_cost_usd", "cost_usd"):
                if isinstance(meta.get(key), (int, float)):
                    vals.append(float(meta[key]))
                    break
            else:
                if isinstance(row.get("cost_usd"), (int, float)):
                    vals.append(float(row["cost_usd"]))
                else:
                    it = meta.get("input_tokens") or row.get("input_tokens")
                    ot = meta.get("output_tokens") or row.get("output_tokens")
                    model = meta.get("model") or "gpt-4o-mini"
                    if isinstance(it, int) and isinstance(ot, int):
                        est = estimate_cost_usd(str(model), it, ot)
                        if est is not None:
                            vals.append(float(est))
        return (sum(vals) / len(vals)) if vals else None

    cost_latency = {
        "prior_avg_total_ms": (
            sum(_row_total_ms(prior_rows[e]) for e in shared) / len(shared) if shared else None
        ),
        "classifier_avg_total_ms": (
            sum(_row_total_ms(clf_rows[e]) for e in shared) / len(shared) if shared else None
        ),
        "classifier_avg_selection_ms": _avg_meta(clf_rows, ["stage_latency_ms", "selection"]),
        "classifier_avg_classification_ms": _avg_meta(
            clf_rows, ["stage_latency_ms", "edge_classification"]
        ),
        "prior_avg_selection_ms": _avg_meta(prior_rows, ["stage_latency_ms", "selection"]),
        "prior_avg_dependency_ms": _avg_meta(
            prior_rows, ["stage_latency_ms", "dependency_generation"]
        ),
        "prior_avg_estimated_cost_usd": _avg_row_cost(prior_rows),
        "classifier_avg_estimated_cost_usd": _avg_row_cost(clf_rows),
    }

    tp_gain = (clf_agg.get("required_edge_recall") or 0) - (prior_agg.get("required_edge_recall") or 0)
    # Approximate additional correct required edges from scores' mean counts if available
    prior_hit = sum(len(c["true_positives"]) for c in case_reports if "true_positives" in c)
    # Recompute TP from prior vs clf matched
    prior_tp_total = 0
    clf_tp_total = 0
    for c in case_reports:
        gold = {tuple(e) for e in c["gold_edges"]}
        prior_e = {tuple(e) for e in c["domain_prior_edges"]}
        clf_e = {tuple(e) for e in c["pair_classifier_edges"]}
        prior_tp_total += len(gold & prior_e)
        clf_tp_total += len(gold & clf_e)
    extra_correct = clf_tp_total - prior_tp_total
    cost_delta = None
    if (
        cost_latency.get("classifier_avg_estimated_cost_usd") is not None
        and cost_latency.get("prior_avg_estimated_cost_usd") is not None
    ):
        cost_delta = (
            float(cost_latency["classifier_avg_estimated_cost_usd"])
            - float(cost_latency["prior_avg_estimated_cost_usd"])
        ) * len(shared)
    cost_per_edge = (
        (cost_delta / extra_correct) if cost_delta is not None and extra_correct > 0 else None
    )

    src_rows = []
    for src, st in sorted(source_centered.items(), key=lambda x: -x[1]["gold"]):
        src_rows.append(
            {
                "source": src,
                "gold_targets": st["gold"],
                "prior_target_recall": _safe_rate(st["prior_hit"], st["gold"]),
                "classifier_target_recall": _safe_rate(st["clf_hit"], st["gold"]),
                "prior_fp": st["prior_fp"],
                "classifier_fp": st["clf_fp"],
            }
        )

    out_dir = Path(output_dir) if output_dir else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    json_path = out_dir / f"{ts}_constrained_dependency_analysis.json"
    md_path = out_dir / f"{ts}_constrained_dependency_analysis.md"

    report = {
        "timestamp": ts,
        "artifact": str(target),
        "benchmark_configuration": {
            "model": payload.get("model"),
            "systems": list(systems.keys()),
            "matching": "curated_alias + edge_calibrated",
            "cases": shared,
            "inventory_domains": sorted({case_map[e] for e in shared}),
        },
        "aggregate": {
            "baseline": base_agg,
            "domain_curriculum_prior": prior_agg,
            "domain_prior_edge_classifier": clf_agg,
            "delta_classifier_minus_prior": delta,
        },
        "pair_metrics": {
            **pair_stats,
            "pair_precision": pair_precision,
            "pair_recall": pair_recall,
            "pair_f1": pair_f1,
            "uncertain_rate": _safe_rate(
                pair_stats["uncertain"], max(1, pair_stats["candidate_pair_count"])
            ),
            "invalid_pair_output_rate": _safe_rate(
                pair_stats["invalid_pair_outputs"],
                max(1, pair_stats["candidate_pair_count"]),
            ),
            "unknown_id_rate": _safe_rate(
                pair_stats["unknown_ids"], max(1, pair_stats["candidate_pair_count"])
            ),
            "duplicate_decision_rate": _safe_rate(
                pair_stats["duplicates"], max(1, pair_stats["candidate_pair_count"])
            ),
        },
        "failure_categories": {"false_negative": dict(fn_counts), "false_positive": dict(fp_counts)},
        "source_centered": src_rows,
        "domain_summary": domain_summary,
        "cost_latency": {
            **cost_latency,
            "extra_correct_required_edges": extra_correct,
            "cost_per_additional_correct_required_edge": cost_per_edge,
            "recall_delta": recall_delta,
            "tp_gain_proxy": tp_gain,
        },
        "cases": case_reports,
        "final_diagnosis": diagnosis,
        "production_recommendation": "Remain experimental / opt-in; do not promote to production default.",
    }
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    lines = [
        f"# Constrained Dependency Classification Analysis — {ts}",
        "",
        f"Artifact: `{target}`",
        f"Cases: {len(shared)}",
        f"Diagnosis: **{diagnosis}**",
        "",
        "## Main results (Domain Prior → Edge Classifier)",
        "",
        "| Metric | Domain Prior | Edge Classifier | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for k in metric_keys:
        pv = prior_agg.get(k)
        cv = clf_agg.get(k)
        dv = delta.get(k)
        lines.append(
            f"| {k} | {_fmt(pv)} | {_fmt(cv)} | {_fmt(dv)} |"
        )
    lines += [
        "",
        "## Pair metrics",
        "",
        f"- Pair Precision: `{pair_precision:.3f}`",
        f"- Pair Recall: `{pair_recall:.3f}`",
        f"- Pair F1: `{pair_f1:.3f}`",
        f"- Uncertain rate: `{report['pair_metrics']['uncertain_rate']:.3f}`",
        f"- Invalid pair output rate: `{report['pair_metrics']['invalid_pair_output_rate']:.3f}`",
        f"- Unknown ID rate: `{report['pair_metrics']['unknown_id_rate']:.3f}`",
        "",
        "## Failure categories",
        "",
        f"- FN: `{dict(fn_counts)}`",
        f"- FP: `{dict(fp_counts)}`",
        "",
        "## Cost / latency",
        "",
        f"- Prior avg total ms: `{_fmt(cost_latency.get('prior_avg_total_ms'))}`",
        f"- Classifier avg total ms: `{_fmt(cost_latency.get('classifier_avg_total_ms'))}`",
        f"- Classifier selection ms: `{_fmt(cost_latency.get('classifier_avg_selection_ms'))}`",
        f"- Classifier classification ms: `{_fmt(cost_latency.get('classifier_avg_classification_ms'))}`",
        f"- Prior est. cost USD: `{_fmt(cost_latency.get('prior_avg_estimated_cost_usd'))}`",
        f"- Classifier est. cost USD: `{_fmt(cost_latency.get('classifier_avg_estimated_cost_usd'))}`",
        f"- Extra correct required edges: `{extra_correct}`",
        f"- Cost per additional correct required edge: `{_fmt(cost_per_edge)}`",
        "",
        "## Per-case verdicts",
        "",
    ]
    for c in case_reports:
        lines += [
            f"### {c['example_id']} ({c['domain']}) — **{c['verdict']}**",
            "",
            f"- Goal: {c['learning_goal']}",
            f"- Selected ({len(c['selected_concepts'])}): {', '.join(c['selected_concepts'])}",
            f"- Candidate pairs: {c['candidate_pair_count']}",
            f"- Required Edge Recall: prior=`{_fmt(c['required_edge_recall']['prior'])}` "
            f"classifier=`{_fmt(c['required_edge_recall']['classifier'])}`",
            f"- Required Edge F1: prior=`{_fmt(c['required_edge_f1']['prior'])}` "
            f"classifier=`{_fmt(c['required_edge_f1']['classifier'])}`",
            f"- TP={len(c['true_positives'])} FP={len(c['false_positives'])} "
            f"FN={len(c['false_negatives'])} cycles_rejected={c['cycle_rejected_edges']}",
            f"- Gold edges: {c['gold_edges']}",
            f"- Domain Prior edges: {c['domain_prior_edges']}",
            f"- Pair Classifier edges: {c['pair_classifier_edges']}",
            "",
        ]
    lines += [
        "## Domain-level Required Edge Recall",
        "",
    ]
    for domain, block in sorted(domain_summary.items()):
        lines.append(
            f"- **{domain}**: prior=`{_fmt(block['prior'].get('required_edge_recall'))}` "
            f"classifier=`{_fmt(block['classifier'].get('required_edge_recall'))}` "
            f"baseline=`{_fmt(block['baseline'].get('required_edge_recall'))}`"
        )
    lines += [
        "",
        "## Production recommendation",
        "",
        report["production_recommendation"],
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)
