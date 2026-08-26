"""Final 40-case system comparison (offline analysis + optional live driver).

Compares production baseline vs domain_curriculum_prior vs domain_prior_edge_classifier.
Does not change generation systems. Selection is independent (each system runs as implemented).
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.curriculum.inventory import load_case_domain_map
from app.evaluation.cost import estimate_cost_usd
from app.evaluation.dataset import load_dataset
from app.evaluation.edge_ambiguity import adapt_example_for_edge_mode
from app.evaluation.inspect import _graph_from_row
from app.evaluation.latency import summarize_latencies_ms
from app.evaluation.metrics import (
    aggregate_scores,
    compare_graphs,
    normalize_topic,
    score_graph,
)
from app.evaluation.schemas import EvalExample, GeneratedGraph, GraphQualityScores
from app.evaluation.reliability import run_reliability_benchmark
from app.services.generation_strategy import resolve_generation_strategy

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BENCH = _REPO_ROOT / "results" / "benchmarks"
DEFAULT_DS = _REPO_ROOT / "data" / "eval" / "learning_graph_quality_v1.jsonl"

REQUIRED_SYSTEMS = ("synapse", "domain_curriculum_prior", "domain_prior_edge_classifier")
SYSTEM_LABELS = {
    "synapse": "baseline",
    "domain_curriculum_prior": "domain_prior",
    "domain_prior_edge_classifier": "edge_classifier",
}
PRIMARY_METRICS = (
    "topic_precision",
    "topic_recall",
    "topic_f1",
    "required_edge_precision",
    "required_edge_recall",
    "required_edge_f1",
    "missing_required_edge_rate",
    "invalid_extra_edge_rate",
    "dependency_direction_error_rate",
    "hallucinated_topic_rate",
    "redundant_transitive_edge_rate",
)
REGRESSION_EPS = 0.02


class FinalBenchmarkError(ValueError):
    """Artifact does not meet the frozen final-benchmark contract."""


def relative_delta(new: float, old: float) -> float | None:
    """(new-old)/|old| when old != 0, else None."""
    if old == 0:
        return None
    return (new - old) / abs(old)


def regression_label(experimental: float, baseline: float, *, eps: float = REGRESSION_EPS) -> str:
    if experimental > baseline + eps:
        return "IMPROVED"
    if experimental < baseline - eps:
        return "REGRESSED"
    return "UNCHANGED"


def case_winner(scores: dict[str, float]) -> str:
    """Winner by Required Edge F1. Tie → TIE."""
    items = sorted(scores.items(), key=lambda kv: -kv[1])
    if not items:
        return "TIE"
    if len(items) == 1:
        return items[0][0]
    if abs(items[0][1] - items[1][1]) < 1e-9:
        return "TIE"
    return items[0][0]


def _percentile(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


def _dist(vals: list[float]) -> dict[str, float | int | None]:
    if not vals:
        return {"n": 0, "mean": None, "median": None, "std_dev": None, "min": None, "max": None}
    if len(vals) == 1:
        v = vals[0]
        return {"n": 1, "mean": v, "median": v, "std_dev": None, "min": v, "max": v}
    return {
        "n": len(vals),
        "mean": float(statistics.fmean(vals)),
        "median": float(statistics.median(vals)),
        "std_dev": float(statistics.pstdev(vals)),
        "min": float(min(vals)),
        "max": float(max(vals)),
    }


def _empty_graph() -> GeneratedGraph:
    return GeneratedGraph(topics=[], dependencies=[], parse_ok=False)


def _score_row(example: EvalExample, row: dict[str, Any]) -> GraphQualityScores:
    graph = _graph_from_row(row)
    if not graph.parse_ok and not graph.topics:
        graph = _empty_graph()
        return score_graph(example, graph)
    return score_graph(example, graph)


def _row_cost_usd(row: dict[str, Any], model: str) -> float:
    meta = row.get("generation_meta") or {}
    for key in ("estimated_cost_usd", "cost_usd"):
        v = meta.get(key)
        if isinstance(v, (int, float)):
            return float(v)
    v = row.get("cost_usd")
    if isinstance(v, (int, float)):
        return float(v)
    it = meta.get("input_tokens") if meta.get("input_tokens") is not None else row.get("input_tokens")
    ot = meta.get("output_tokens") if meta.get("output_tokens") is not None else row.get("output_tokens")
    if isinstance(it, int) and isinstance(ot, int):
        est = estimate_cost_usd(model, it, ot)
        if est is not None:
            return float(est)
    return 0.0


def validate_final_artifact(
    payload: dict[str, Any],
    *,
    expected_cases: int = 40,
    expected_generations: int | None = None,
) -> None:
    systems = payload.get("systems") or {}
    missing = [s for s in REQUIRED_SYSTEMS if s not in systems]
    if missing:
        raise FinalBenchmarkError(f"Missing systems {missing}; found {list(systems)}")
    for sys in REQUIRED_SYSTEMS:
        rows = systems[sys].get("example_results") or []
        ids = {str(r.get("example_id")) for r in rows}
        if len(ids) != expected_cases:
            raise FinalBenchmarkError(
                f"{sys} has {len(ids)} cases, expected {expected_cases}"
            )
        gens = sorted({int(r.get("generation_index", r.get("repetition", 0))) for r in rows})
        if expected_generations is not None and len(gens) != expected_generations:
            raise FinalBenchmarkError(
                f"{sys} has generations {gens}, expected {expected_generations}"
            )
        for r in rows:
            if "seed" not in r:
                raise FinalBenchmarkError(f"{sys} row missing seed field")


def _domain_for_case(example: EvalExample, case_map: dict[str, str]) -> str:
    return case_map.get(example.id) or f"unmapped:{example.category}"


def _attr_counts(example: EvalExample, graph: GeneratedGraph) -> dict[str, int]:
    cmp = compare_graphs(example, graph)
    matched_n = {normalize_topic(t) for t in cmp["matched_required_topics"]}
    both_present_miss = 0
    endpoint_miss = 0
    for frm, to in cmp["missing_dependencies"]:
        nf, nt = normalize_topic(frm), normalize_topic(to)
        if nf in matched_n and nt in matched_n:
            both_present_miss += 1
        else:
            endpoint_miss += 1
    return {
        "missing_required_topics": len(cmp["missing_topics"]),
        "missing_required_edges": len(cmp["missing_dependencies"]),
        "both_endpoints_present_omissions": both_present_miss,
        "endpoint_missing_omissions": endpoint_miss,
        "invalid_extras": len(cmp["extra_dependencies"]),
        "direction_errors": len(cmp["reversed_dependencies"]),
        "hallucinated_topics": len(cmp["extra_topics"]),
    }


def build_final_comparison(
    artifact_path: str | Path,
    *,
    dataset_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    reliability: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    target = Path(artifact_path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    ds_path = Path(dataset_path) if dataset_path else DEFAULT_DS
    examples = {ex.id: ex for ex in load_dataset(ds_path)}
    expected_n = len(examples)
    gens = int(payload.get("generations") or payload.get("repetitions") or 1)
    validate_final_artifact(payload, expected_cases=expected_n, expected_generations=gens)

    case_map = load_case_domain_map()
    model = str(payload.get("model") or "gpt-4o-mini")
    systems_raw = payload["systems"]

    # Per system: list of (example_id, gen, scores, latency, cost, meta, row)
    per_sys: dict[str, list[dict[str, Any]]] = {s: [] for s in REQUIRED_SYSTEMS}
    for sys in REQUIRED_SYSTEMS:
        for row in systems_raw[sys].get("example_results") or []:
            eid = str(row.get("example_id"))
            ex = examples[eid]
            adapted = adapt_example_for_edge_mode(
                ex, "edge_calibrated", topic_matching_mode="curated_alias"
            )
            scores = _score_row(adapted, row)
            if not (0.0 <= scores.missing_required_edge_rate <= 1.0):
                raise FinalBenchmarkError("missing_required_edge_rate out of [0,1]")
            meta = dict(row.get("generation_meta") or {})
            per_sys[sys].append(
                {
                    "example_id": eid,
                    "generation_index": int(row.get("generation_index", row.get("repetition", 0))),
                    "seed": row.get("seed") if row.get("seed") is not None else meta.get("seed"),
                    "seed_supported": row.get("seed_supported", meta.get("seed_supported")),
                    "scores": scores,
                    "total_latency_ms": float(row.get("total_latency_ms") or 0.0),
                    "llm_latency_ms": float(row.get("llm_latency_ms") or 0.0),
                    "deterministic_latency_ms": float(row.get("deterministic_latency_ms") or 0.0),
                    "stage_latency_ms": meta.get("stage_latency_ms") or {},
                    "cost_usd": _row_cost_usd(row, model),
                    "parse_ok": bool(row.get("parse_ok")),
                    "inventory_version": meta.get("inventory_version")
                    or meta.get("curriculum_inventory_version"),
                    "prompt_version": meta.get("prompt_version")
                    or payload.get("prompt_version"),
                    "prompt_variant": meta.get("edge_classifier_prompt_variant")
                    or meta.get("prompt_variant"),
                    "domain_status": meta.get("domain_status"),
                    "row": row,
                    "adapted": adapted,
                }
            )

    def case_means(sys: str) -> dict[str, GraphQualityScores]:
        by: dict[str, list[GraphQualityScores]] = defaultdict(list)
        for rec in per_sys[sys]:
            by[rec["example_id"]].append(rec["scores"])
        out = {}
        for eid, scs in by.items():
            agg = aggregate_scores(scs)
            # reconstruct a scores-like mean via first + overwrite primary attrs
            proto = scs[0]
            kwargs = {f: getattr(proto, f) for f in proto.__dataclass_fields__}
            for k, v in agg.items():
                if k in kwargs and k != "failures":
                    kwargs[k] = v
            kwargs["failures"] = []
            out[eid] = GraphQualityScores(**kwargs)
        return out

    means = {s: case_means(s) for s in REQUIRED_SYSTEMS}

    def sys_agg(sys: str) -> dict[str, float]:
        return aggregate_scores(list(means[sys].values()))

    overall = {s: sys_agg(s) for s in REQUIRED_SYSTEMS}

    def pairwise(a: str, b: str) -> dict[str, Any]:
        oa, ob = overall[a], overall[b]
        out = {}
        for m in PRIMARY_METRICS:
            av, bv = float(oa.get(m) or 0), float(ob.get(m) or 0)
            out[m] = {
                "left": av,
                "right": bv,
                "absolute_delta": bv - av,
                "relative_delta": relative_delta(bv, av),
            }
        return out

    deltas = {
        "domain_prior_vs_baseline": pairwise("synapse", "domain_curriculum_prior"),
        "edge_classifier_vs_domain_prior": pairwise(
            "domain_curriculum_prior", "domain_prior_edge_classifier"
        ),
        "edge_classifier_vs_baseline": pairwise("synapse", "domain_prior_edge_classifier"),
    }

    # Case-level
    case_rows = []
    win_counts = Counter()
    prior_vs_base = Counter()
    clf_vs_prior = Counter()
    clf_vs_base = Counter()
    for eid, ex in sorted(examples.items()):
        bf = means["synapse"][eid].required_edge_f1
        pf = means["domain_curriculum_prior"][eid].required_edge_f1
        cf = means["domain_prior_edge_classifier"][eid].required_edge_f1
        winner = case_winner({"baseline": bf, "domain_prior": pf, "edge_classifier": cf})
        p_lab = regression_label(pf, bf)
        c_lab = regression_label(cf, bf)
        c_vs_p = regression_label(cf, pf)
        prior_vs_base[p_lab] += 1
        clf_vs_base[c_lab] += 1
        clf_vs_prior[c_vs_p] += 1
        win_counts[winner] += 1
        case_rows.append(
            {
                "case_id": eid,
                "category": ex.category,
                "curriculum_domain": case_map.get(eid),
                "baseline_edge_f1": bf,
                "prior_edge_f1": pf,
                "classifier_edge_f1": cf,
                "winner": winner,
                "prior_vs_baseline": p_lab,
                "classifier_vs_baseline": c_lab,
                "classifier_vs_prior": c_vs_p,
            }
        )

    # Domain aggregation (curriculum domain if mapped else dataset category)
    domain_scores: dict[str, dict[str, list[GraphQualityScores]]] = defaultdict(
        lambda: {s: [] for s in REQUIRED_SYSTEMS}
    )
    domain_lat: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {s: [] for s in REQUIRED_SYSTEMS}
    )
    domain_cost: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {s: [] for s in REQUIRED_SYSTEMS}
    )
    for sys in REQUIRED_SYSTEMS:
        for rec in per_sys[sys]:
            ex = examples[rec["example_id"]]
            d = _domain_for_case(ex, case_map)
            domain_scores[d][sys].append(rec["scores"])
            domain_lat[d][sys].append(rec["total_latency_ms"])
            domain_cost[d][sys].append(rec["cost_usd"])

    domain_summary = {}
    for d, by_sys in sorted(domain_scores.items()):
        domain_summary[d] = {}
        for sys in REQUIRED_SYSTEMS:
            scs = by_sys[sys]
            # mean per case then aggregate: group by example
            by_eid: dict[str, list[GraphQualityScores]] = defaultdict(list)
            # recover eids from per_sys
            eids = {r["example_id"] for r in per_sys[sys] if _domain_for_case(examples[r["example_id"]], case_map) == d}
            case_mean_scores = [means[sys][eid] for eid in sorted(eids)]
            lat = domain_lat[d][sys]
            cost = domain_cost[d][sys]
            domain_summary[d][sys] = {
                "metrics": aggregate_scores(case_mean_scores),
                "p50_latency_ms": _percentile(lat, 50),
                "mean_latency_ms": (sum(lat) / len(lat)) if lat else 0.0,
                "estimated_cost_per_case": (
                    sum(cost) / max(1, len(eids)) if cost else 0.0
                ),
                "n_cases": len(eids),
            }

    # Stability across generations (per case then dist of case-means already; also gen-level)
    stability = {}
    for sys in REQUIRED_SYSTEMS:
        by_metric: dict[str, list[float]] = {m: [] for m in (
            "topic_f1",
            "required_edge_f1",
            "required_edge_recall",
            "missing_required_edge_rate",
            "invalid_extra_edge_rate",
            "hallucinated_topic_rate",
        )}
        for rec in per_sys[sys]:
            sc = rec["scores"]
            by_metric["topic_f1"].append(sc.topic_f1)
            by_metric["required_edge_f1"].append(sc.required_edge_f1)
            by_metric["required_edge_recall"].append(sc.required_edge_recall)
            by_metric["missing_required_edge_rate"].append(sc.missing_required_edge_rate)
            by_metric["invalid_extra_edge_rate"].append(sc.invalid_extra_edge_rate)
            by_metric["hallucinated_topic_rate"].append(sc.hallucinated_topic_rate)
        stability[sys] = {m: _dist(vs) for m, vs in by_metric.items()}

    # Latency / cost
    def lat_cost(sys: str) -> dict[str, Any]:
        recs = per_sys[sys]
        totals = [r["total_latency_ms"] for r in recs]
        llms = [r["llm_latency_ms"] for r in recs]
        dets = [r["deterministic_latency_ms"] for r in recs]
        costs = [r["cost_usd"] for r in recs]
        n_cases = expected_n
        sel = [
            float(r["stage_latency_ms"]["selection"])
            for r in recs
            if isinstance(r["stage_latency_ms"].get("selection"), (int, float))
        ]
        dep = [
            float(r["stage_latency_ms"]["dependency_generation"])
            for r in recs
            if isinstance(r["stage_latency_ms"].get("dependency_generation"), (int, float))
        ]
        clf = [
            float(r["stage_latency_ms"]["edge_classification"])
            for r in recs
            if isinstance(r["stage_latency_ms"].get("edge_classification"), (int, float))
        ]
        parse_ok = sum(1 for r in recs if r["parse_ok"]) / len(recs) if recs else 0.0
        return {
            "total": summarize_latencies_ms(totals),
            "llm": summarize_latencies_ms(llms),
            "deterministic": summarize_latencies_ms(dets),
            "selection_ms": summarize_latencies_ms(sel) if sel else None,
            "dependency_generation_ms": summarize_latencies_ms(dep) if dep else None,
            "edge_classification_ms": summarize_latencies_ms(clf) if clf else None,
            "estimated_cost_total_usd": sum(costs),
            "estimated_cost_per_generation_usd": (sum(costs) / len(costs)) if costs else 0.0,
            "estimated_cost_per_case_usd": (sum(costs) / n_cases) if n_cases else 0.0,
            "parse_ok_rate": parse_ok,
        }

    latency_cost = {s: lat_cost(s) for s in REQUIRED_SYSTEMS}

    # Root-cause (reuse compare_graphs; mean per generation then sum)
    attribution = {s: Counter() for s in REQUIRED_SYSTEMS}
    for sys in REQUIRED_SYSTEMS:
        for rec in per_sys[sys]:
            g = _graph_from_row(rec["row"])
            if not g.parse_ok and not g.topics:
                g = _empty_graph()
            counts = _attr_counts(rec["adapted"], g)
            attribution[sys].update(counts)
    attribution_out = {s: dict(c) for s, c in attribution.items()}

    # Mapped-only slice
    mapped_ids = set(case_map)
    mapped_overall = {}
    for sys in REQUIRED_SYSTEMS:
        mapped_overall[sys] = aggregate_scores(
            [means[sys][eid] for eid in mapped_ids if eid in means[sys]]
        )

    # Inventory / prompt recording check
    inventory_versions = {
        rec["inventory_version"]
        for rec in per_sys["domain_curriculum_prior"] + per_sys["domain_prior_edge_classifier"]
        if rec["inventory_version"]
    }
    prompt_versions = {
        rec["prompt_variant"]
        for rec in per_sys["domain_prior_edge_classifier"]
        if rec["prompt_variant"]
    }

    # Extra correct required edges vs baseline (sum of mean matched recall * gold n approx)
    extra_edges = {}
    for sys in ("domain_curriculum_prior", "domain_prior_edge_classifier"):
        gain = 0.0
        for eid, ex in examples.items():
            gold_n = len(ex.required_dependency_list()) or 0
            br = means["synapse"][eid].required_edge_recall
            er = means[sys][eid].required_edge_recall
            gain += (er - br) * gold_n
        extra_edges[sys] = gain

    cost_per_extra = {}
    for sys, gain in extra_edges.items():
        extra_cost = (
            latency_cost[sys]["estimated_cost_total_usd"]
            - latency_cost["synapse"]["estimated_cost_total_usd"]
        )
        cost_per_extra[sys] = (extra_cost / gain) if gain > 0 else None

    rel = reliability or {}
    rel_metrics = (rel.get("metrics") or {}) if isinstance(rel, dict) else {}

    # Architecture decision (deterministic, documented)
    base_f1 = overall["synapse"]["required_edge_f1"]
    prior_f1 = overall["domain_curriculum_prior"]["required_edge_f1"]
    clf_f1 = overall["domain_prior_edge_classifier"]["required_edge_f1"]
    mapped_prior_f1 = mapped_overall["domain_curriculum_prior"]["required_edge_f1"]
    mapped_base_f1 = mapped_overall["synapse"]["required_edge_f1"]
    mapped_clf_f1 = mapped_overall["domain_prior_edge_classifier"]["required_edge_f1"]
    prior_parse = latency_cost["domain_curriculum_prior"]["parse_ok_rate"]
    clf_parse = latency_cost["domain_prior_edge_classifier"]["parse_ok_rate"]
    prior_reg = prior_vs_base.get("REGRESSED", 0)
    clf_reg = clf_vs_base.get("REGRESSED", 0)

    # Domain prior cannot cover unmapped cases (expected). Production candidate only if
    # mapped slice is strong AND parse rate / operational model is acceptable.
    prior_status = "EXPERIMENTAL_ONLY"
    clf_status = "REJECTED"
    overall_rec = "KEEP_DOMAIN_PRIOR_EXPERIMENTAL"

    mapped_gain = mapped_prior_f1 - mapped_base_f1
    mapped_clf_gain = mapped_clf_f1 - mapped_prior_f1
    if prior_parse < 0.2 and mapped_gain >= 0.05 and prior_reg <= expected_n * 0.5:
        prior_status = "EXPERIMENTAL_ONLY"
        overall_rec = "KEEP_DOMAIN_PRIOR_EXPERIMENTAL"
    if mapped_clf_gain >= 0.05 and clf_f1 >= prior_f1:
        clf_status = "EXPERIMENTAL_ONLY"
    # Never auto-promote: inventories cover 6/40 cases.
    if prior_parse >= 0.95 and mapped_gain >= 0.08 and prior_reg < 8:
        prior_status = "PRODUCTION_CANDIDATE"
        overall_rec = "ADOPT_DOMAIN_CURRICULUM_PRIOR"
    if clf_parse >= 0.95 and mapped_clf_gain >= 0.08:
        clf_status = "PRODUCTION_CANDIDATE"
        overall_rec = "ADOPT_DOMAIN_PRIOR_EDGE_CLASSIFIER"
    if mapped_gain < 0.02 and mapped_clf_gain < 0.02:
        overall_rec = "KEEP_BASELINE"
        prior_status = "EXPERIMENTAL_ONLY"
        clf_status = "REJECTED"

    # Override: coverage on 6/40 is an operational blocker for production.
    if prior_parse < 0.5:
        if prior_status == "PRODUCTION_CANDIDATE":
            prior_status = "EXPERIMENTAL_ONLY"
        if overall_rec == "ADOPT_DOMAIN_CURRICULUM_PRIOR":
            overall_rec = "KEEP_DOMAIN_PRIOR_EXPERIMENTAL"
    if clf_parse < 0.5:
        if clf_status == "PRODUCTION_CANDIDATE":
            clf_status = "EXPERIMENTAL_ONLY"
        if overall_rec == "ADOPT_DOMAIN_PRIOR_EDGE_CLASSIFIER":
            overall_rec = "KEEP_BOTH_EXPERIMENTAL" if mapped_gain >= 0.05 else "KEEP_DOMAIN_PRIOR_EXPERIMENTAL"

    if mapped_gain >= 0.05 and mapped_clf_gain < 0.03:
        clf_status = "REJECTED" if mapped_clf_gain <= 0 else "EXPERIMENTAL_ONLY"
        if prior_parse < 0.5:
            overall_rec = "KEEP_DOMAIN_PRIOR_EXPERIMENTAL"
            prior_status = "EXPERIMENTAL_ONLY"

    production_invariants = {
        "production_default": resolve_generation_strategy(None),
        "gold_not_used_at_runtime": True,
        "dag_validation_unchanged": True,
        "reliability_metrics": rel_metrics,
        "reliability_ok": all(
            float(rel_metrics.get(k) or 0) >= 1.0
            for k in (
                "validation_catch_rate",
                "cycle_prevention_rate",
                "transaction_integrity_rate",
                "rollback_correctness_rate",
            )
            if rel_metrics
        )
        if rel_metrics
        else None,
    }

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    report = {
        "timestamp": ts,
        "benchmark_type": "final_40_case_comparison",
        "source_artifact": str(target),
        "frozen_configuration": {
            "dataset": str(ds_path),
            "cases": expected_n,
            "generations": gens,
            "model": model,
            "matching": "curated_alias",
            "edge_evaluation": "edge_calibrated",
            "systems": list(REQUIRED_SYSTEMS),
            "edge_classifier_prompt": "edge_classifier_baseline",
            "inventories": "frozen v1",
            "selection": "INDEPENDENT",
            "selection_note": (
                "domain_curriculum_prior and domain_prior_edge_classifier each run their "
                "own concept-selection LLM call. Shared selection is not used."
            ),
            "mapped_case_count": len(mapped_ids),
            "inventory_versions": sorted(inventory_versions),
            "edge_classifier_prompt_variants_seen": sorted(prompt_versions),
        },
        "overall": overall,
        "mapped_subset_overall": mapped_overall,
        "deltas": deltas,
        "domain_summary": domain_summary,
        "stability": stability,
        "latency_cost": latency_cost,
        "cost_per_additional_correct_required_edge": cost_per_extra,
        "extra_correct_required_edges_vs_baseline": extra_edges,
        "case_results": case_rows,
        "win_counts": dict(win_counts),
        "regression_counts": {
            "prior_vs_baseline": dict(prior_vs_base),
            "classifier_vs_baseline": dict(clf_vs_base),
            "classifier_vs_prior": dict(clf_vs_prior),
        },
        "attribution": attribution_out,
        "reliability": rel_metrics,
        "production_invariants": production_invariants,
        "architecture_decision": {
            "baseline": "PRODUCTION",
            "domain_curriculum_prior": prior_status,
            "domain_prior_edge_classifier": clf_status,
            "overall_recommendation": overall_rec,
            "rationale": (
                f"Full-set prior parse_ok={prior_parse:.2f} (inventories cover "
                f"{len(mapped_ids)}/{expected_n} cases). Mapped Required Edge F1: "
                f"baseline={mapped_base_f1:.3f} prior={mapped_prior_f1:.3f} "
                f"classifier={mapped_clf_f1:.3f}."
            ),
        },
        "generations_exploratory": gens < 3,
    }

    out_dir = Path(output_dir) if output_dir else DEFAULT_BENCH
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{ts}_final_40_case_comparison.json"
    md_path = out_dir / f"{ts}_final_40_case_comparison.md"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    md_path.write_text(_render_md(report), encoding="utf-8")
    return json_path, md_path


def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _render_md(report: dict[str, Any]) -> str:
    o = report["overall"]
    d = report["architecture_decision"]
    lc = report["latency_cost"]
    lines = [
        f"# Final 40-Case System Evaluation — {report['timestamp']}",
        "",
        "## 1. Executive summary",
        "",
        f"- Overall recommendation: **{d['overall_recommendation']}**",
        f"- Baseline: **PRODUCTION**",
        f"- Domain curriculum prior: **{d['domain_curriculum_prior']}**",
        f"- Domain prior edge classifier: **{d['domain_prior_edge_classifier']}**",
        f"- Selection: **{report['frozen_configuration']['selection']}**",
        f"- Generations: {report['frozen_configuration']['generations']}"
        + (" (exploratory n=1)" if report.get("generations_exploratory") else " (n=3 preferred)"),
        f"- {d['rationale']}",
        "",
        "## 2. Frozen configuration",
        "",
        json.dumps(report["frozen_configuration"], indent=2),
        "",
        "## 3. Overall results",
        "",
        "| Metric | Baseline | Domain Prior | Edge Classifier |",
        "| --- | ---: | ---: | ---: |",
    ]
    table_keys = [
        ("Topic F1", "topic_f1"),
        ("Required Edge Precision", "required_edge_precision"),
        ("Required Edge Recall", "required_edge_recall"),
        ("Required Edge F1", "required_edge_f1"),
        ("Missing Required Edge Rate", "missing_required_edge_rate"),
        ("Invalid Extra Edge Rate", "invalid_extra_edge_rate"),
        ("Hallucinated Topic Rate", "hallucinated_topic_rate"),
        ("Direction Error Rate", "dependency_direction_error_rate"),
        ("Transitive Redundancy Rate", "redundant_transitive_edge_rate"),
    ]
    for label, key in table_keys:
        lines.append(
            f"| {label} | {_fmt(o['synapse'].get(key))} | "
            f"{_fmt(o['domain_curriculum_prior'].get(key))} | "
            f"{_fmt(o['domain_prior_edge_classifier'].get(key))} |"
        )
    lines.append(
        f"| p50 Latency (ms) | {_fmt(lc['synapse']['total'].get('p50_ms'))} | "
        f"{_fmt(lc['domain_curriculum_prior']['total'].get('p50_ms'))} | "
        f"{_fmt(lc['domain_prior_edge_classifier']['total'].get('p50_ms'))} |"
    )
    lines.append(
        f"| Estimated Cost / Case | {_fmt(lc['synapse']['estimated_cost_per_case_usd'])} | "
        f"{_fmt(lc['domain_curriculum_prior']['estimated_cost_per_case_usd'])} | "
        f"{_fmt(lc['domain_prior_edge_classifier']['estimated_cost_per_case_usd'])} |"
    )
    lines.append(
        f"| Parse-ok rate | {_fmt(lc['synapse']['parse_ok_rate'])} | "
        f"{_fmt(lc['domain_curriculum_prior']['parse_ok_rate'])} | "
        f"{_fmt(lc['domain_prior_edge_classifier']['parse_ok_rate'])} |"
    )
    lines += [
        "",
        "Mapped subset (inventory-covered cases only):",
        "",
        "| Metric | Baseline | Domain Prior | Edge Classifier |",
        "| --- | ---: | ---: | ---: |",
    ]
    mo = report["mapped_subset_overall"]
    for label, key in table_keys[:5]:
        lines.append(
            f"| {label} | {_fmt(mo['synapse'].get(key))} | "
            f"{_fmt(mo['domain_curriculum_prior'].get(key))} | "
            f"{_fmt(mo['domain_prior_edge_classifier'].get(key))} |"
        )
    lines += ["", "## 4. Domain results", "", "| Domain | System | Topic F1 | Edge Recall | Edge F1 | Invalid Extras | p50 | Cost |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for domain, block in report["domain_summary"].items():
        for sys, label in (
            ("synapse", "baseline"),
            ("domain_curriculum_prior", "domain_prior"),
            ("domain_prior_edge_classifier", "edge_classifier"),
        ):
            m = block[sys]["metrics"]
            lines.append(
                f"| {domain} | {label} | {_fmt(m.get('topic_f1'))} | "
                f"{_fmt(m.get('required_edge_recall'))} | {_fmt(m.get('required_edge_f1'))} | "
                f"{_fmt(m.get('invalid_extra_edge_rate'))} | {_fmt(block[sys]['p50_latency_ms'])} | "
                f"{_fmt(block[sys]['estimated_cost_per_case'])} |"
            )
    lines += ["", "## 5. Latency / cost", ""]
    for sys, label in SYSTEM_LABELS.items():
        block = lc[sys]
        lines.append(
            f"- **{label}**: p50={_fmt(block['total'].get('p50_ms'))} "
            f"mean={_fmt(block['total'].get('mean_ms'))} "
            f"p95={_fmt(block['total'].get('p95_ms'))} "
            f"cost/case={_fmt(block['estimated_cost_per_case_usd'])} "
            f"parse_ok={_fmt(block['parse_ok_rate'])}"
        )
    lines += ["", "## 6. Stability", ""]
    for sys, label in SYSTEM_LABELS.items():
        st = report["stability"][sys]
        lines.append(f"### {label}")
        for m, dist in st.items():
            lines.append(
                f"- {m}: mean={_fmt(dist['mean'])} median={_fmt(dist['median'])} "
                f"std={_fmt(dist['std_dev'])} min={_fmt(dist['min'])} max={_fmt(dist['max'])}"
            )
    lines += [
        "",
        "## 7. Case-level improvements/regressions",
        "",
        f"- Win counts: `{report['win_counts']}`",
        f"- Prior vs baseline: `{report['regression_counts']['prior_vs_baseline']}`",
        f"- Classifier vs baseline: `{report['regression_counts']['classifier_vs_baseline']}`",
        f"- Classifier vs prior: `{report['regression_counts']['classifier_vs_prior']}`",
        "",
        "| Case | Baseline Edge F1 | Prior Edge F1 | Classifier Edge F1 | Winner |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for c in report["case_results"]:
        lines.append(
            f"| {c['case_id']} | {_fmt(c['baseline_edge_f1'])} | {_fmt(c['prior_edge_f1'])} | "
            f"{_fmt(c['classifier_edge_f1'])} | {c['winner']} |"
        )
    lines += [
        "",
        "## 8. Root-cause attribution",
        "",
        "Counts summed over all generations (compare_graphs; existing taxonomy).",
        "",
        json.dumps(report["attribution"], indent=2),
        "",
        "## 9. Final architecture decision",
        "",
        json.dumps(d, indent=2),
        "",
        "## 10. Production recommendation",
        "",
        f"Production default remains **baseline**. Domain prior is **{d['domain_curriculum_prior']}**. "
        f"Edge classifier is **{d['domain_prior_edge_classifier']}**.",
        "",
        "Reliability (separate suite, not mixed into quality scores):",
        "",
        json.dumps(report.get("reliability") or {}, indent=2),
        "",
    ]
    return "\n".join(lines)


async def run_final_40_case_live(
    *,
    model: str = "gpt-4o-mini",
    generations: int = 3,
    temperature: float = 0.0,
    seed: int = 42,
    output_dir: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    """Live 40-case n-generation run of the three frozen systems, then comparison + reliability."""
    import os

    from app.evaluation.benchmark import run_benchmark
    from app.evaluation.dataset import load_dataset
    from app.evaluation.reporting import write_benchmark_result
    from app.services.llm import reset_llm_provider

    os.environ["OPENAI_MODEL"] = model
    reset_llm_provider()

    examples = load_dataset(DEFAULT_DS)
    if len(examples) != 40:
        raise FinalBenchmarkError(f"Expected 40 quality cases, found {len(examples)}")

    print(
        f"Running final 40-case benchmark: systems={list(REQUIRED_SYSTEMS)}, "
        f"generations={generations}, model={model}, selection=INDEPENDENT…",
        flush=True,
    )
    result = await run_benchmark(
        examples,
        systems=REQUIRED_SYSTEMS,
        repetitions=generations,
        temperature=temperature,
        seed=seed,
        include_ops_latency=False,
        ops_latency_samples=0,
        dataset_name="learning_graph_quality_v1",
        model=model,
        prompt_variant="baseline",
        edge_classifier_prompt_variant="edge_classifier_baseline",
        benchmark_type="quality_stability" if generations > 1 else "quality",
    )
    result["notes"] = list(result.get("notes") or []) + [
        "FINAL 40-case system evaluation. Prompt experiments are closed.",
        "Selection is INDEPENDENT across domain_curriculum_prior and domain_prior_edge_classifier.",
        "Unmapped cases: domain systems return DOMAIN_UNRESOLVED (actual system behavior).",
    ]
    out = Path(output_dir) if output_dir else DEFAULT_BENCH
    quality_path = write_benchmark_result(result, out)
    print(f"Wrote {quality_path}", flush=True)

    reliability = run_reliability_benchmark()
    json_path, md_path = build_final_comparison(
        quality_path,
        output_dir=out,
        reliability=reliability,
    )
    return quality_path, json_path, md_path
