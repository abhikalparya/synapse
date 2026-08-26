"""Baseline generation stability and error-consistency analysis (evaluation only).

Repeats the same cases across multiple stored generations and measures whether
failures persist or vary. Does not change prompts, matching, gold, or generation.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.evaluation.dataset import load_dataset
from app.evaluation.edge_ambiguity import adapt_example_for_edge_mode
from app.evaluation.inspect import _graph_from_row
from app.evaluation.metrics import compare_graphs, normalize_topic, score_graph
from app.evaluation.schemas import EvalExample

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = _REPO_ROOT / "results" / "failure_analysis"

# --- Explicit classification thresholds (documented; not tuned to force outcomes) ---
FREQ_CONSISTENT = 0.8
FREQ_INTERMITTENT = 0.4

# Per-case stability
GOOD_TOPIC_F1 = 0.65
GOOD_EDGE_F1 = 0.35
BAD_TOPIC_F1 = 0.45
BAD_EDGE_F1 = 0.25
STABLE_RANGE_TOPIC = 0.15
STABLE_RANGE_EDGE = 0.20
HIGH_VAR_RANGE_TOPIC = 0.25
HIGH_VAR_RANGE_EDGE = 0.30
HIGH_VAR_STD_TOPIC = 0.12
HIGH_VAR_STD_EDGE = 0.15

# Diagnosis thresholds
SYSTEMATIC_PERSISTENCE_MIN = 0.55  # share of failures in ≥2 gens
SYSTEMATIC_JACCARD_MIN = 0.45
STOCHASTIC_JACCARD_MAX = 0.25
STOCHASTIC_HIGH_VAR_CASE_MIN = 0.40


def _safe_std(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    return float(statistics.pstdev(vals))


def _safe_mean(vals: list[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else 0.0


def _safe_median(vals: list[float]) -> float:
    return float(statistics.median(vals)) if vals else 0.0


def _distribution(vals: list[float]) -> dict[str, float | int | None]:
    if not vals:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "std_dev": None,
            "min": None,
            "max": None,
            "cv": None,
        }
    mean = _safe_mean(vals)
    std = _safe_std(vals)
    return {
        "n": len(vals),
        "mean": mean,
        "median": _safe_median(vals),
        "std_dev": std,
        "min": float(min(vals)),
        "max": float(max(vals)),
        "cv": (std / mean) if mean else None,
    }


def _jaccard(a: set[Any], b: set[Any]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def _mean_pairwise_jaccard(sets: list[set[Any]]) -> float:
    if len(sets) < 2:
        return 1.0 if sets else 0.0
    scores: list[float] = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            scores.append(_jaccard(sets[i], sets[j]))
    return _safe_mean(scores)


def _freq_bucket(freq: float) -> str:
    if freq <= 0.0:
        return "NEVER"
    if freq < FREQ_INTERMITTENT:
        return "RARELY"
    if freq < FREQ_CONSISTENT:
        return "INTERMITTENTLY"
    return "CONSISTENTLY"


def _norm_edge(frm: str, to: str) -> tuple[str, str]:
    return (normalize_topic(frm), normalize_topic(to))


def _group_rows(system_block: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in system_block.get("example_results") or []:
        eid = str(row.get("example_id") or "")
        if not eid:
            continue
        by_case[eid].append(row)
    for eid in by_case:
        by_case[eid].sort(key=lambda r: int(r.get("repetition") or r.get("generation_index") or 0))
    return dict(by_case)


def _score_row(example: EvalExample, row: dict[str, Any]) -> dict[str, Any]:
    graph = _graph_from_row(row)
    adapted = adapt_example_for_edge_mode(
        example, "edge_calibrated", topic_matching_mode="curated_alias"
    )
    scores = score_graph(adapted, graph) if graph.parse_ok else None
    cmp = compare_graphs(adapted, graph) if graph.parse_ok else None

    missing_topics = list(cmp["missing_topics"]) if cmp else []
    # Unmatched generated topics (same definition as score_graph hallucinated_topic_rate).
    extra_topics = list(cmp["extra_topics"]) if cmp else []
    hallucinated = list(extra_topics)
    missing_edges = [tuple(e) for e in (cmp["missing_dependencies"] if cmp else [])]
    invalid_edges = [tuple(e) for e in (cmp["extra_dependencies"] if cmp else [])]
    matched_edges = [tuple(e) for e in (cmp["matched_dependencies"] if cmp else [])]
    reversed_edges = cmp["reversed_dependencies"] if cmp else []
    reversed_gold_edges: list[tuple[str, str]] = []
    for item in reversed_edges:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            gold = item[1]
            if isinstance(gold, (list, tuple)) and len(gold) == 2:
                reversed_gold_edges.append((str(gold[0]), str(gold[1])))

    # matched_required_topics from compare_graphs are already normalized titles.
    present_gold: list[str] = []
    matched_required_norms = set(cmp["matched_required_topics"]) if cmp else set()
    for g in adapted.required_topic_list():
        if normalize_topic(g) in matched_required_norms:
            present_gold.append(g)

    seed = None
    seed_supported = None
    meta = row.get("generation_meta") or {}
    if isinstance(meta, dict):
        seed = meta.get("seed", row.get("seed"))
        seed_supported = meta.get("seed_supported", row.get("seed_supported"))
    else:
        seed = row.get("seed")
        seed_supported = row.get("seed_supported")

    return {
        "example_id": example.id,
        "generation_index": int(row.get("generation_index", row.get("repetition") or 0)),
        "seed": seed,
        "seed_supported": seed_supported,
        "parse_ok": bool(row.get("parse_ok", True)),
        "topics": list(graph.topics),
        "dependencies": [list(d) for d in graph.dependencies],
        "n_topics": len(graph.topics),
        "n_dependencies": len(graph.dependencies),
        "scores": None
        if scores is None
        else {
            "topic_precision": scores.topic_precision,
            "topic_recall": scores.topic_recall,
            "topic_f1": scores.topic_f1,
            "required_edge_precision": scores.required_edge_precision,
            "required_edge_recall": scores.required_edge_recall,
            "required_edge_f1": scores.required_edge_f1,
            "missing_required_edge_rate": scores.missing_required_edge_rate,
            "invalid_extra_edge_rate": scores.invalid_extra_edge_rate,
            "hallucinated_topic_rate": scores.hallucinated_topic_rate,
            "dependency_direction_error_rate": scores.dependency_direction_error_rate,
            "extra_dependency_rate": scores.extra_dependency_rate,
        },
        "missing_topics": missing_topics,
        "present_gold_topics": present_gold,
        "hallucinated_topics": hallucinated,
        "extra_topics": extra_topics,
        "missing_edges": [list(e) for e in missing_edges],
        "matched_edges": [list(e) for e in matched_edges],
        "invalid_edges": [list(e) for e in invalid_edges],
        "reversed_gold_edges": [list(e) for e in reversed_gold_edges],
        "reversed_edge_count": len(reversed_edges),
        "total_latency_ms": float(row.get("total_latency_ms") or 0.0),
        "cost_usd": row.get("cost_usd"),
        "input_tokens": row.get("input_tokens"),
        "output_tokens": row.get("output_tokens"),
        "topic_failure_set": sorted(
            {normalize_topic(t) for t in missing_topics}
            | {normalize_topic(t) for t in hallucinated}
        ),
        "edge_failure_set": sorted(
            {f"{a}→{b}" for a, b in (_norm_edge(*e) for e in missing_edges)}
            | {f"{a}→{b}" for a, b in (_norm_edge(*e) for e in invalid_edges)}
        ),
    }


def classify_case_stability(gens: list[dict[str, Any]]) -> str:
    """Deterministic per-case stability label."""
    scored = [g for g in gens if g.get("scores")]
    if not scored:
        return "MIXED"
    topic_f1 = [float(g["scores"]["topic_f1"]) for g in scored]
    edge_f1 = [float(g["scores"]["required_edge_f1"]) for g in scored]
    t_mean, e_mean = _safe_mean(topic_f1), _safe_mean(edge_f1)
    t_range = max(topic_f1) - min(topic_f1)
    e_range = max(edge_f1) - min(edge_f1)
    t_std, e_std = _safe_std(topic_f1), _safe_std(edge_f1)

    if (
        t_range > HIGH_VAR_RANGE_TOPIC
        or e_range > HIGH_VAR_RANGE_EDGE
        or t_std > HIGH_VAR_STD_TOPIC
        or e_std > HIGH_VAR_STD_EDGE
    ):
        return "HIGH_VARIANCE"
    if (
        t_mean >= GOOD_TOPIC_F1
        and e_mean >= GOOD_EDGE_F1
        and t_range <= STABLE_RANGE_TOPIC
        and e_range <= STABLE_RANGE_EDGE
    ):
        return "CONSISTENTLY_GOOD"
    if (
        t_mean < BAD_TOPIC_F1
        and e_mean < BAD_EDGE_F1
        and t_range <= STABLE_RANGE_TOPIC
        and e_range <= STABLE_RANGE_EDGE
    ):
        return "CONSISTENTLY_BAD"
    return "MIXED"


def run_baseline_stability_analysis(
    artifact_path: str | Path,
    *,
    dataset_path: str | Path | None = None,
    system: str = "synapse",
    output_dir: str | Path | None = None,
    max_representatives_per_class: int = 3,
) -> tuple[Path, Path]:
    """Analyze a multi-generation quality artifact; write JSON + Markdown."""
    target = Path(artifact_path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    systems = payload.get("systems") or {}
    if system not in systems:
        raise ValueError(f"System {system!r} not in artifact; found {list(systems)}")

    ds_stem = payload.get("dataset") or payload.get("dataset_version") or "learning_graph_quality_v1"
    ds_path = Path(dataset_path) if dataset_path else _REPO_ROOT / "data" / "eval" / f"{ds_stem}.jsonl"
    if not ds_path.is_file():
        ds_path = _REPO_ROOT / "data" / "eval" / "learning_graph_quality_v1.jsonl"
    examples = {ex.id: ex for ex in load_dataset(ds_path)}

    by_case = _group_rows(systems[system])
    n_gens = max((len(v) for v in by_case.values()), default=0)
    if n_gens < 2:
        # Still produce analysis, but diagnosis will be INSUFFICIENT_EVIDENCE
        pass

    case_results: list[dict[str, Any]] = []
    all_metric_series: dict[str, list[float]] = defaultdict(list)
    latency_vals: list[float] = []
    cost_vals: list[float] = []

    # Global frequency accumulators
    gold_topic_hits: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [present, total]
    gold_edge_hits: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    hall_topic_case_counts: dict[tuple[str, str], int] = Counter()  # (case, norm_title) -> gens
    invalid_edge_case_counts: dict[tuple[str, str], int] = Counter()  # (case, edge_key)

    persistence_buckets = {
        "missing_topics": Counter(),
        "hallucinated_topics": Counter(),
        "missing_edges": Counter(),
        "invalid_edges": Counter(),
        "direction_errors": Counter(),
    }
    # For each unique failure item across cases: count how many gens it appears in, then bucket

    topic_jaccards: list[float] = []
    edge_jaccards: list[float] = []

    for eid, rows in by_case.items():
        ex = examples.get(eid)
        if not ex:
            continue
        adapted = adapt_example_for_edge_mode(
            ex, "edge_calibrated", topic_matching_mode="curated_alias"
        )
        gens = [_score_row(adapted, row) for row in rows]
        n = len(gens)
        if n == 0:
            continue

        classification = classify_case_stability(gens)
        topic_sets = [set(g["topic_failure_set"]) for g in gens]
        edge_sets = [set(g["edge_failure_set"]) for g in gens]
        t_jac = _mean_pairwise_jaccard(topic_sets)
        e_jac = _mean_pairwise_jaccard(edge_sets)
        topic_jaccards.append(t_jac)
        edge_jaccards.append(e_jac)

        topic_f1s = [float(g["scores"]["topic_f1"]) for g in gens if g.get("scores")]
        edge_f1s = [float(g["scores"]["required_edge_f1"]) for g in gens if g.get("scores")]
        miss_rates = [float(g["scores"]["missing_required_edge_rate"]) for g in gens if g.get("scores")]
        inv_rates = [float(g["scores"]["invalid_extra_edge_rate"]) for g in gens if g.get("scores")]
        hall_rates = [float(g["scores"]["hallucinated_topic_rate"]) for g in gens if g.get("scores")]
        dir_rates = [float(g["scores"]["dependency_direction_error_rate"]) for g in gens if g.get("scores")]

        for key, series in [
            ("topic_f1", topic_f1s),
            ("required_edge_f1", edge_f1s),
            ("missing_required_edge_rate", miss_rates),
            ("invalid_extra_edge_rate", inv_rates),
            ("hallucinated_topic_rate", hall_rates),
            ("dependency_direction_error_rate", dir_rates),
        ]:
            all_metric_series[key].extend(series)

        for g in gens:
            latency_vals.append(float(g["total_latency_ms"]))
            if g.get("cost_usd") is not None:
                cost_vals.append(float(g["cost_usd"]))
            for k in (
                "topic_precision",
                "topic_recall",
                "required_edge_precision",
                "required_edge_recall",
            ):
                if g.get("scores") and k in g["scores"]:
                    all_metric_series[k].append(float(g["scores"][k]))

        # Topic frequencies for gold required topics
        topic_freq: dict[str, float] = {}
        for gtopic in adapted.required_topic_list():
            hits = sum(
                1
                for gen in gens
                if normalize_topic(gtopic)
                in {normalize_topic(t) for t in gen["present_gold_topics"]}
            )
            freq = hits / n
            topic_freq[gtopic] = freq
            gold_topic_hits[f"{eid}::{gtopic}"][0] += hits
            gold_topic_hits[f"{eid}::{gtopic}"][1] += n

        # Edge frequencies
        edge_freq: dict[str, float] = {}
        for frm, to in adapted.required_dependency_list():
            key = f"{frm}→{to}"
            hits = 0
            for gen in gens:
                missing_norms = {
                    _norm_edge(a, b) for a, b in (tuple(e) for e in gen["missing_edges"])
                }
                reversed_norms = {
                    _norm_edge(a, b)
                    for a, b in (tuple(e) for e in gen.get("reversed_gold_edges") or [])
                }
                en = _norm_edge(frm, to)
                if en not in missing_norms and en not in reversed_norms:
                    hits += 1
            freq = hits / n
            edge_freq[key] = freq
            gold_edge_hits[f"{eid}::{key}"][0] += hits
            gold_edge_hits[f"{eid}::{key}"][1] += n

        # Hallucination / invalid edge persistence within case
        hall_counts: Counter[str] = Counter()
        inv_counts: Counter[str] = Counter()
        miss_topic_counts: Counter[str] = Counter()
        miss_edge_counts: Counter[str] = Counter()
        dir_present = 0
        for gen in gens:
            for t in gen["hallucinated_topics"]:
                nt = normalize_topic(t)
                hall_counts[nt] += 1
                hall_topic_case_counts[(eid, nt)] += 1
            for e in gen["invalid_edges"]:
                ek = f"{_norm_edge(e[0], e[1])[0]}→{_norm_edge(e[0], e[1])[1]}"
                inv_counts[ek] += 1
                invalid_edge_case_counts[(eid, ek)] += 1
            for t in gen["missing_topics"]:
                miss_topic_counts[normalize_topic(t)] += 1
            for e in gen["missing_edges"]:
                ek = f"{_norm_edge(e[0], e[1])[0]}→{_norm_edge(e[0], e[1])[1]}"
                miss_edge_counts[ek] += 1
            if gen["reversed_edge_count"] > 0:
                dir_present += 1

        def _persist_bucket(count: int) -> str:
            if count <= 1:
                return "one_off"
            if count >= n:
                return "all_generations"
            return "repeated"

        for _, c in miss_topic_counts.items():
            persistence_buckets["missing_topics"][_persist_bucket(c)] += 1
        for _, c in hall_counts.items():
            persistence_buckets["hallucinated_topics"][_persist_bucket(c)] += 1
        for _, c in miss_edge_counts.items():
            persistence_buckets["missing_edges"][_persist_bucket(c)] += 1
        for _, c in inv_counts.items():
            persistence_buckets["invalid_edges"][_persist_bucket(c)] += 1
        if dir_present:
            persistence_buckets["direction_errors"][_persist_bucket(dir_present)] += 1

        case_results.append(
            {
                "case_id": eid,
                "goal": ex.goal,
                "gold_topics": list(adapted.required_topic_list()),
                "gold_dependencies": [list(d) for d in adapted.required_dependency_list()],
                "n_generations": n,
                "classification": classification,
                "topic_f1": _distribution(topic_f1s),
                "required_edge_f1": _distribution(edge_f1s),
                "missing_required_edge_rate": _distribution(miss_rates),
                "invalid_extra_edge_rate": _distribution(inv_rates),
                "n_topics": _distribution([float(g["n_topics"]) for g in gens]),
                "n_dependencies": _distribution([float(g["n_dependencies"]) for g in gens]),
                "topic_failure_jaccard": t_jac,
                "edge_failure_jaccard": e_jac,
                "gold_topic_frequencies": topic_freq,
                "required_edge_frequencies": edge_freq,
                "stable_missing_edges": [k for k, f in edge_freq.items() if f == 0.0],
                "repeated_hallucinations": [t for t, c in hall_counts.items() if c >= 2],
                "repeated_invalid_edges": [e for e, c in inv_counts.items() if c >= 2],
                "generations": gens,
            }
        )

    # Aggregate frequency matrices
    def _bucket_counts(items: dict[str, list[int]], never_label: str, rare: str, intermittent: str, consistent: str) -> dict[str, Any]:
        buckets = Counter()
        details = []
        for key, (hits, total) in items.items():
            if total <= 0:
                continue
            freq = hits / total
            b = _freq_bucket(freq)
            label = {
                "NEVER": never_label,
                "RARELY": rare,
                "INTERMITTENTLY": intermittent,
                "CONSISTENTLY": consistent,
            }[b]
            buckets[label] += 1
            details.append({"key": key, "frequency": freq, "bucket": label, "hits": hits, "total": total})
        return {"counts": dict(buckets), "total_items": sum(buckets.values()), "items": details}

    topic_matrix = _bucket_counts(
        gold_topic_hits,
        "NEVER_PRESENT",
        "RARELY_PRESENT",
        "INTERMITTENTLY_PRESENT",
        "CONSISTENTLY_PRESENT",
    )
    edge_matrix = _bucket_counts(
        gold_edge_hits,
        "NEVER_GENERATED",
        "RARELY_GENERATED",
        "INTERMITTENTLY_GENERATED",
        "CONSISTENTLY_GENERATED",
    )

    # Hallucination / invalid edge global classification
    repeated_hall = sum(1 for (_, _), c in hall_topic_case_counts.items() if c >= 2)
    one_off_hall = sum(1 for (_, _), c in hall_topic_case_counts.items() if c == 1)
    all_gen_hall = sum(1 for (_, _), c in hall_topic_case_counts.items() if n_gens and c >= n_gens)

    repeated_inv = sum(1 for (_, _), c in invalid_edge_case_counts.items() if c >= 2)
    one_off_inv = sum(1 for (_, _), c in invalid_edge_case_counts.items() if c == 1)
    all_gen_inv = sum(1 for (_, _), c in invalid_edge_case_counts.items() if n_gens and c >= n_gens)

    case_class_counts = Counter(c["classification"] for c in case_results)

    # Granularity mismatch rate: approximate from node attribution if available; else skip dense calc
    # Use hallucinated + missing as proxy already covered; optional light pass omitted for speed.

    aggregate_distributions = {k: _distribution(v) for k, v in all_metric_series.items()}
    aggregate_distributions["latency_ms"] = _distribution(latency_vals)
    aggregate_distributions["cost_usd"] = _distribution(cost_vals)

    # Correlations (weak; report honestly)
    correlations: dict[str, Any] = {}
    pairs = [
        ("topic_f1", "latency_ms"),
        ("required_edge_f1", "output_tokens"),
        ("hallucinated_topic_rate", "output_tokens"),
    ]
    # Build aligned series from case generations
    tf1, lats, ef1, out_toks, halls, out_toks2 = [], [], [], [], [], []
    for c in case_results:
        for g in c["generations"]:
            if not g.get("scores"):
                continue
            tf1.append(float(g["scores"]["topic_f1"]))
            lats.append(float(g["total_latency_ms"]))
            ef1.append(float(g["scores"]["required_edge_f1"]))
            out_toks.append(float(g.get("output_tokens") or 0))
            halls.append(float(g["scores"]["hallucinated_topic_rate"]))
            out_toks2.append(float(g.get("output_tokens") or 0))

    def _pearson(xs: list[float], ys: list[float]) -> float | None:
        if len(xs) < 5 or len(xs) != len(ys):
            return None
        mx, my = _safe_mean(xs), _safe_mean(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
        deny = math.sqrt(sum((y - my) ** 2 for y in ys))
        if denx == 0 or deny == 0:
            return None
        return num / (denx * deny)

    correlations["topic_f1_vs_latency"] = {
        "r": _pearson(tf1, lats),
        "n": len(tf1),
        "note": "Interpret cautiously; N is exploratory.",
    }
    correlations["required_edge_f1_vs_output_tokens"] = {
        "r": _pearson(ef1, out_toks),
        "n": len(ef1),
    }
    correlations["hallucination_rate_vs_output_tokens"] = {
        "r": _pearson(halls, out_toks2),
        "n": len(halls),
    }

    # Seed handling summary
    seeds = []
    seed_supported_flags = []
    for c in case_results:
        for g in c["generations"]:
            if g.get("seed") is not None:
                seeds.append(g["seed"])
            if g.get("seed_supported") is not None:
                seed_supported_flags.append(bool(g["seed_supported"]))
    seed_supported = any(seed_supported_flags) if seed_supported_flags else bool(payload.get("seed") is not None)

    # Diagnosis
    diagnosis, rationale = _choose_diagnosis(
        n_gens=n_gens,
        case_class_counts=case_class_counts,
        n_cases=len(case_results),
        mean_topic_jaccard=_safe_mean(topic_jaccards),
        mean_edge_jaccard=_safe_mean(edge_jaccards),
        persistence=persistence_buckets,
        topic_matrix=topic_matrix,
        edge_matrix=edge_matrix,
    )

    representatives = _pick_representatives(case_results, max_representatives_per_class)

    thresholds = {
        "FREQ_CONSISTENT": FREQ_CONSISTENT,
        "FREQ_INTERMITTENT": FREQ_INTERMITTENT,
        "GOOD_TOPIC_F1": GOOD_TOPIC_F1,
        "GOOD_EDGE_F1": GOOD_EDGE_F1,
        "BAD_TOPIC_F1": BAD_TOPIC_F1,
        "BAD_EDGE_F1": BAD_EDGE_F1,
        "STABLE_RANGE_TOPIC": STABLE_RANGE_TOPIC,
        "STABLE_RANGE_EDGE": STABLE_RANGE_EDGE,
        "HIGH_VAR_RANGE_TOPIC": HIGH_VAR_RANGE_TOPIC,
        "HIGH_VAR_RANGE_EDGE": HIGH_VAR_RANGE_EDGE,
        "HIGH_VAR_STD_TOPIC": HIGH_VAR_STD_TOPIC,
        "HIGH_VAR_STD_EDGE": HIGH_VAR_STD_EDGE,
    }

    out_dir = Path(output_dir) if output_dir else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    json_path = out_dir / f"{ts}_baseline_stability_analysis.json"
    md_path = out_dir / f"{ts}_baseline_stability_analysis.md"

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_artifact": str(target),
        "dataset": str(ds_path),
        "system": system,
        "matching_mode": "curated_alias",
        "edge_mode": "edge_calibrated",
        "benchmark_config": {
            "model": payload.get("model"),
            "provider": payload.get("provider"),
            "prompt_variant": payload.get("prompt_variant"),
            "temperature": payload.get("temperature"),
            "base_seed": payload.get("seed"),
            "repetitions_in_artifact": payload.get("repetitions"),
            "generations_per_case_observed": n_gens,
            "n_cases": len(case_results),
            "total_generation_rows": sum(c["n_generations"] for c in case_results),
            "seed_supported": seed_supported,
            "seeds_observed": sorted(set(seeds)) if seeds else [],
        },
        "thresholds": thresholds,
        "diagnosis": {"code": diagnosis, "rationale": rationale},
        "aggregate_distributions": aggregate_distributions,
        "per_case_stability_counts": dict(case_class_counts),
        "per_case_stability_rates": {
            k: (case_class_counts[k] / len(case_results)) if case_results else 0.0
            for k in ("CONSISTENTLY_GOOD", "CONSISTENTLY_BAD", "HIGH_VARIANCE", "MIXED")
        },
        "topic_stability": topic_matrix["counts"],
        "topic_stability_total": topic_matrix["total_items"],
        "edge_stability": edge_matrix["counts"],
        "edge_stability_total": edge_matrix["total_items"],
        "stable_missing_edge_count": edge_matrix["counts"].get("NEVER_GENERATED", 0),
        "hallucination_persistence": {
            "one_off": one_off_hall,
            "repeated": repeated_hall,
            "all_generations": all_gen_hall,
            "total_unique": one_off_hall + repeated_hall,
        },
        "invalid_edge_persistence": {
            "one_off": one_off_inv,
            "repeated": repeated_inv,
            "all_generations": all_gen_inv,
            "total_unique": one_off_inv + repeated_inv,
        },
        "failure_persistence": {
            k: {
                "one_off": v.get("one_off", 0),
                "repeated": v.get("repeated", 0),
                "all_generations": v.get("all_generations", 0),
                "total": sum(v.values()),
            }
            for k, v in persistence_buckets.items()
        },
        "error_signature_similarity": {
            "mean_topic_failure_jaccard": _safe_mean(topic_jaccards),
            "mean_edge_failure_jaccard": _safe_mean(edge_jaccards),
            "min_topic_jaccard": float(min(topic_jaccards)) if topic_jaccards else None,
            "max_topic_jaccard": float(max(topic_jaccards)) if topic_jaccards else None,
            "min_edge_jaccard": float(min(edge_jaccards)) if edge_jaccards else None,
            "max_edge_jaccard": float(max(edge_jaccards)) if edge_jaccards else None,
        },
        "correlations": correlations,
        "representative_case_ids": {
            cls: [c["case_id"] for c in reps] for cls, reps in representatives.items()
        },
        "cases": case_results,
        "notes": [
            "Stability analysis recomputes scores with curated_alias + edge_calibrated.",
            "No quality intervention; observes natural multi-generation variance of the baseline path.",
            "Aliases count toward topic presence via existing match_topic / compare_graphs.",
        ],
    }
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(_render_md(result, representatives), encoding="utf-8")
    return md_path, json_path


def _choose_diagnosis(
    *,
    n_gens: int,
    case_class_counts: Counter[str],
    n_cases: int,
    mean_topic_jaccard: float,
    mean_edge_jaccard: float,
    persistence: dict[str, Counter[str]],
    topic_matrix: dict[str, Any],
    edge_matrix: dict[str, Any],
) -> tuple[str, str]:
    if n_gens < 2 or n_cases < 5:
        return (
            "INSUFFICIENT_EVIDENCE",
            f"Need ≥2 generations and enough cases; observed n_gens={n_gens}, n_cases={n_cases}.",
        )

    # Persistence share for missing edges + hallucinated topics + invalid edges
    def _persist_rate(counter: Counter[str]) -> float:
        total = sum(counter.values())
        if not total:
            return 0.0
        return (counter.get("repeated", 0) + counter.get("all_generations", 0)) / total

    persist_rates = {
        k: _persist_rate(v) for k, v in persistence.items() if sum(v.values()) > 0
    }
    mean_persist = _safe_mean(list(persist_rates.values())) if persist_rates else 0.0
    mean_jac = _safe_mean([mean_topic_jaccard, mean_edge_jaccard])
    high_var_rate = (case_class_counts.get("HIGH_VARIANCE", 0) / n_cases) if n_cases else 0.0
    consistent_bad = (case_class_counts.get("CONSISTENTLY_BAD", 0) / n_cases) if n_cases else 0.0
    never_edge_share = 0.0
    if edge_matrix["total_items"]:
        never_edge_share = edge_matrix["counts"].get("NEVER_GENERATED", 0) / edge_matrix["total_items"]

    systematic_signals = 0
    stochastic_signals = 0
    if mean_persist >= SYSTEMATIC_PERSISTENCE_MIN:
        systematic_signals += 2
    if mean_jac >= SYSTEMATIC_JACCARD_MIN:
        systematic_signals += 2
    if never_edge_share >= 0.35:
        systematic_signals += 1
    if consistent_bad >= 0.25:
        systematic_signals += 1

    if mean_jac <= STOCHASTIC_JACCARD_MAX:
        stochastic_signals += 2
    if high_var_rate >= STOCHASTIC_HIGH_VAR_CASE_MIN:
        stochastic_signals += 2
    if mean_persist < 0.35:
        stochastic_signals += 1

    rationale = (
        f"persist={mean_persist:.3f}, jaccard={mean_jac:.3f} "
        f"(topic={mean_topic_jaccard:.3f}, edge={mean_edge_jaccard:.3f}), "
        f"high_var_cases={high_var_rate:.3f}, never_edges={never_edge_share:.3f}, "
        f"signals sys={systematic_signals} stoch={stochastic_signals}."
    )

    if systematic_signals >= 3 and stochastic_signals >= 3:
        return ("MIXED_STABILITY", rationale)
    if systematic_signals >= 3 and systematic_signals > stochastic_signals:
        return ("SYSTEMATIC_FAILURE_DOMINANT", rationale)
    if stochastic_signals >= 3 and stochastic_signals > systematic_signals:
        return ("STOCHASTIC_FAILURE_DOMINANT", rationale)
    if systematic_signals > 0 and stochastic_signals > 0:
        return ("MIXED_STABILITY", rationale)
    if systematic_signals == 0 and stochastic_signals == 0:
        return ("INSUFFICIENT_EVIDENCE", rationale + " Weak signals.")
    if systematic_signals > stochastic_signals:
        return ("SYSTEMATIC_FAILURE_DOMINANT", rationale)
    return ("STOCHASTIC_FAILURE_DOMINANT", rationale)


def _pick_representatives(
    cases: list[dict[str, Any]],
    k: int,
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {
        "CONSISTENTLY_GOOD": [],
        "CONSISTENTLY_BAD": [],
        "HIGH_VARIANCE": [],
        "MIXED": [],
    }
    for cls in out:
        pool = [c for c in cases if c["classification"] == cls]
        # Prefer extreme means for good/bad; prefer high range for variance
        if cls == "CONSISTENTLY_GOOD":
            pool.sort(key=lambda c: float(c["topic_f1"]["mean"] or 0), reverse=True)
        elif cls == "CONSISTENTLY_BAD":
            pool.sort(key=lambda c: float(c["topic_f1"]["mean"] or 0))
        elif cls == "HIGH_VARIANCE":
            pool.sort(
                key=lambda c: float((c["topic_f1"]["max"] or 0) - (c["topic_f1"]["min"] or 0)),
                reverse=True,
            )
        out[cls] = pool[:k]
    return out


def _render_md(payload: dict[str, Any], representatives: dict[str, list[dict[str, Any]]]) -> str:
    dist = payload["aggregate_distributions"]
    lines = [
        "# Baseline Generation Stability Analysis",
        "",
        f"- Source: `{payload['source_artifact']}`",
        f"- System: `{payload['system']}`",
        f"- Matching: `curated_alias` + `edge_calibrated`",
        f"- Generations/case observed: **{payload['benchmark_config']['generations_per_case_observed']}**",
        f"- Cases: {payload['benchmark_config']['n_cases']}",
        f"- Seed supported: `{payload['benchmark_config']['seed_supported']}`",
        f"- Diagnosis: **{payload['diagnosis']['code']}**",
        f"- Rationale: {payload['diagnosis']['rationale']}",
        "",
        "## Aggregate stability",
        "",
        "| Metric | Mean | Median | Std Dev | Min | Max |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in (
        "topic_f1",
        "required_edge_f1",
        "missing_required_edge_rate",
        "invalid_extra_edge_rate",
        "hallucinated_topic_rate",
        "latency_ms",
        "cost_usd",
    ):
        d = dist.get(key) or {}
        def fmt(x):
            return "—" if x is None else f"{float(x):.3f}"
        lines.append(
            f"| {key} | {fmt(d.get('mean'))} | {fmt(d.get('median'))} | "
            f"{fmt(d.get('std_dev'))} | {fmt(d.get('min'))} | {fmt(d.get('max'))} |"
        )

    rates = payload["per_case_stability_rates"]
    counts = payload["per_case_stability_counts"]
    lines.extend(
        [
            "",
            "## Per-case stability",
            "",
            "| Class | Count | Rate |",
            "| --- | ---: | ---: |",
        ]
    )
    for cls in ("CONSISTENTLY_GOOD", "CONSISTENTLY_BAD", "HIGH_VARIANCE", "MIXED"):
        lines.append(f"| {cls} | {counts.get(cls, 0)} | {rates.get(cls, 0):.3f} |")

    lines.extend(
        [
            "",
            "## Topic stability (gold required topics)",
            "",
            f"`{payload['topic_stability']}` (total={payload['topic_stability_total']})",
            "",
            f"Repeated hallucinations (unique case×title with ≥2 gens): "
            f"{payload['hallucination_persistence']['repeated']}",
            "",
            "## Edge stability (required gold edges)",
            "",
            f"`{payload['edge_stability']}` (total={payload['edge_stability_total']})",
            "",
            f"STABLE_MISSING_EDGES (NEVER_GENERATED): {payload['stable_missing_edge_count']}",
            f"Repeated invalid edges: {payload['invalid_edge_persistence']['repeated']}",
            "",
            "## Failure persistence",
            "",
            "| Failure Type | One-Off | Repeated (≥2) | All Generations |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for k, v in payload["failure_persistence"].items():
        lines.append(
            f"| {k} | {v['one_off']} | {v['repeated']} | {v['all_generations']} |"
        )

    sim = payload["error_signature_similarity"]
    lines.extend(
        [
            "",
            "## Error signature similarity",
            "",
            f"- Mean topic-failure Jaccard: **{sim['mean_topic_failure_jaccard']:.3f}** "
            f"(min={sim['min_topic_jaccard']}, max={sim['max_topic_jaccard']})",
            f"- Mean edge-failure Jaccard: **{sim['mean_edge_failure_jaccard']:.3f}** "
            f"(min={sim['min_edge_jaccard']}, max={sim['max_edge_jaccard']})",
            "",
            "## Representative cases",
            "",
        ]
    )
    for cls, cases in representatives.items():
        if not cases:
            lines.append(f"### {cls}\n\n_No cases in this category._\n")
            continue
        lines.append(f"### {cls}\n")
        for c in cases:
            lines.extend(
                [
                    f"#### {c['case_id']}",
                    "",
                    f"**Objective:** {c['goal']}",
                    "",
                    f"- Gold topics: {c['gold_topics']}",
                    f"- Gold deps: {c['gold_dependencies']}",
                    f"- Topic F1: {_distribution_brief(c['topic_f1'])}",
                    f"- Required Edge F1: {_distribution_brief(c['required_edge_f1'])}",
                    f"- Failure Jaccard topic/edge: {c['topic_failure_jaccard']:.3f} / {c['edge_failure_jaccard']:.3f}",
                    f"- Stable missing edges: {c['stable_missing_edges']}",
                    f"- Repeated hallucinations: {c['repeated_hallucinations']}",
                    "",
                ]
            )
            for g in c["generations"]:
                sc = g.get("scores") or {}
                lines.append(
                    f"  - gen {g['generation_index']} seed={g.get('seed')}: "
                    f"topics={g['topics']} deps={g['dependencies']} "
                    f"T-F1={sc.get('topic_f1')} E-F1={sc.get('required_edge_f1')} "
                    f"missing_topics={g['missing_topics']} hall={g['hallucinated_topics']} "
                    f"missing_edges={g['missing_edges']} invalid_edges={g['invalid_edges']}"
                )
            lines.append("")
    return "\n".join(lines) + "\n"


def _distribution_brief(d: dict[str, Any]) -> str:
    if not d or d.get("mean") is None:
        return "n/a"
    return (
        f"mean={d['mean']:.3f} med={d['median']:.3f} std={d['std_dev']:.3f} "
        f"range=[{d['min']:.3f},{d['max']:.3f}]"
    )
