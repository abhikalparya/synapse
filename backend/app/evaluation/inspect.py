"""Deterministic failure inspection over stored benchmark generations.

No LLM judge. Reads a quality-benchmark JSON plus the gold dataset and writes a
human-inspectable artifact under ``results/failure_analysis/``.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.evaluation.dataset import load_dataset
from app.evaluation.metrics import compare_graphs, normalize_topic, topic_similarity, topic_tokens
from app.evaluation.schemas import EvalExample, GeneratedGraph

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = _REPO_ROOT / "results" / "failure_analysis"


def _graph_from_row(row: dict[str, Any]) -> GeneratedGraph:
    deps = []
    for d in row.get("generated_dependencies") or []:
        if isinstance(d, (list, tuple)) and len(d) == 2:
            deps.append((str(d[0]), str(d[1])))
    return GeneratedGraph(
        topics=[str(t) for t in row.get("generated_topics") or []],
        dependencies=deps,
        skipped_dependencies=list(row.get("skipped_dependencies") or []),
        parse_ok=bool(row.get("parse_ok", True)),
        error=row.get("error"),
    )


def _looks_like_curriculum_framing(title: str, example: EvalExample) -> bool:
    n = normalize_topic(title)
    if n.startswith(("introduction", "intro", "overview", "basic", "fundamental", "application")):
        return True
    return bool(topic_tokens(title) & topic_tokens(example.goal))


def _topic_category(
    title: str,
    closest: str,
    similarity: float,
    *,
    extra: bool,
    example: EvalExample,
) -> tuple[str, bool, bool, bool, str]:
    """Return (category, is_matching_problem, is_gold_ambiguity, is_actual_model_error, explanation)."""
    gen_toks = topic_tokens(title)
    gold_toks = topic_tokens(closest) if closest else set()
    containment = bool(gold_toks and (gold_toks <= gen_toks or gen_toks <= gold_toks))
    if extra:
        if similarity >= 0.5:
            return (
                "ALIAS_MISMATCH",
                True,
                False,
                False,
                f"Generated {title!r} is in-scope by similarity ({similarity:.2f} vs {closest!r}) "
                "but was left unmatched — likely an alias/assignment gap.",
            )
        if containment and len(gen_toks) != len(gold_toks):
            return (
                "GRANULARITY_MISMATCH",
                True,
                False,
                False,
                f"Generated {title!r} contains/is contained in gold {closest!r} (sim={similarity:.2f}) "
                "but did not pass the matcher threshold. Different granularity, not a random extra topic.",
            )
        if 0.2 <= similarity < 0.5:
            return (
                "TITLE_PARAPHRASE",
                True,
                False,
                False,
                f"Generated {title!r} paraphrases gold {closest!r} (sim={similarity:.2f}) "
                "without enough token overlap for a deterministic match.",
            )
        if _looks_like_curriculum_framing(title, example):
            return (
                "EXTRA_TOPIC",
                False,
                True,
                False,
                f"Generated {title!r} looks like a reasonable curriculum heading for this goal "
                f"(closest gold {closest!r}, sim={similarity:.2f}). The gold graph is one curated "
                "reference, not the only valid topic set.",
            )
        return (
            "HALLUCINATED_TOPIC",
            False,
            False,
            True,
            f"Generated {title!r} has no close gold counterpart (closest {closest!r}, sim={similarity:.2f}).",
        )
    if containment and similarity < 0.5:
        return (
            "GRANULARITY_MISMATCH",
            True,
            False,
            False,
            f"Required {title!r} is a coarsening/refinement of generated {closest!r} (sim={similarity:.2f}).",
        )
    if 0.2 <= similarity < 0.5:
        return (
            "ALIAS_MISMATCH",
            True,
            False,
            False,
            f"Required {title!r} is close to generated {closest!r} (sim={similarity:.2f}) but unmatched. "
            "A curated alias would likely credit this.",
        )
    return (
        "MISSING_TOPIC",
        False,
        False,
        True,
        f"Required topic {title!r} is absent. Closest generated title {closest!r} (sim={similarity:.2f}).",
    )


def classify_comparison(example: EvalExample, comparison: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    closest_map = {c["title"]: c for c in comparison.get("closest_unmatched") or []}

    reversed_gold: set[tuple[str, str]] = set()
    for _gen_edge, gold_edge in comparison.get("reversed_dependencies") or []:
        if isinstance(gold_edge, (list, tuple)) and len(gold_edge) == 2:
            reversed_gold.add((str(gold_edge[0]), str(gold_edge[1])))
            reversed_gold.add((normalize_topic(str(gold_edge[0])), normalize_topic(str(gold_edge[1]))))

    for title in comparison.get("extra_topics") or []:
        info = closest_map.get(title) or {"closest": "", "similarity": 0.0}
        cat, match_p, gold_a, model_e, expl = _topic_category(
            title,
            str(info.get("closest") or ""),
            float(info.get("similarity") or 0.0),
            extra=True,
            example=example,
        )
        failures.append(
            {
                "case_id": example.id,
                "category": cat,
                "gold": info.get("closest") or "",
                "generated": title,
                "explanation": expl,
                "is_matching_problem": match_p,
                "is_gold_ambiguity": gold_a,
                "is_actual_model_error": model_e,
            },
        )

    for title in comparison.get("missing_topics") or []:
        info = closest_map.get(title) or {"closest": "", "similarity": 0.0}
        cat, match_p, gold_a, model_e, expl = _topic_category(
            title,
            str(info.get("closest") or ""),
            float(info.get("similarity") or 0.0),
            extra=False,
            example=example,
        )
        failures.append(
            {
                "case_id": example.id,
                "category": cat,
                "gold": title,
                "generated": info.get("closest") or "",
                "explanation": expl,
                "is_matching_problem": match_p,
                "is_gold_ambiguity": gold_a,
                "is_actual_model_error": model_e,
            },
        )

    missing_topics = set(comparison.get("missing_topics") or [])
    for gold_edge in comparison.get("missing_dependencies") or []:
        key = (str(gold_edge[0]), str(gold_edge[1]))
        if key in reversed_gold or (normalize_topic(key[0]), normalize_topic(key[1])) in reversed_gold:
            continue
        endpoints_unmatched = str(gold_edge[0]) in missing_topics or str(gold_edge[1]) in missing_topics
        failures.append(
            {
                "case_id": example.id,
                "category": "MISSING_PREREQUISITE",
                "gold": f"{gold_edge[0]} requires {gold_edge[1]}",
                "generated": "",
                "explanation": (
                    f"Required edge {gold_edge[0]!r} -> {gold_edge[1]!r} (from requires to) is absent. "
                    + (
                        "One or both endpoints were unmatched, so this may be a matching artifact."
                        if endpoints_unmatched
                        else "Both topics were matched; the model omitted this prerequisite."
                    )
                ),
                "is_matching_problem": endpoints_unmatched,
                "is_gold_ambiguity": False,
                "is_actual_model_error": not endpoints_unmatched,
            },
        )

    for gen_edge in comparison.get("extra_dependencies") or []:
        redundant = any(
            list(gen_edge) == list(r) or tuple(gen_edge) == tuple(r)
            for r in comparison.get("redundant_transitive_among_extra") or []
        ) or any(
            list(gen_edge) == list(r) or tuple(gen_edge) == tuple(r)
            for r in comparison.get("redundant_transitive_edges") or []
        )
        if redundant:
            failures.append(
                {
                    "case_id": example.id,
                    "category": "REDUNDANT_TRANSITIVE_EDGE",
                    "gold": "",
                    "generated": f"{gen_edge[0]} requires {gen_edge[1]}",
                    "explanation": (
                        f"Generated edge {gen_edge[0]!r} -> {gen_edge[1]!r} is implied by a longer "
                        "prerequisite path in the same generated graph (transitive shortcut)."
                    ),
                    "is_matching_problem": False,
                    "is_gold_ambiguity": False,
                    "is_actual_model_error": True,
                },
            )
        else:
            failures.append(
                {
                    "case_id": example.id,
                    "category": "EXTRA_DEPENDENCY",
                    "gold": "",
                    "generated": f"{gen_edge[0]} requires {gen_edge[1]}",
                    "explanation": (
                        f"Generated edge {gen_edge[0]!r} -> {gen_edge[1]!r} is neither a required nor an "
                        "acceptable alternative. It may still be pedagogically valid — the gold graph is one "
                        "curated reference, not the only correct structure."
                    ),
                    "is_matching_problem": False,
                    "is_gold_ambiguity": True,
                    "is_actual_model_error": False,
                },
            )

    for gen_edge, gold_edge in comparison.get("reversed_dependencies") or []:
        gold_s = f"{gold_edge[0]} requires {gold_edge[1]}" if isinstance(gold_edge, (list, tuple)) else str(gold_edge)
        gen_s = f"{gen_edge[0]} requires {gen_edge[1]}"
        failures.append(
            {
                "case_id": example.id,
                "category": "WRONG_DEPENDENCY_DIRECTION",
                "gold": gold_s,
                "generated": gen_s,
                "explanation": "Generated the reverse of a required prerequisite edge.",
                "is_matching_problem": False,
                "is_gold_ambiguity": False,
                "is_actual_model_error": True,
            },
        )

    return failures


def analyze_benchmark(
    result_path: str | Path,
    *,
    dataset_path: str | Path | None = None,
    system: str = "synapse",
    output_dir: str | Path | None = None,
    max_failures: int = 80,
) -> Path:
    target = Path(result_path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    examples = {e.id: e for e in load_dataset(dataset_path)}
    rows = ((payload.get("systems") or {}).get(system) or {}).get("example_results") or []

    cases: list[dict[str, Any]] = []
    all_failures: list[dict[str, Any]] = []
    for row in rows:
        eid = str(row.get("example_id") or "")
        example = examples.get(eid)
        if example is None:
            continue
        graph = _graph_from_row(row)
        comparison = compare_graphs(example, graph)
        classified = classify_comparison(example, comparison)
        all_failures.extend(classified)
        cases.append(
            {
                "case_id": eid,
                "goal": example.goal,
                "gold_topics": example.required_topic_list(),
                "optional_topics": example.optional_topic_list(),
                "gold_edges": [list(d) for d in example.required_dependency_list()],
                "scores": row.get("scores"),
                "comparison": comparison,
                "failures": classified,
            },
        )

    by_cat: dict[str, list[dict[str, Any]]] = {}
    for f in all_failures:
        by_cat.setdefault(str(f["category"]), []).append(f)
    representative: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    def _add(item: dict[str, Any]) -> bool:
        key = (item["case_id"], item["category"], str(item.get("gold")) + str(item.get("generated")))
        if key in seen:
            return False
        seen.add(key)
        representative.append(item)
        return True

    for _cat, items in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
        for item in items:
            if _add(item):
                break
    # Diversify case_ids so the sample is not dominated by the first example
    seen_cases = {f["case_id"] for f in representative}
    for item in all_failures:
        if len(representative) >= max(20, min(max_failures, 24)):
            break
        if item["case_id"] in seen_cases:
            continue
        if _add(item):
            seen_cases.add(item["case_id"])
    for item in all_failures:
        if len(representative) >= max(20, min(max_failures, 24)):
            break
        _add(item)

    counts = dict(Counter(f["category"] for f in all_failures))
    matching_n = sum(1 for f in all_failures if f.get("is_matching_problem"))
    model_n = sum(1 for f in all_failures if f.get("is_actual_model_error"))
    gold_n = sum(1 for f in all_failures if f.get("is_gold_ambiguity"))

    artifact = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_benchmark": str(target),
        "system": system,
        "dataset": str(dataset_path) if dataset_path else "learning_graph_eval_v1",
        "note": (
            "The benchmark measures agreement with curated reference structures and does not "
            "claim that there is only one universally correct learning graph. Classifications "
            "are deterministic heuristics, not an LLM judge."
        ),
        "summary": {
            "cases": len(cases),
            "failure_records": len(all_failures),
            "by_category": dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))),
            "matching_problem_share": (matching_n / len(all_failures)) if all_failures else 0.0,
            "actual_model_error_share": (model_n / len(all_failures)) if all_failures else 0.0,
            "gold_ambiguity_share": (gold_n / len(all_failures)) if all_failures else 0.0,
        },
        "representative_failures": representative[:max_failures],
        "cases": cases,
    }

    out = Path(output_dir) if output_dir else DEFAULT_OUT
    out.mkdir(parents=True, exist_ok=True)
    model = str(payload.get("model") or "unknown")
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = out / f"{stamp}_{model}_analysis.json"
    path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def rescore_benchmark(
    result_path: str | Path,
    *,
    dataset_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    """Recompute quality metrics on stored generations (no new LLM calls)."""
    from app.evaluation.metrics import aggregate_scores, score_graph
    from app.evaluation.reporting import write_benchmark_result

    target = Path(result_path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    examples = {e.id: e for e in load_dataset(dataset_path)}
    systems_out: dict[str, Any] = {}
    for sys_name, block in (payload.get("systems") or {}).items():
        scores = []
        new_rows = []
        for row in block.get("example_results") or []:
            example = examples.get(str(row.get("example_id")))
            if example is None:
                new_rows.append(row)
                continue
            graph = _graph_from_row(row)
            sc = score_graph(example, graph) if graph.parse_ok else None
            if sc is not None:
                scores.append(sc)
                row = {
                    **row,
                    "scores": {
                        "topic_precision": sc.topic_precision,
                        "topic_recall": sc.topic_recall,
                        "topic_f1": sc.topic_f1,
                        "dependency_precision": sc.dependency_precision,
                        "dependency_recall": sc.dependency_recall,
                        "dependency_f1": sc.dependency_f1,
                        "graph_valid": sc.graph_valid,
                        "cycle_attempt": sc.cycle_attempt,
                        "missing_prerequisite_rate": sc.missing_prerequisite_rate,
                        "hallucinated_topic_rate": sc.hallucinated_topic_rate,
                        "extra_dependency_rate": sc.extra_dependency_rate,
                        "dependency_direction_error_rate": sc.dependency_direction_error_rate,
                    },
                    "failures": sc.failures,
                }
            new_rows.append(row)
        systems_out[sys_name] = {
            **block,
            "metrics": aggregate_scores(scores),
            "example_results": new_rows,
        }

    dataset_version = Path(dataset_path).stem if dataset_path else payload.get("dataset")
    rescored = {
        **payload,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmark_type": "quality",
        "dataset_version": dataset_version,
        "rescored_from": str(target),
        "systems": systems_out,
        "metrics": {name: b.get("metrics") for name, b in systems_out.items()},
        "notes": list(payload.get("notes") or [])
        + ["Rescored stored generations against the current dataset/schema; no new LLM calls."],
    }
    out = Path(output_dir) if output_dir else target.parent
    rescored["model"] = f"rescored-{payload.get('model') or 'unknown'}"
    return write_benchmark_result(rescored, out)
