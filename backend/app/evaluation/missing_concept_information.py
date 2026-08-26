"""Missing-concept information availability analysis (offline, evaluation only).

For NEVER_PRESENT gold concepts from a stability run, classify whether the concept
was available in the runtime request input (goal + optional input_notes).

Does not use an LLM judge, does not modify generation, and does not treat gold
as runtime knowledge beyond measuring availability of gold titles in input text.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.evaluation.baselines import build_source_text
from app.evaluation.dataset import load_dataset
from app.evaluation.edge_ambiguity import adapt_example_for_edge_mode
from app.evaluation.inspect import _graph_from_row
from app.evaluation.metrics import (
    _alias_index,
    compare_graphs,
    normalize_topic,
    topic_tokens,
)
from app.evaluation.node_edge_attribution import (
    classify_gold_topic_representation,
    load_node_representation_map,
)
from app.evaluation.persistent_failure_attribution import (
    FREQ_CONSISTENT,
    _edge_matched,
    _group_rows,
    _topic_present,
    find_latest_stability_artifact,
)
from app.evaluation.schemas import EvalExample, GeneratedGraph

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = _REPO_ROOT / "results" / "failure_analysis"

CLASSIFICATIONS = (
    "SOURCE_EXPLICIT",
    "SOURCE_IMPLICIT",
    "EXTERNAL_PREREQUISITE",
    "GOAL_DERIVED",
    "AMBIGUOUS",
    "UNKNOWN",
)

# Deterministic implication rules: phrase must appear in runtime input (normalized).
# Conservative; prefer AMBIGUOUS over forcing SOURCE_IMPLICIT.
# (phrase_normalized_substring, gold_topic_normalized, reason, confidence)
_IMPLICIT_RULES: tuple[tuple[str, str, str, str], ...] = (
    (
        "abstract syntax tree",
        "parsing",
        "AST construction from source code implies parsing as the transformation step.",
        "HIGH",
    ),
    (
        "into an ast",
        "parsing",
        "Transforming source into an AST implies a parsing step.",
        "HIGH",
    ),
    (
        "parse tree",
        "parsing",
        "Mention of a parse tree implies parsing.",
        "HIGH",
    ),
    (
        "consumer group",
        "kafka",
        "Consumer groups are a Kafka-specific construct.",
        "HIGH",
    ),
    (
        "broker partition",
        "kafka",
        "Broker partitions in a streaming context imply Kafka-style messaging.",
        "MEDIUM",
    ),
    (
        "replicated state machine",
        "replication",
        "Replicated state machines presuppose replication.",
        "HIGH",
    ),
    (
        "raft log",
        "replication",
        "Raft log replication implies the replication concept.",
        "MEDIUM",
    ),
)


def _safe_rate(num: float, den: float) -> float:
    return (num / den) if den else 0.0


def available_runtime_input(example: EvalExample) -> dict[str, Any]:
    """Text Synapse actually receives in quality eval (goal + optional input_notes)."""
    goal = (example.goal or "").strip()
    notes = (example.input_notes or "").strip() if example.input_notes else ""
    # Intentionally exclude example.notes (evaluator commentary / gold rationale).
    text = build_source_text(example)
    return {
        "learning_goal": goal,
        "input_notes": notes or None,
        "available_input_text": text,
        "learning_goal_present": bool(goal),
        "source_content_present": bool(notes),
        "evaluator_notes_excluded": True,
        "evaluator_notes_preview": ((example.notes or "").strip()[:120] or None),
    }


def _find_span(haystack: str, needle: str) -> str | None:
    if not haystack or not needle:
        return None
    idx = haystack.casefold().find(needle.casefold())
    if idx < 0:
        return None
    start = max(0, idx - 20)
    end = min(len(haystack), idx + len(needle) + 20)
    return haystack[start:end].strip()


def _phrase_in_normalized(norm_hay: str, phrase: str) -> bool:
    """Word-boundary-ish containment of a normalized phrase in normalized haystack."""
    p = normalize_topic(phrase)
    if not p or not norm_hay:
        return False
    if p == norm_hay:
        return True
    # Pad with spaces so token boundaries are respected for multi-token phrases.
    return f" {p} " in f" {norm_hay} "


def detect_source_explicit(
    gold_topic: str,
    example: EvalExample,
    available_input_text: str,
) -> dict[str, Any] | None:
    """Deterministic explicit mention of gold title or approved alias in runtime input."""
    if not available_input_text.strip():
        return None
    norm_input = normalize_topic(available_input_text)
    surfaces: list[tuple[str, str]] = [(gold_topic, "normalized_exact")]
    for alias in example.topic_aliases.get(gold_topic, []):
        surfaces.append((alias, "alias_exact"))
    # Also search alias index keys that map to this gold
    for norm_alias, canon in _alias_index(example).items():
        if normalize_topic(canon) == normalize_topic(gold_topic) and norm_alias != normalize_topic(
            gold_topic
        ):
            surfaces.append((norm_alias, "alias_exact"))

    seen: set[str] = set()
    for surface, method in surfaces:
        key = normalize_topic(surface)
        if not key or key in seen:
            continue
        seen.add(key)
        if _phrase_in_normalized(norm_input, surface):
            span = _find_span(available_input_text, surface) or _find_span(
                available_input_text, gold_topic
            )
            return {
                "classification": "SOURCE_EXPLICIT",
                "evidence": span or surface,
                "evidence_text": span or surface,
                "matching_method": method,
                "matched_surface": surface,
                "reason": f"Concept surface {surface!r} appears in runtime input via {method}.",
                "confidence": "HIGH",
            }

    # Token-level: for multi-token gold titles, require all tokens (≥3 chars) present.
    toks = [t for t in topic_tokens(gold_topic) if len(t) >= 3]
    if len(toks) >= 2 and all(_phrase_in_normalized(norm_input, t) for t in toks):
        return {
            "classification": "SOURCE_EXPLICIT",
            "evidence": available_input_text[:160],
            "evidence_text": available_input_text[:160],
            "matching_method": "token_all_present",
            "matched_surface": gold_topic,
            "reason": "All significant tokens of the gold title appear in runtime input.",
            "confidence": "MEDIUM",
        }
    return None


def detect_source_implicit(
    gold_topic: str,
    available_input_text: str,
) -> dict[str, Any] | None:
    """Apply curated deterministic implication rules only (no LLM)."""
    if not available_input_text.strip():
        return None
    norm_input = normalize_topic(available_input_text)
    gold_n = normalize_topic(gold_topic)
    for phrase, topic_n, reason, confidence in _IMPLICIT_RULES:
        if topic_n != gold_n:
            continue
        if phrase in norm_input or _phrase_in_normalized(norm_input, phrase):
            span = _find_span(available_input_text, phrase) or phrase
            return {
                "classification": "SOURCE_IMPLICIT",
                "evidence": span,
                "evidence_text": span,
                "source_evidence": span,
                "matching_method": "deterministic_implication_rule",
                "reason": reason,
                "confidence": confidence,
            }
    return None


def classify_missing_concept_availability(
    gold_topic: str,
    example: EvalExample,
    *,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return exactly one primary information-availability classification."""
    runtime = runtime or available_runtime_input(example)
    text = runtime["available_input_text"]
    source_present = bool(runtime["source_content_present"])
    goal = runtime["learning_goal"]

    explicit = detect_source_explicit(gold_topic, example, text)
    if explicit:
        return explicit

    implicit = detect_source_implicit(gold_topic, text)
    if implicit:
        return implicit

    if not text.strip():
        return {
            "classification": "AMBIGUOUS",
            "evidence": None,
            "evidence_text": None,
            "matching_method": None,
            "reason": "No runtime input text available.",
            "confidence": "LOW",
        }

    # Goal-only regime: cannot claim grounded source extraction.
    if not source_present:
        # Weak lexical relation to the goal → still goal-derived curriculum inference.
        goal_toks = {t for t in topic_tokens(goal) if len(t) >= 4}
        gold_toks = {t for t in topic_tokens(gold_topic) if len(t) >= 4}
        overlap = goal_toks & gold_toks
        if overlap:
            return {
                "classification": "GOAL_DERIVED",
                "evidence": goal,
                "evidence_text": goal,
                "matching_method": "goal_token_overlap",
                "reason": (
                    f"No source document; concept shares goal tokens {sorted(overlap)} "
                    "and is treated as curriculum knowledge implied by the learning objective."
                ),
                "confidence": "MEDIUM",
            }
        return {
            "classification": "GOAL_DERIVED",
            "evidence": goal,
            "evidence_text": goal,
            "matching_method": "goal_only_curriculum",
            "reason": (
                "Dataset provides a learning goal but no grounded source content. "
                "Missing concept is a gold prerequisite associated with the objective, "
                "not extractable from supplied source text (INPUT_CONTEXT_LIMITATION)."
            ),
            "confidence": "MEDIUM",
            "input_context_limitation": True,
        }

    # Source notes exist but concept still not found.
    return {
        "classification": "EXTERNAL_PREREQUISITE",
        "evidence": None,
        "evidence_text": runtime.get("input_notes"),
        "matching_method": None,
        "reason": (
            "Source/input notes are present but do not explicitly or deterministically "
            "imply this concept; recovering it would require external domain knowledge."
        ),
        "confidence": "MEDIUM",
        "why_source_lacks_evidence": "No exact/alias/token/implication match in runtime input.",
    }


def collect_never_present_inventory(
    artifact_path: str | Path,
    *,
    dataset_path: str | Path | None = None,
    system: str = "synapse",
) -> dict[str, Any]:
    """Rebuild NEVER_PRESENT gold topics + impacts from a stability artifact."""
    target = Path(artifact_path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    systems = payload.get("systems") or {}
    if system not in systems:
        raise ValueError(f"System {system!r} not in artifact; found {list(systems)}")

    rows_by_case = _group_rows(systems[system])
    ds_stem = payload.get("dataset") or "learning_graph_quality_v1"
    ds_path = Path(dataset_path) if dataset_path else _REPO_ROOT / "data" / "eval" / f"{ds_stem}.jsonl"
    if not ds_path.is_file():
        ds_path = _REPO_ROOT / "data" / "eval" / "learning_graph_quality_v1.jsonl"
    examples = {ex.id: ex for ex in load_dataset(ds_path)}
    rep_map = load_node_representation_map()

    inventory: list[dict[str, Any]] = []
    stable_missing_total = 0

    for eid, rows in rows_by_case.items():
        ex = examples.get(eid)
        if not ex:
            continue
        adapted = adapt_example_for_edge_mode(
            ex, "edge_calibrated", topic_matching_mode="curated_alias"
        )
        graphs = [_graph_from_row(r) for r in rows]
        cmps = [compare_graphs(adapted, g) for g in graphs]
        n = len(graphs)
        if n == 0:
            continue

        required_deps = adapted.required_dependency_list()
        # Stable missing edges for this case
        case_stable_missing: list[tuple[str, str]] = []
        for frm, to in required_deps:
            hits = sum(1 for cmp in cmps if _edge_matched(frm, to, cmp))
            if hits == 0:
                case_stable_missing.append((frm, to))
        stable_missing_total += len(case_stable_missing)

        for gold in adapted.required_topic_list():
            present_hits = sum(1 for cmp in cmps if _topic_present(gold, cmp))
            freq = present_hits / n
            if freq > 0.0:
                continue
            # NEVER_PRESENT
            src_edges = [(a, b) for a, b in case_stable_missing if normalize_topic(a) == normalize_topic(gold)]
            tgt_edges = [(a, b) for a, b in case_stable_missing if normalize_topic(b) == normalize_topic(gold)]
            # Representation candidates across gens
            candidates: list[str] = []
            statuses: list[str] = []
            for g in graphs:
                lab = classify_gold_topic_representation(gold, adapted, g, rep_map=rep_map)
                statuses.append(str(lab.get("status") or "UNKNOWN"))
                for c in lab.get("candidates") or []:
                    candidates.append(str(c))
            status_mode = Counter(statuses).most_common(1)[0][0] if statuses else "UNKNOWN"
            inventory.append(
                {
                    "case_id": eid,
                    "domain": adapted.category,
                    "difficulty": adapted.difficulty,
                    "gold_topic": gold,
                    "stable_missing": True,
                    "stable_missing_generations": n,
                    "topic_presence_frequency": freq,
                    "associated_required_edges": [list(e) for e in src_edges + tgt_edges],
                    "source_edge_count": len(src_edges),
                    "target_edge_count": len(tgt_edges),
                    "number_of_stable_missing_edges": len(src_edges) + len(tgt_edges),
                    "total_dependency_impact": len(src_edges) + len(tgt_edges),
                    "dependency_impact": len(src_edges) + len(tgt_edges),
                    "generated_topic_candidates": sorted({normalize_topic(c) for c in candidates}),
                    "matching_status": status_mode,
                    "representation_subtype": status_mode,
                    "learning_goal": adapted.goal,
                }
            )

    return {
        "artifact": str(target),
        "dataset": str(ds_path),
        "examples": examples,
        "inventory": inventory,
        "stable_missing_edge_total": stable_missing_total,
        "never_present_count": len(inventory),
        "n_cases": len(rows_by_case),
        "payload_meta": {
            "model": (payload.get("config") or {}).get("model") or payload.get("model"),
            "dataset": payload.get("dataset"),
        },
        "rows_by_case": rows_by_case,
        "system": system,
    }


def run_missing_concept_information_analysis(
    artifact_path: str | Path | None = None,
    *,
    dataset_path: str | Path | None = None,
    system: str = "synapse",
    output_dir: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    """Offline analysis. Returns (analysis_md, analysis_json, pareto_md)."""
    target = Path(artifact_path) if artifact_path else find_latest_stability_artifact()
    collected = collect_never_present_inventory(
        target, dataset_path=dataset_path, system=system
    )
    examples: dict[str, EvalExample] = collected["examples"]
    inventory = collected["inventory"]
    stable_total = int(collected["stable_missing_edge_total"])

    classified: list[dict[str, Any]] = []
    case_matrix: dict[str, dict[str, Any]] = {}
    source_present_cases = 0
    goal_only_cases = 0

    # Case matrix init from examples that appear in inventory or artifact
    case_ids = {rec["case_id"] for rec in inventory} | set(
        collected["rows_by_case"].keys()
    )
    for eid in sorted(case_ids):
        ex = examples.get(eid)
        if not ex:
            continue
        runtime = available_runtime_input(ex)
        if runtime["source_content_present"]:
            source_present_cases += 1
        else:
            goal_only_cases += 1
        case_matrix[eid] = {
            "case_id": eid,
            "learning_goal": runtime["learning_goal"],
            "learning_goal_present": runtime["learning_goal_present"],
            "source_content_present": runtime["source_content_present"],
            "number_of_missing_topics": 0,
            "number_of_SOURCE_EXPLICIT_missing_topics": 0,
            "number_of_SOURCE_IMPLICIT_missing_topics": 0,
            "number_of_EXTERNAL_PREREQUISITE_missing_topics": 0,
            "number_of_GOAL_DERIVED_missing_topics": 0,
            "number_of_AMBIGUOUS_missing_topics": 0,
            "number_of_UNKNOWN_missing_topics": 0,
            "stable_missing_edge_count": 0,
        }

    for rec in inventory:
        eid = rec["case_id"]
        ex = examples[eid]
        runtime = available_runtime_input(ex)
        avail = classify_missing_concept_availability(rec["gold_topic"], ex, runtime=runtime)
        row = {
            **rec,
            "available_input_text": runtime["available_input_text"],
            "available_context": {
                "learning_goal": runtime["learning_goal"],
                "input_notes": runtime["input_notes"],
                "source_content_present": runtime["source_content_present"],
            },
            "classification": avail["classification"],
            "evidence": avail.get("evidence"),
            "evidence_text": avail.get("evidence_text"),
            "matching_method": avail.get("matching_method"),
            "reason": avail.get("reason"),
            "confidence": avail.get("confidence"),
            "source_evidence": avail.get("source_evidence"),
            "why_source_lacks_evidence": avail.get("why_source_lacks_evidence"),
            "input_context_limitation": bool(avail.get("input_context_limitation"))
            or (not runtime["source_content_present"]),
            "source_availability": avail["classification"],
            "was_present_in_input": avail["classification"]
            in {"SOURCE_EXPLICIT", "SOURCE_IMPLICIT"},
            "generated_alternatives": {
                "candidates": rec.get("generated_topic_candidates") or [],
                "matching_status": rec.get("matching_status"),
                "representation_subtype": rec.get("representation_subtype"),
            },
        }
        classified.append(row)
        cm = case_matrix[eid]
        cm["number_of_missing_topics"] += 1
        cm[f"number_of_{avail['classification']}_missing_topics"] = (
            cm.get(f"number_of_{avail['classification']}_missing_topics", 0) + 1
        )
        cm["stable_missing_edge_count"] += int(rec["total_dependency_impact"])

    # Impact by class
    by_class: dict[str, dict[str, Any]] = {
        c: {"concepts": 0, "stable_missing_edges": 0} for c in CLASSIFICATIONS
    }
    for row in classified:
        c = row["classification"]
        by_class[c]["concepts"] += 1
        by_class[c]["stable_missing_edges"] += int(row["total_dependency_impact"])

    pareto_class_rows: list[dict[str, Any]] = []
    cum = 0
    impact_den = max(stable_total, 1)
    for c, stats in sorted(
        by_class.items(), key=lambda kv: (-kv[1]["stable_missing_edges"], kv[0])
    ):
        if stats["concepts"] == 0 and stats["stable_missing_edges"] == 0:
            continue
        cum += stats["stable_missing_edges"]
        pareto_class_rows.append(
            {
                "classification": c,
                "concepts": stats["concepts"],
                "stable_missing_edges": stats["stable_missing_edges"],
                "impact_pct": _safe_rate(stats["stable_missing_edges"], impact_den),
                "cumulative_pct": _safe_rate(cum, impact_den),
            }
        )

    concept_pareto = sorted(
        classified,
        key=lambda r: (-r["total_dependency_impact"], r["case_id"], r["gold_topic"]),
    )
    concept_pareto_rows: list[dict[str, Any]] = []
    cum_c = 0
    for i, r in enumerate(concept_pareto, start=1):
        cum_c += r["total_dependency_impact"]
        concept_pareto_rows.append(
            {
                "rank": i,
                "case_id": r["case_id"],
                "gold_topic": r["gold_topic"],
                "classification": r["classification"],
                "dependency_impact": r["total_dependency_impact"],
                "cumulative_pct": _safe_rate(cum_c, impact_den),
            }
        )

    n_concepts = len(classified)
    source_grounded = sum(
        1 for r in classified if r["classification"] in {"SOURCE_EXPLICIT", "SOURCE_IMPLICIT"}
    )
    explicit_n = by_class["SOURCE_EXPLICIT"]["concepts"]
    implicit_n = by_class["SOURCE_IMPLICIT"]["concepts"]
    external_n = by_class["EXTERNAL_PREREQUISITE"]["concepts"]
    goal_n = by_class["GOAL_DERIVED"]["concepts"]
    amb_n = by_class["AMBIGUOUS"]["concepts"]

    frac_source_cases = _safe_rate(source_present_cases, source_present_cases + goal_only_cases)
    input_limitation = frac_source_cases < 0.2

    diagnosis, rationale = _diagnose(
        by_class=by_class,
        n_concepts=n_concepts,
        input_limitation=input_limitation,
        source_present_cases=source_present_cases,
        goal_only_cases=goal_only_cases,
        stable_total=stable_total,
    )

    out_dir = Path(output_dir) if output_dir else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    json_path = out_dir / f"{ts}_missing_concept_information_analysis.json"
    md_path = out_dir / f"{ts}_missing_concept_information_analysis.md"
    pareto_path = out_dir / f"{ts}_missing_concept_information_pareto.md"

    payload_out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "no_new_llm_calls": True,
        "no_llm_judge": True,
        "source_artifact": str(target),
        "dataset": collected["dataset"],
        "system": system,
        "matching_mode": "curated_alias",
        "edge_mode": "edge_calibrated",
        "evaluation_configuration": {
            "runtime_input_fields": ["goal", "input_notes"],
            "excluded_fields": ["notes", "gold_topic_summaries", "gold_topics"],
            "FREQ_CONSISTENT": FREQ_CONSISTENT,
            "input_context_limitation": input_limitation,
            "cases_with_source_content": source_present_cases,
            "cases_goal_only": goal_only_cases,
        },
        "never_present_count": n_concepts,
        "stable_missing_edge_total": stable_total,
        "metrics": {
            "SOURCE_EXPLICIT_concepts": explicit_n,
            "SOURCE_IMPLICIT_concepts": implicit_n,
            "EXTERNAL_PREREQUISITE_concepts": external_n,
            "GOAL_DERIVED_concepts": goal_n,
            "AMBIGUOUS_concepts": amb_n,
            "source_grounded_concepts": source_grounded,
            "source_grounded_rate": _safe_rate(source_grounded, n_concepts),
            "goal_derived_rate": _safe_rate(goal_n, n_concepts),
            "external_rate": _safe_rate(external_n, n_concepts),
        },
        "impact_by_class": by_class,
        "pareto_by_class": pareto_class_rows,
        "pareto_by_concept": concept_pareto_rows,
        "case_matrix": list(case_matrix.values()),
        "diagnosis": {"code": diagnosis, "rationale": rationale},
        "engineering_implication": _implication(diagnosis),
        "missing_concepts": classified,
    }
    json_path.write_text(json.dumps(payload_out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(_render_analysis_md(payload_out), encoding="utf-8")
    pareto_path.write_text(_render_pareto_md(payload_out), encoding="utf-8")
    return md_path, json_path, pareto_path


def _diagnose(
    *,
    by_class: dict[str, dict[str, Any]],
    n_concepts: int,
    input_limitation: bool,
    source_present_cases: int,
    goal_only_cases: int,
    stable_total: int,
) -> tuple[str, str]:
    if n_concepts == 0:
        return ("INSUFFICIENT_EVIDENCE", "No NEVER_PRESENT concepts found.")

    counts = {c: by_class[c]["concepts"] for c in CLASSIFICATIONS}
    impacts = {c: by_class[c]["stable_missing_edges"] for c in CLASSIFICATIONS}
    top = max(CLASSIFICATIONS, key=lambda c: (counts[c], impacts[c]))
    top_rate = _safe_rate(counts[top], n_concepts)

    rationale = (
        f"n_never_present={n_concepts}, top={top} ({top_rate:.2f}), "
        f"source_cases={source_present_cases}, goal_only_cases={goal_only_cases}, "
        f"stable_missing_edges={stable_total}, "
        f"INPUT_CONTEXT_LIMITATION={input_limitation}."
    )

    if input_limitation and counts["SOURCE_EXPLICIT"] + counts["SOURCE_IMPLICIT"] < max(
        1, n_concepts * 0.15
    ):
        # Goal-only dataset: cannot prove extraction vs open-world inference from source docs.
        if counts["GOAL_DERIVED"] / n_concepts >= 0.5:
            return (
                "GOAL_DERIVED_DOMINANT",
                rationale
                + " Dataset is goal-only for most cases; missing concepts are predominantly "
                "curriculum prerequisites associated with the learning objective, not extractable "
                "from grounded source material.",
            )
        return (
            "INSUFFICIENT_EVIDENCE",
            rationale
            + " The dataset does not provide enough grounded source material to distinguish "
            "extraction failure from external-prerequisite inference.",
        )

    mapping = {
        "SOURCE_EXPLICIT": "SOURCE_GROUNDED_DOMINANT",
        "SOURCE_IMPLICIT": "IMPLICIT_SOURCE_DOMINANT",
        "EXTERNAL_PREREQUISITE": "EXTERNAL_KNOWLEDGE_DOMINANT",
        "GOAL_DERIVED": "GOAL_DERIVED_DOMINANT",
        "AMBIGUOUS": "INSUFFICIENT_EVIDENCE",
        "UNKNOWN": "INSUFFICIENT_EVIDENCE",
    }
    if top_rate >= 0.5:
        return mapping[top], rationale

    material = [c for c in CLASSIFICATIONS if _safe_rate(counts[c], n_concepts) >= 0.2]
    if len(material) >= 2:
        return "MIXED", rationale + f" Material classes: {material}."
    return mapping.get(top, "MIXED"), rationale


def _implication(diagnosis: str) -> str:
    return {
        "SOURCE_GROUNDED_DOMINANT": (
            "Extraction / grounding is the likely next direction: concepts were available "
            "in input but not surfaced in the generated inventory."
        ),
        "IMPLICIT_SOURCE_DOMINANT": (
            "Stronger source understanding / structured extraction from context is the "
            "likely next direction."
        ),
        "EXTERNAL_KNOWLEDGE_DOMINANT": (
            "A controlled external knowledge / prerequisite layer is needed; input alone "
            "does not contain the missing concepts."
        ),
        "GOAL_DERIVED_DOMINANT": (
            "Open-world prerequisite inference from the learning goal is the core problem; "
            "do not frame this primarily as document extraction."
        ),
        "MIXED": (
            "Quantify dominant categories separately; no single information regime justifies "
            "one architecture change."
        ),
        "INSUFFICIENT_EVIDENCE": (
            "Enrich the evaluation dataset with grounded source/context material before "
            "choosing extraction vs external-knowledge architecture."
        ),
    }.get(diagnosis, "Re-evaluate after inspecting class breakdown.")


def _pick(rows: list[dict[str, Any]], cls: str, n: int) -> list[dict[str, Any]]:
    return [r for r in rows if r["classification"] == cls][:n]


def _render_analysis_md(payload: dict[str, Any]) -> str:
    m = payload["metrics"]
    cfg = payload["evaluation_configuration"]
    lines = [
        "# Missing Concept Information Availability Analysis",
        "",
        f"- Source artifact: `{payload['source_artifact']}`",
        f"- Dataset: `{payload['dataset']}`",
        f"- NO_NEW_LLM_CALLS / no LLM judge: `{payload['no_new_llm_calls']}`",
        f"- Diagnosis: **{payload['diagnosis']['code']}**",
        f"- Rationale: {payload['diagnosis']['rationale']}",
        f"- Engineering implication: {payload['engineering_implication']}",
        "",
        "## Input context",
        "",
        f"- Runtime fields: `{cfg['runtime_input_fields']}`",
        f"- Excluded (not runtime): `{cfg['excluded_fields']}`",
        f"- Cases with source content (`input_notes`): **{cfg['cases_with_source_content']}**",
        f"- Goal-only cases: **{cfg['cases_goal_only']}**",
        f"- INPUT_CONTEXT_LIMITATION: **{cfg['input_context_limitation']}**",
        "",
    ]
    if cfg["input_context_limitation"]:
        lines.extend(
            [
                "> **INPUT_CONTEXT_LIMITATION:** The dataset does not provide enough grounded "
                "source material to distinguish extraction failure from external-prerequisite "
                "inference for most cases. Runtime input is primarily the learning goal.",
                "",
            ]
        )
    lines.extend(
        [
            "## Inventory",
            "",
            f"- NEVER_PRESENT concepts: **{payload['never_present_count']}**",
            f"- Stable missing edges (denominator): **{payload['stable_missing_edge_total']}**",
            f"- Source-grounded (EXPLICIT+IMPLICIT): **{m['source_grounded_concepts']}** "
            f"({m['source_grounded_rate']:.3f})",
            "",
            "## Information availability breakdown",
            "",
            "| Classification | Concept Count | Stable Missing Edges | Impact % |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    den = max(int(payload["stable_missing_edge_total"]), 1)
    for row in payload["pareto_by_class"]:
        lines.append(
            f"| {row['classification']} | {row['concepts']} | "
            f"{row['stable_missing_edges']} | {row['impact_pct']:.3f} |"
        )
    # Ensure zero classes appear
    present = {r["classification"] for r in payload["pareto_by_class"]}
    for c in CLASSIFICATIONS:
        if c not in present:
            lines.append(f"| {c} | 0 | 0 | 0.000 |")

    lines.extend(["", "## Representative cases", ""])
    for cls, n, title in [
        ("SOURCE_EXPLICIT", 3, "SOURCE_EXPLICIT"),
        ("SOURCE_IMPLICIT", 3, "SOURCE_IMPLICIT"),
        ("EXTERNAL_PREREQUISITE", 3, "EXTERNAL_PREREQUISITE"),
        ("GOAL_DERIVED", 3, "GOAL_DERIVED"),
        ("AMBIGUOUS", 3, "AMBIGUOUS"),
    ]:
        picks = _pick(payload["missing_concepts"], cls, n)
        lines.append(f"### {title}")
        lines.append("")
        if not picks:
            lines.append(f"_No cases in this category (n=0)._")
            lines.append("")
            continue
        for r in picks:
            lines.extend(
                [
                    f"#### {r['case_id']}: `{r['gold_topic']}`",
                    "",
                    f"**Learning goal:** {r['learning_goal']}",
                    f"- Available input: `{r['available_input_text'][:240]}`",
                    f"- Classification: **{r['classification']}**",
                    f"- Evidence: `{r.get('evidence')}`",
                    f"- Method: `{r.get('matching_method')}`",
                    f"- Reason: {r.get('reason')}",
                    f"- Stable missing edges affected: {r['total_dependency_impact']} "
                    f"`{r['associated_required_edges']}`",
                    f"- Generated alternatives: `{r.get('generated_alternatives')}`",
                    "",
                ]
            )
    return "\n".join(lines) + "\n"


def _render_pareto_md(payload: dict[str, Any]) -> str:
    lines = [
        "# Missing Concept Information Pareto",
        "",
        f"- Artifact: `{payload['source_artifact']}`",
        f"- NEVER_PRESENT: {payload['never_present_count']}",
        f"- Stable missing edges: {payload['stable_missing_edge_total']}",
        f"- Diagnosis: **{payload['diagnosis']['code']}**",
        "",
        "## By information class",
        "",
        "| Information Class | Concepts | Stable Missing Edges | Impact % | Cumulative % |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["pareto_by_class"]:
        lines.append(
            f"| {row['classification']} | {row['concepts']} | {row['stable_missing_edges']} | "
            f"{row['impact_pct']:.3f} | {row['cumulative_pct']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Top missing concepts by dependency impact",
            "",
            "| Rank | Case | Topic | Class | Impact | Cum % |",
            "| ---: | --- | --- | --- | ---: | ---: |",
        ]
    )
    for row in (payload.get("pareto_by_concept") or [])[:25]:
        lines.append(
            f"| {row['rank']} | {row['case_id']} | {row['gold_topic']} | "
            f"{row['classification']} | {row['dependency_impact']} | {row['cumulative_pct']:.3f} |"
        )
    return "\n".join(lines) + "\n"
