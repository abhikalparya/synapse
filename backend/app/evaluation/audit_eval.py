"""Audit detection + isolated before/after repair evaluation.

Repair is eval-only: gold/reference edges incident to topics the production audit
flagged are added on an in-memory copy. The live graph is never modified.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, patch

from app.evaluation.audit_cases import audit_v1
from app.evaluation.metrics import score_graph
from app.evaluation.schemas import EvalExample, GeneratedGraph
from app.models.ai_ops import AuditFinding, AuditReport
from app.services.audit import audit_graph

_STRUCTURAL = {"orphaned_topic", "duplicate_title", "thin_topic"}
_SEMANTIC = {"missing_prerequisite", "cycle_risk"}

_TYPE_ALIASES = {
    "MISSING_PREREQUISITE": "missing_prerequisite",
    "DUPLICATE_CONCEPT": "duplicate_title",
    "ISOLATED_ADVANCED_TOPIC": "orphaned_topic",
    "SUSPICIOUS_DEPENDENCY_DIRECTION": "cycle_risk",
    "INCORRECT_DEPENDENCY": "cycle_risk",
    "THIN_TOPIC": "thin_topic",
}


def _canon_type(raw: str) -> str:
    t = (raw or "").strip()
    return _TYPE_ALIASES.get(t, t)


def _topic_dicts(graph: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for row in graph.get("topics") or []:
        out.append(
            {
                "id": str(row.get("id") or row.get("title")),
                "title": str(row.get("title") or row.get("id")),
                "summary": str(row.get("summary") or ""),
                "status": "not_started",
                "resources": [],
                "quiz_passed": False,
                "zone_id": None,
            },
        )
    return out


def _dep_dicts(graph: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"from_topic_id": str(d["from_topic_id"]), "to_topic_id": str(d["to_topic_id"])}
        for d in graph.get("dependencies") or []
    ]


def _finding_titles(finding: AuditFinding, id_to_title: dict[str, str]) -> set[str]:
    titles = {id_to_title[i].casefold() for i in finding.topic_ids if i in id_to_title}
    return titles


def _match_finding(
    known: dict[str, Any],
    finding: AuditFinding,
    id_to_title: dict[str, str],
) -> bool:
    if _canon_type(str(known.get("type") or "")) != finding.type:
        return False
    topic = str(known.get("topic") or "").casefold()
    if not topic:
        return True
    titles = _finding_titles(finding, id_to_title)
    return topic in titles or any(topic in t or t in topic for t in titles)


def score_audit_case(case: dict[str, Any], report: AuditReport) -> dict[str, Any]:
    topics = _topic_dicts(case["graph"])
    id_to_title = {t["id"]: t["title"] for t in topics}
    known = list(case.get("known_issues") or [])
    predicted = list(report.findings)
    matched_known: set[int] = set()
    matched_pred: set[int] = set()
    for i, k in enumerate(known):
        for j, f in enumerate(predicted):
            if j in matched_pred:
                continue
            if _match_finding(k, f, id_to_title):
                matched_known.add(i)
                matched_pred.add(j)
                break

    tp = len(matched_known)
    fp = len(predicted) - len(matched_pred)
    fn = len(known) - len(matched_known)
    precision = (tp / (tp + fp)) if (tp + fp) else 1.0
    recall = (tp / (tp + fn)) if (tp + fn) else 1.0
    fpr = (fp / len(predicted)) if predicted else 0.0

    def split(types: set[str]) -> dict[str, float]:
        k = [x for x in known if _canon_type(str(x.get("type") or "")) in types]
        p = [f for f in predicted if f.type in types]
        mk: set[int] = set()
        mp: set[int] = set()
        for i, item in enumerate(k):
            for j, f in enumerate(p):
                if j in mp:
                    continue
                if _match_finding(item, f, id_to_title):
                    mk.add(i)
                    mp.add(j)
                    break
        tp_s = len(mk)
        fp_s = len(p) - len(mp)
        fn_s = len(k) - len(mk)
        return {
            "precision": (tp_s / (tp_s + fp_s)) if (tp_s + fp_s) else 1.0,
            "recall": (tp_s / (tp_s + fn_s)) if (tp_s + fn_s) else 1.0,
            "false_positive_rate": (fp_s / len(p)) if p else 0.0,
            "true_positives": float(tp_s),
            "false_positives": float(fp_s),
            "false_negatives": float(fn_s),
            "known": float(len(k)),
            "predicted": float(len(p)),
        }

    by_cat: dict[str, dict[str, int]] = {}
    for item in known:
        cat = _canon_type(str(item.get("type") or "unknown"))
        by_cat.setdefault(cat, {"known": 0, "detected": 0})
        by_cat[cat]["known"] += 1
    for i, item in enumerate(known):
        if i in matched_known:
            cat = _canon_type(str(item.get("type") or "unknown"))
            by_cat.setdefault(cat, {"known": 0, "detected": 0})
            by_cat[cat]["detected"] += 1

    return {
        "id": case["id"],
        "mode": case.get("mode"),
        "precision": precision,
        "recall": recall,
        "false_positive_rate": fpr,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "status": report.status,
        "semantic_analysis": report.semantic_analysis,
        "structural": split(_STRUCTURAL),
        "semantic": split(_SEMANTIC),
        "by_category": by_cat,
        "predicted_types": [f.type for f in predicted],
    }


def eval_repair_from_findings(
    topics: list[dict[str, Any]],
    dependencies: list[dict[str, str]],
    findings: list[AuditFinding],
    repair_edges: list[list[str]],
) -> list[tuple[str, str]]:
    """Eval-only repair: add reference edges incident to flagged topic titles."""
    id_to_title = {t["id"]: t["title"] for t in topics}
    title_to_id = {t["title"].casefold(): t["id"] for t in topics}
    flagged = {id_to_title[i].casefold() for f in findings for i in f.topic_ids if i in id_to_title}
    existing = {
        (id_to_title.get(d["from_topic_id"], "").casefold(), id_to_title.get(d["to_topic_id"], "").casefold())
        for d in dependencies
    }
    out: list[tuple[str, str]] = [
        (id_to_title[d["from_topic_id"]], id_to_title[d["to_topic_id"]])
        for d in dependencies
        if d["from_topic_id"] in id_to_title and d["to_topic_id"] in id_to_title
    ]
    for frm, to in repair_edges:
        if frm.casefold() not in title_to_id or to.casefold() not in title_to_id:
            continue
        if frm.casefold() not in flagged and to.casefold() not in flagged:
            continue
        key = (frm.casefold(), to.casefold())
        if key in existing:
            continue
        out.append((frm, to))
        existing.add(key)
    return out


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(r.get(key) or 0.0) for r in rows) / len(rows)


def _nested_mean(rows: list[dict[str, Any]], bucket: str, key: str) -> float:
    if not rows:
        return 0.0
    return sum(float((r.get(bucket) or {}).get(key) or 0.0) for r in rows) / len(rows)


async def run_audit_benchmark(*, no_llm: bool = True) -> dict[str, Any]:
    cases = audit_v1()
    scored: list[dict[str, Any]] = []
    repairs: list[dict[str, Any]] = []

    llm_patcher = None
    if no_llm:
        logging.getLogger("app.services.audit").setLevel(logging.ERROR)
        llm_patcher = patch(
            "app.services.audit.call_llm",
            new=AsyncMock(side_effect=RuntimeError("audit eval --no-llm")),
        )
        llm_patcher.start()
    try:
        for case in cases:
            topics = _topic_dicts(case["graph"])
            deps = _dep_dicts(case["graph"])
            report = await audit_graph(topics, deps)
            row = score_audit_case(case, report)
            scored.append(row)

            ref_topics = case.get("reference_topics")
            ref_deps = case.get("reference_dependencies")
            repair_edges = case.get("repair_edges") or []
            if ref_topics and ref_deps:
                example = EvalExample(
                    id=case["id"],
                    category="audit_repair",
                    difficulty="beginner",
                    goal="audit repair",
                    gold_topics=list(ref_topics),
                    gold_dependencies=[(str(a), str(b)) for a, b in (tuple(x) for x in ref_deps)],
                )
                title_by_id = {t["id"]: t["title"] for t in topics}
                before_graph = GeneratedGraph(
                    topics=[t["title"] for t in topics],
                    dependencies=[(title_by_id[d["from_topic_id"]], title_by_id[d["to_topic_id"]]) for d in deps],
                )
                before = score_graph(example, before_graph)
                repaired_deps = eval_repair_from_findings(
                    topics,
                    deps,
                    list(report.findings),
                    [list(e) for e in repair_edges],
                )
                after_graph = GeneratedGraph(topics=[t["title"] for t in topics], dependencies=repaired_deps)
                after = score_graph(example, after_graph)
                repairs.append(
                    {
                        "id": case["id"],
                        "dependency_recall_before": before.dependency_recall,
                        "dependency_recall_after": after.dependency_recall,
                        "dependency_f1_before": before.dependency_f1,
                        "dependency_f1_after": after.dependency_f1,
                        "missing_prerequisite_rate_before": before.missing_prerequisite_rate,
                        "missing_prerequisite_rate_after": after.missing_prerequisite_rate,
                        "dependency_direction_error_rate_before": before.dependency_direction_error_rate,
                        "dependency_direction_error_rate_after": after.dependency_direction_error_rate,
                        "improved": after.dependency_f1 > before.dependency_f1 + 1e-9,
                        "regressed": after.dependency_f1 < before.dependency_f1 - 1e-9,
                    },
                )
    finally:
        if llm_patcher is not None:
            llm_patcher.stop()

    improved = sum(1 for r in repairs if r["improved"])
    regressed = sum(1 for r in repairs if r["regressed"])
    structural_rows = [r for r in scored if (r.get("structural") or {}).get("known", 0) or (r.get("structural") or {}).get("predicted", 0)]
    semantic_rows = [r for r in scored if (r.get("semantic") or {}).get("known", 0) or (r.get("semantic") or {}).get("predicted", 0)]
    metrics = {
        "precision": _mean(scored, "precision"),
        "recall": _mean(scored, "recall"),
        "false_positive_rate": _mean(scored, "false_positive_rate"),
        "structural_precision": _nested_mean(structural_rows, "structural", "precision"),
        "structural_recall": _nested_mean(structural_rows, "structural", "recall"),
        "structural_false_positive_rate": _nested_mean(structural_rows, "structural", "false_positive_rate"),
        "semantic_precision": _nested_mean(semantic_rows, "semantic", "precision") if semantic_rows else None,
        "semantic_recall": _nested_mean(semantic_rows, "semantic", "recall") if semantic_rows else None,
        "semantic_false_positive_rate": _nested_mean(semantic_rows, "semantic", "false_positive_rate") if semantic_rows else None,
        "n": float(len(scored)),
        "n_structural_scored": float(len(structural_rows)),
        "n_semantic_scored": float(len(semantic_rows)),
        "semantic_mode": "unavailable" if no_llm else "available",
        "repair_cases": float(len(repairs)),
        "repair_improved": float(improved),
        "repair_regressed": float(regressed),
        "dependency_recall_delta": (
            _mean(repairs, "dependency_recall_after") - _mean(repairs, "dependency_recall_before")
            if repairs
            else 0.0
        ),
        "dependency_f1_delta": (
            _mean(repairs, "dependency_f1_after") - _mean(repairs, "dependency_f1_before") if repairs else 0.0
        ),
        "missing_prerequisite_rate_delta": (
            _mean(repairs, "missing_prerequisite_rate_after") - _mean(repairs, "missing_prerequisite_rate_before")
            if repairs
            else 0.0
        ),
    }
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmark_type": "audit",
        "dataset": "graph_audit_eval_v1",
        "dataset_version": "graph_audit_eval_v1",
        "model": "none" if no_llm else "configured",
        "provider": "none" if no_llm else "configured",
        "seed": 42,
        "repetitions": 1,
        "example_count": len(scored),
        "metrics": metrics,
        "failures": {r["id"]: r for r in scored if r["false_negatives"] or r["false_positives"]},
        "latency": {},
        "cost": {"note": "Structural audit is free. Semantic pass uses the configured LLM unless --no-llm."},
        "cases": scored,
        "repair": {
            "note": (
                "Eval-only: add reference edges incident to topics flagged by production audit. "
                "Live graph is not modified. Improvement is reported only when it occurs."
            ),
            "cases": repairs,
        },
        "notes": [
            "Structural findings use production _structural_findings via audit_graph.",
            "--no-llm forces semantic_analysis=unavailable (status=partial), matching production degraded mode.",
            "The benchmark measures agreement with curated reference structures and does not claim a unique correct graph.",
        ],
    }
