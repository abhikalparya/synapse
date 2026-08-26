"""Reliability benchmark: production validators vs Direct on adversarial inputs.

Does not call an LLM. Uses an isolated SQLite file (never the live graph).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from app.evaluation.baselines import run_direct_from_raw
from app.evaluation.reliability_cases import ReliabilityCase, reliability_v1
from app.evaluation.synapse_system import run_synapse_from_raw
from app.models.proposal import ProposedDependency, ProposedMerge, ProposedTopic, Proposal
from app.models.topic import TopicCreate
from app.services.proposals import apply_proposal, save_proposal
from app.services.reshape import filter_reshape_new_dependencies
from app.services.snapshots import restore_snapshot
from app.services.topics import load_all_topics, load_dependencies, save_topic


def graph_fingerprint() -> dict[str, Any]:
    topics = load_all_topics()
    deps = load_dependencies()
    by_id = {t["id"]: t["title"] for t in topics}
    return {
        "topics": sorted(by_id.values()),
        "dependencies": sorted(
            (by_id[d["from_topic_id"]], by_id[d["to_topic_id"]])
            for d in deps
            if d["from_topic_id"] in by_id and d["to_topic_id"] in by_id
        ),
    }


def _eval_ingest_pair(raw: str) -> dict[str, Any]:
    direct = run_direct_from_raw(raw)
    synapse = run_synapse_from_raw(raw)
    return {
        "direct_parse_ok": direct.parse_ok,
        "synapse_parse_ok": synapse.parse_ok,
        "direct_error": direct.error,
        "synapse_error": synapse.error,
        "direct_topics": direct.topics,
        "synapse_topics": synapse.topics,
        "direct_dependencies": [list(d) for d in direct.dependencies],
        "synapse_dependencies": [list(d) for d in synapse.dependencies],
        "synapse_skipped": synapse.skipped_dependencies,
        "direct_edge_count": len(direct.dependencies),
        "synapse_edge_count": len(synapse.dependencies),
    }


def _run_cycle(case: ReliabilityCase) -> dict[str, Any]:
    pair = _eval_ingest_pair(case.raw or "")
    synapse_caught = any("cycle" in (s.get("reason") or "").casefold() for s in pair["synapse_skipped"])
    direct_kept = pair["direct_edge_count"] > pair["synapse_edge_count"]
    ok = synapse_caught and direct_kept and pair["synapse_parse_ok"]
    return {"ok": ok, "synapse_caught": synapse_caught, "direct_retained_invalid": direct_kept, **pair}


def _run_self_loop(case: ReliabilityCase) -> dict[str, Any]:
    pair = _eval_ingest_pair(case.raw or "")
    direct_has = any(a == b for a, b in (tuple(x) for x in pair["direct_dependencies"]))
    synapse_has = any(a == b for a, b in (tuple(x) for x in pair["synapse_dependencies"]))
    synapse_caught = (not synapse_has) and (
        any("cycle" in (s.get("reason") or "").casefold() for s in pair["synapse_skipped"])
        or pair["synapse_edge_count"] < pair["direct_edge_count"]
    )
    ok = direct_has and synapse_caught
    return {"ok": ok, "synapse_caught": synapse_caught, "direct_retained_invalid": direct_has, **pair}


def _run_unknown(case: ReliabilityCase) -> dict[str, Any]:
    pair = _eval_ingest_pair(case.raw or "")
    unknown_skips = [s for s in pair["synapse_skipped"] if "unknown" in (s.get("reason") or "").casefold()]
    synapse_caught = bool(unknown_skips)
    direct_kept = pair["direct_edge_count"] > pair["synapse_edge_count"]
    ok = synapse_caught and direct_kept
    return {
        "ok": ok,
        "synapse_caught": synapse_caught,
        "direct_retained_invalid": direct_kept,
        "invalid_references_generated": max(0, pair["direct_edge_count"] - pair["synapse_edge_count"]),
        "invalid_references_rejected": len(unknown_skips),
        **pair,
    }


def _run_reshape_oos(case: ReliabilityCase) -> dict[str, Any]:
    spec = case.reshape or {}
    selected = list(spec.get("selected_titles") or [])
    title_to_id = {t.casefold(): f"id-{i}" for i, t in enumerate(selected)}
    for row in spec.get("new_topics") or []:
        title_to_id[str(row.get("title", "")).casefold()] = f"new-{row.get('title')}"
    seed = []
    for a, b in spec.get("existing_internal_edges") or []:
        fa, fb = title_to_id.get(str(a).casefold()), title_to_id.get(str(b).casefold())
        if fa and fb:
            seed.append({"from_topic_id": fa, "to_topic_id": fb})
    proposed, skipped = filter_reshape_new_dependencies(
        list(spec.get("new_dependencies") or []),
        title_to_id=title_to_id,
        accepted_dep_dicts=seed,
    )
    oos = [s for s in skipped if "out-of-scope" in s.reason.casefold() or "unknown" in s.reason.casefold()]
    ok = bool(oos)
    return {
        "ok": ok,
        "synapse_caught": ok,
        "accepted": len(proposed),
        "skipped": [{"from_title": s.from_title, "to_title": s.to_title, "reason": s.reason} for s in skipped],
        "invalid_references_generated": len(oos),
        "invalid_references_rejected": len(oos),
    }


def _run_malformed(case: ReliabilityCase) -> dict[str, Any]:
    synapse = run_synapse_from_raw(case.raw or "")
    direct = run_direct_from_raw(case.raw or "")
    if case.kind == "malformed_types":
        synapse_caught = (
            all(not (isinstance(a, str) and a.isdigit()) for a, _b in synapse.dependencies)
            if synapse.parse_ok
            else True
        )
        ok = synapse_caught and synapse.parse_ok
        return {
            "ok": ok,
            "synapse_caught": synapse_caught,
            "direct_parse_ok": direct.parse_ok,
            "synapse_parse_ok": synapse.parse_ok,
            "direct_dependencies": [list(d) for d in direct.dependencies],
            "synapse_dependencies": [list(d) for d in synapse.dependencies],
        }
    synapse_caught = not synapse.parse_ok
    ok = synapse_caught and not direct.parse_ok
    return {
        "ok": ok,
        "synapse_caught": synapse_caught,
        "direct_parse_ok": direct.parse_ok,
        "synapse_parse_ok": synapse.parse_ok,
        "synapse_error": synapse.error,
        "direct_error": direct.error,
    }


def _ensure_topic(title: str) -> dict[str, Any]:
    for t in load_all_topics():
        if t["title"] == title:
            return t
    return save_topic(TopicCreate(title=title, summary=f"A reasonably long summary for {title} used in eval."))


def _run_transaction(case: ReliabilityCase) -> dict[str, Any]:
    seed = list((case.extra or {}).get("seed_topics") or ["Keep"])
    new_title = str((case.extra or {}).get("new_topic_title") or "ShouldNotPersist")
    keep = _ensure_topic(seed[0])
    before = graph_fingerprint()
    proposal = Proposal(
        id=uuid.uuid4().hex,
        status="pending",
        mode="ingest",
        source="reliability:transaction_failure",
        topics=[
            ProposedTopic(
                temp_id="new1",
                title=new_title,
                summary="A reasonably long summary that must not persist after abort.",
                confidence=0.9,
            ),
        ],
        merges=[ProposedMerge(source_topic_id="does-not-exist", target_topic_id=keep["id"], reason="eval abort")],
    )
    save_proposal(proposal)
    raised = False
    err = None
    try:
        apply_proposal(proposal.id)
    except (ValueError, LookupError) as exc:
        raised = True
        err = str(exc)
    after = graph_fingerprint()
    leaked = new_title in after["topics"]
    ok = raised and not leaked and after["topics"] == before["topics"] and after["dependencies"] == before["dependencies"]
    return {
        "ok": ok,
        "raised": raised,
        "error": err,
        "leaked_new_topic": leaked,
        "before": before,
        "after": after,
    }


def _run_rollback(case: ReliabilityCase) -> dict[str, Any]:
    extra = case.extra or {}
    seed = list(extra.get("seed_topics") or ["Root"])
    child = str(extra.get("new_topic_title") or "Child")
    root = _ensure_topic(seed[0])
    before = graph_fingerprint()
    proposal = Proposal(
        id=uuid.uuid4().hex,
        status="pending",
        mode="ingest",
        source="reliability:rollback",
        topics=[
            ProposedTopic(temp_id="c1", title=child, summary="A reasonably long child summary for rollback.", confidence=0.9),
        ],
        dependencies=[ProposedDependency(from_temp_id="c1", to_temp_id=root["id"])],
    )
    save_proposal(proposal)
    applied = apply_proposal(proposal.id)
    mid = graph_fingerprint()
    restore_snapshot(applied.snapshot_id)
    after = graph_fingerprint()
    ok = after == before and child in mid["topics"]
    return {
        "ok": ok,
        "before": before,
        "mid": mid,
        "after": after,
        "snapshot_id": applied.snapshot_id,
    }


def run_reliability_case(case: ReliabilityCase) -> dict[str, Any]:
    if case.kind == "cycle":
        body = _run_cycle(case)
    elif case.kind == "self_loop":
        body = _run_self_loop(case)
    elif case.kind == "unknown_reference":
        body = _run_unknown(case)
    elif case.kind == "out_of_scope_reshape":
        body = _run_reshape_oos(case)
    elif case.kind in ("malformed_json", "missing_fields", "malformed_types"):
        body = _run_malformed(case)
    elif case.kind == "transaction_failure":
        body = _run_transaction(case)
    elif case.kind == "rollback":
        body = _run_rollback(case)
    else:
        body = {"ok": False, "error": f"unknown kind {case.kind}"}
    return {"id": case.id, "kind": case.kind, "description": case.description, **body}


def _rate(ok: int, n: int) -> float:
    return (ok / n) if n else 1.0


def run_reliability_benchmark(*, cases: list[ReliabilityCase] | None = None) -> dict[str, Any]:
    cases = list(cases or reliability_v1())
    results = [run_reliability_case(c) for c in cases]

    def kind_ok(kind: str) -> tuple[int, int]:
        rows = [r for r in results if r["kind"] == kind]
        return sum(1 for r in rows if r.get("ok")), len(rows)

    cycle_ok, cycle_n = kind_ok("cycle")
    loop_ok, loop_n = kind_ok("self_loop")
    unk_ok, unk_n = kind_ok("unknown_reference")
    oos_ok, oos_n = kind_ok("out_of_scope_reshape")
    mal_kinds = ("malformed_json", "missing_fields", "malformed_types")
    mal_rows = [r for r in results if r["kind"] in mal_kinds]
    mal_ok, mal_n = sum(1 for r in mal_rows if r.get("ok")), len(mal_rows)
    tx_ok, tx_n = kind_ok("transaction_failure")
    rb_ok, rb_n = kind_ok("rollback")

    invalid_rows = [
        r
        for r in results
        if r["kind"] in ("cycle", "self_loop", "unknown_reference", "out_of_scope_reshape", *mal_kinds)
    ]
    caught = sum(1 for r in invalid_rows if r.get("synapse_caught") or r.get("ok"))
    invalid_n = len(invalid_rows)

    ref_gen = sum(int(r.get("invalid_references_generated") or 0) for r in results)
    ref_rej = sum(int(r.get("invalid_references_rejected") or 0) for r in results)

    metrics = {
        "validation_catch_rate": _rate(caught, invalid_n),
        "cycle_prevention_rate": _rate(cycle_ok, cycle_n),
        "self_loop_prevention_rate": _rate(loop_ok, loop_n),
        "invalid_reference_rejection_rate": _rate(ref_rej, ref_gen) if ref_gen else _rate(unk_ok + oos_ok, unk_n + oos_n),
        "malformed_output_catch_rate": _rate(mal_ok, mal_n),
        "transaction_integrity_rate": _rate(tx_ok, tx_n),
        "rollback_correctness_rate": _rate(rb_ok, rb_n),
        "n": float(len(results)),
        "n_invalid_outputs": float(invalid_n),
        "n_caught": float(caught),
        "n_cycle_attempts": float(cycle_n),
        "n_unknown_or_oos_refs": float(unk_n + oos_n),
        "n_transaction_failures": float(tx_n),
        "n_rollback_attempts": float(rb_n),
    }
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmark_type": "reliability",
        "dataset": "learning_graph_reliability_v1",
        "dataset_version": "learning_graph_reliability_v1",
        "model": "none",
        "provider": "none",
        "seed": 42,
        "repetitions": 1,
        "example_count": len(results),
        "metrics": metrics,
        "failures": {r["id"]: {"kind": r["kind"], "ok": r.get("ok")} for r in results if not r.get("ok")},
        "latency": {},
        "cost": {"note": "Reliability cases are deterministic and incur no LLM cost."},
        "cases": results,
        "notes": [
            "Direct vs Synapse use production parse_direct_dependency_graph vs build_topics_and_dependencies.",
            "Reshape out-of-scope uses production filter_reshape_new_dependencies.",
            "Transaction/rollback use production apply_proposal and restore_snapshot on an isolated DB.",
            "No LLM calls. Failures are adversarial fixtures, not random model errors.",
        ],
    }
