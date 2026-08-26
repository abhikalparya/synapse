import asyncio
from unittest.mock import AsyncMock, patch

from datetime import datetime, timezone

from app.evaluation.audit_eval import eval_repair_from_findings, score_audit_case
from app.evaluation.reliability import graph_fingerprint, run_reliability_benchmark, run_reliability_case
from app.evaluation.reliability_cases import reliability_v1
from app.models.ai_ops import AuditFinding, AuditReport
from app.models.proposal import ProposedDependency, ProposedMerge, ProposedTopic, Proposal
from app.models.topic import TopicCreate
from app.services.audit import audit_graph
from app.services.proposals import apply_proposal, save_proposal
from app.services.reshape import filter_reshape_new_dependencies
from app.services.snapshots import restore_snapshot
from app.services.topics import save_topic


def _by_id():
    return {c.id: c for c in reliability_v1()}


def test_cycle_caught_by_synapse_kept_by_direct():
    result = run_reliability_case(_by_id()["cycle_abc_001"])
    assert result["ok"] is True
    assert result["synapse_caught"] is True
    assert result["direct_retained_invalid"] is True
    assert any("cycle" in (s.get("reason") or "").casefold() for s in result["synapse_skipped"])


def test_self_loop_caught():
    result = run_reliability_case(_by_id()["self_loop_001"])
    assert result["ok"] is True
    assert ["A", "A"] not in result["synapse_dependencies"]
    assert ["A", "A"] in result["direct_dependencies"]


def test_unknown_reference_handled():
    result = run_reliability_case(_by_id()["unknown_ref_001"])
    assert result["ok"] is True
    assert result["invalid_references_rejected"] >= 1


def test_out_of_scope_reshape_handled():
    result = run_reliability_case(_by_id()["reshape_oos_001"])
    assert result["ok"] is True
    assert any("out-of-scope" in s["reason"] or "unknown" in s["reason"] for s in result["skipped"])


def test_malformed_json_handled():
    result = run_reliability_case(_by_id()["malformed_json_001"])
    assert result["ok"] is True
    assert result["synapse_parse_ok"] is False


def test_filter_reshape_uses_production_reason():
    title_to_id = {"a": "id-a", "b": "id-b"}
    proposed, skipped = filter_reshape_new_dependencies(
        [{"from": "B", "to": "Z"}],
        title_to_id=title_to_id,
        accepted_dep_dicts=[],
    )
    assert proposed == []
    assert skipped[0].reason == "unknown or out-of-scope topic reference"


def test_partial_failure_leaves_no_state_change():
    keep = save_topic(TopicCreate(title="Keep-tx", summary="A reasonably long summary for the keep topic."))
    before = graph_fingerprint()
    proposal = Proposal(
        id="tx-fail-test",
        status="pending",
        mode="ingest",
        source="test",
        topics=[
            ProposedTopic(temp_id="n1", title="ShouldNotPersist-tx", summary="A reasonably long new topic summary.", confidence=0.9),
        ],
        merges=[ProposedMerge(source_topic_id="missing-source", target_topic_id=keep["id"])],
    )
    save_proposal(proposal)
    raised = False
    try:
        apply_proposal(proposal.id)
    except ValueError:
        raised = True
    assert raised
    after = graph_fingerprint()
    assert after == before
    assert "ShouldNotPersist-tx" not in after["topics"]


def test_rollback_restores_exact_pre_apply_graph():
    root = save_topic(TopicCreate(title="Root-rb", summary="A reasonably long summary for the root topic."))
    before = graph_fingerprint()
    proposal = Proposal(
        id="rb-test",
        status="pending",
        mode="ingest",
        source="test",
        topics=[
            ProposedTopic(temp_id="c1", title="Child-rb", summary="A reasonably long child summary.", confidence=0.9),
        ],
        dependencies=[ProposedDependency(from_temp_id="c1", to_temp_id=root["id"])],
    )
    save_proposal(proposal)
    applied = apply_proposal(proposal.id)
    mid = graph_fingerprint()
    assert "Child-rb" in mid["topics"]
    restore_snapshot(applied.snapshot_id)
    after = graph_fingerprint()
    assert after == before


def test_reliability_benchmark_reports_rates_without_llm():
    result = run_reliability_benchmark()
    m = result["metrics"]
    assert result["benchmark_type"] == "reliability"
    assert m["validation_catch_rate"] == 1.0
    assert m["cycle_prevention_rate"] == 1.0
    assert m["invalid_reference_rejection_rate"] == 1.0
    assert m["transaction_integrity_rate"] == 1.0
    assert m["rollback_correctness_rate"] == 1.0
    assert m["n"] >= 10


def test_audit_detects_known_structural_issues():
    topics = [
        {
            "id": "a",
            "title": "Orphan",
            "summary": "A reasonably long summary that is not thin.",
            "status": "not_started",
            "resources": [],
            "quiz_passed": False,
            "zone_id": None,
        },
        {
            "id": "b",
            "title": "Orphan",
            "summary": "tiny",
            "status": "not_started",
            "resources": [],
            "quiz_passed": False,
            "zone_id": None,
        },
    ]

    async def run():
        with patch("app.services.audit.call_llm", new=AsyncMock(side_effect=RuntimeError("down"))):
            return await audit_graph(topics, [])

    report = asyncio.run(run())
    types = {f.type for f in report.structural_findings}
    assert "orphaned_topic" in types
    assert "duplicate_title" in types
    assert "thin_topic" in types
    assert report.status == "partial"
    assert report.semantic_analysis == "unavailable"


def test_eval_repair_adds_gold_edge_for_flagged_orphan():
    topics = [
        {"id": "a", "title": "Algebra", "summary": "s"},
        {"id": "c", "title": "Calculus", "summary": "s"},
    ]
    findings = [AuditFinding(type="orphaned_topic", topic_ids=["c"], detail="isolated")]
    repaired = eval_repair_from_findings(topics, [], findings, [["Calculus", "Algebra"]])
    assert ("Calculus", "Algebra") in repaired


def test_score_audit_case_matches_orphan():
    case = {
        "id": "x",
        "graph": {
            "topics": [{"id": "a", "title": "Orphan", "summary": "A reasonably long summary that is not thin."}],
            "dependencies": [],
        },
        "known_issues": [{"type": "orphaned_topic", "topic": "Orphan"}],
    }
    report = AuditReport(
        generated_at=datetime.now(timezone.utc),
        total_topics=1,
        findings=[AuditFinding(type="orphaned_topic", topic_ids=["a"], detail="none")],
        status="partial",
        semantic_analysis="unavailable",
        structural_findings=[AuditFinding(type="orphaned_topic", topic_ids=["a"], detail="none")],
    )
    row = score_audit_case(case, report)
    assert row["precision"] == 1.0
    assert row["recall"] == 1.0
    assert row["false_positive_rate"] == 0.0
