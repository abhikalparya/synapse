"""Focused tests for operation_id correlation across LLM calls, proposals, and lifecycle."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

from app.models.proposal import ProposedDependency, ProposedTopic, Proposal
from app.models.topic import TopicCreate
from app.services.llm import LLMCallRecord, _publish_record, llm_operation
from app.services.operation_context import finalize_generation_meta, get_operation_id, synapse_operation
from app.services.proposal_events import (
    iter_proposal_events,
    log_proposal_applied,
    log_proposal_created,
    log_proposal_discarded,
    log_rollback,
)
from app.services.proposals import apply_proposal, find_applied_proposal_by_snapshot, save_proposal
from app.services.snapshots import restore_snapshot
from app.services.topics import save_topic


def _record(text: str, *, operation: str = "test") -> LLMCallRecord:
    return LLMCallRecord(
        text=text,
        latency_ms=12.5,
        provider="mock",
        model="mock-model",
        input_tokens=100,
        output_tokens=200,
        tokens_estimated=False,
        estimated_cost_usd=0.001,
        success=True,
        operation=operation,
    )


def _ingest_json() -> str:
    return json.dumps(
        {
            "topics": [
                {
                    "title": "Graph Basics",
                    "summary": "Nodes and edges form the foundation of graph theory.",
                    "confidence": 0.9,
                }
            ],
            "dependencies": [],
        }
    )


def test_synapse_operation_generates_operation_id():
    with synapse_operation() as op_id:
        assert op_id
        assert len(op_id) == 32
        assert get_operation_id() == op_id
    assert get_operation_id() is None


def test_finalize_generation_meta_preserves_strategy_fields():
    with synapse_operation():
        meta = finalize_generation_meta(
            {
                "generation_strategy": "domain_curriculum_prior",
                "domain": "machine_learning",
                "inventory_version": "v1",
                "selected_concept_count": 12,
            },
            generation_strategy="domain_curriculum_prior",
        )
    assert meta["generation_strategy"] == "domain_curriculum_prior"
    assert meta["domain"] == "machine_learning"
    assert meta["inventory_version"] == "v1"
    assert meta["selected_concept_count"] == 12
    assert meta["operation_id"]
    assert meta["llm_calls"] == []


def test_multiple_llm_calls_share_operation_id():
    seen_operation_ids: list[str | None] = []

    async def fake_detailed(prompt: str, **kwargs):
        with llm_operation("step"):
            record = _record("{}", operation="step")
            _publish_record(record)
            seen_operation_ids.append(record.operation_id)
            return record

    async def _run():
        with patch("app.services.llm.call_llm_detailed", new=AsyncMock(side_effect=fake_detailed)):
            with synapse_operation() as op_id:
                await fake_detailed("one")
                await fake_detailed("two")
                return op_id, finalize_generation_meta({"generation_strategy": "baseline"})

    op_id, meta = asyncio.run(_run())

    assert len(seen_operation_ids) == 2
    assert all(s == op_id for s in seen_operation_ids)
    assert meta["operation_id"] == op_id
    assert len(meta["llm_calls"]) == 2
    assert {c["operation"] for c in meta["llm_calls"]} == {"step"}


def test_llm_usage_log_includes_operation_id(tmp_path: Path, monkeypatch):
    log_path = tmp_path / "llm_usage.jsonl"
    monkeypatch.setattr("app.services.llm.LLM_USAGE_LOG", log_path)
    monkeypatch.setenv("SYNAPSE_LOG_LLM_USAGE", "1")

    with synapse_operation() as op_id:
        record = _record("{}", operation="ingest")
        _publish_record(record)

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["operation_id"] == op_id
    assert payload["operation"] == "ingest"
    assert payload["provider"] == "mock"
    assert "text" not in payload


def test_proposal_lifecycle_events_include_operation_id(tmp_path: Path, monkeypatch):
    log = tmp_path / "events.jsonl"
    monkeypatch.setattr("app.services.proposal_events.PROPOSAL_EVENTS_PATH", log)
    monkeypatch.setenv("SYNAPSE_PROPOSAL_EVENTS_PATH", str(log))

    proposal = Proposal(
        id="prop-op-1",
        status="pending",
        mode="ingest",
        source="test",
        topics=[ProposedTopic(temp_id="t1", title="A", summary="s", confidence=0.9)],
        generation_meta={"operation_id": "op-abc123", "generation_strategy": "baseline", "llm_calls": []},
    )
    log_proposal_created(proposal)

    proposal.status = "applied"
    proposal.snapshot_id = "snap-xyz"
    log_proposal_applied(proposal)

    proposal.status = "discarded"
    discarded = Proposal(
        id="prop-op-2",
        status="discarded",
        mode="ingest",
        source="test",
        topics=[ProposedTopic(temp_id="t2", title="B", summary="s", confidence=0.5)],
        generation_meta={"operation_id": "op-def456", "generation_strategy": "baseline", "llm_calls": []},
    )
    log_proposal_discarded(discarded)

    events = iter_proposal_events(log)
    created = next(e for e in events if e["event"] == "proposal_created")
    applied = next(e for e in events if e["event"] == "proposal_applied")
    discarded_ev = next(e for e in events if e["event"] == "proposal_discarded")

    assert created["operation_id"] == "op-abc123"
    assert applied["operation_id"] == "op-abc123"
    assert applied["snapshot_id"] == "snap-xyz"
    assert discarded_ev["operation_id"] == "op-def456"


def test_rollback_links_proposal_when_deterministic(tmp_path: Path, monkeypatch):
    log = tmp_path / "events.jsonl"
    monkeypatch.setattr("app.services.proposal_events.PROPOSAL_EVENTS_PATH", log)
    monkeypatch.setenv("SYNAPSE_PROPOSAL_EVENTS_PATH", str(log))

    root = save_topic(TopicCreate(title="Root-cor", summary="A reasonably long summary for correlation testing."))
    proposal = Proposal(
        id="rollback-chain",
        status="pending",
        mode="ingest",
        source="test",
        topics=[
            ProposedTopic(
                temp_id="c1",
                title="Child-cor",
                summary="A reasonably long child summary for correlation testing.",
                confidence=0.9,
            )
        ],
        dependencies=[ProposedDependency(from_temp_id="c1", to_temp_id=root["id"])],
        generation_meta={"operation_id": "op-rollback-1", "generation_strategy": "baseline", "llm_calls": []},
    )
    save_proposal(proposal)
    applied = apply_proposal(proposal.id)
    assert applied.snapshot_id

    linked = find_applied_proposal_by_snapshot(applied.snapshot_id)
    assert linked is not None
    assert linked.id == proposal.id

    restore_snapshot(applied.snapshot_id)

    rollback_ev = next(e for e in iter_proposal_events(log) if e["event"] == "rollback")
    assert rollback_ev["snapshot_id"] == applied.snapshot_id
    assert rollback_ev["proposal_id"] == proposal.id
    assert rollback_ev["operation_id"] == "op-rollback-1"


def test_rollback_without_resolvable_proposal_does_not_invent_ids(tmp_path: Path, monkeypatch):
    log = tmp_path / "events.jsonl"
    monkeypatch.setattr("app.services.proposal_events.PROPOSAL_EVENTS_PATH", log)
    monkeypatch.setenv("SYNAPSE_PROPOSAL_EVENTS_PATH", str(log))

    log_rollback("orphan-snapshot-id")

    event = iter_proposal_events(log)[0]
    assert event["event"] == "rollback"
    assert event["snapshot_id"] == "orphan-snapshot-id"
    assert "proposal_id" not in event
    assert "operation_id" not in event


def test_ingest_correlation_chain(tmp_path: Path, monkeypatch):
    log = tmp_path / "events.jsonl"
    usage_log = tmp_path / "llm_usage.jsonl"
    monkeypatch.setattr("app.services.proposal_events.PROPOSAL_EVENTS_PATH", log)
    monkeypatch.setenv("SYNAPSE_PROPOSAL_EVENTS_PATH", str(log))
    monkeypatch.setattr("app.services.llm.LLM_USAGE_LOG", usage_log)
    monkeypatch.setenv("SYNAPSE_LOG_LLM_USAGE", "1")

    async def fake_detailed(prompt: str, **kwargs):
        from app.services.llm import _llm_operation, _publish_record

        record = _record(_ingest_json(), operation=_llm_operation.get())
        _publish_record(record)
        return record

    async def _run():
        with patch("app.services.ingest.call_llm_detailed", new=AsyncMock(side_effect=fake_detailed)):
            with patch("app.services.ingest.load_all_topics", return_value=[]):
                from app.services.ingest import run_ingest

                return await run_ingest(
                    goal="Learn graph basics",
                    topics=None,
                    filenames=None,
                    generation_strategy="baseline",
                )

    proposal = asyncio.run(_run())

    op_id = proposal.generation_meta["operation_id"]
    assert op_id
    assert proposal.generation_meta["generation_strategy"] == "baseline"
    assert len(proposal.generation_meta["llm_calls"]) == 1
    call_summary = proposal.generation_meta["llm_calls"][0]
    assert call_summary["operation"] == "ingest"
    assert call_summary["success"] is True

    usage_payload = json.loads(usage_log.read_text(encoding="utf-8").strip())
    assert usage_payload["operation_id"] == op_id

    created = next(e for e in iter_proposal_events(log) if e["event"] == "proposal_created")
    assert created["proposal_id"] == proposal.id
    assert created["operation_id"] == op_id

    applied = apply_proposal(proposal.id)
    applied_ev = next(e for e in iter_proposal_events(log) if e["event"] == "proposal_applied")
    assert applied_ev["operation_id"] == op_id
    assert applied_ev["snapshot_id"] == applied.snapshot_id
