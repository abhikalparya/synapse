from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select, text

from app.db.models import TopicRow
from app.db.session import SessionLocal, ensure_topic_identity_schema, engine
from app.main import app
from app.models.proposal import ProposedDependency, ProposedTopic, Proposal
from app.models.topic import TopicCreate
from app.services.proposal_common import build_topics_and_dependencies
from app.services.proposals import apply_proposal, save_proposal
from app.services.reshape import run_reshape
from app.services.topic_identity import canonical_topic_title
from app.services.topics import load_all_topics, save_topic
from app.services.llm import LLMCallRecord


def _title(prefix: str) -> str:
    return f"{prefix} {uuid4().hex}"


def test_canonical_topic_title_is_conservative_and_deterministic():
    variants = [
        "Transformer Architecture",
        "transformer architecture",
        "Transformer-Architecture",
        "Transformer   Architecture",
    ]

    assert {canonical_topic_title(title) for title in variants} == {"transformer architecture"}
    assert canonical_topic_title("Café") == "café"


def test_builder_collapses_duplicate_topics_and_remaps_dependencies():
    proposed_topics, proposed_dependencies, skipped = build_topics_and_dependencies(
        [
            {"title": "Transformer Architecture", "summary": "first", "confidence": 0.9},
            {"title": "transformer-architecture", "summary": "duplicate", "confidence": 0.8},
            {"title": "Attention", "summary": "attention", "confidence": 0.8},
        ],
        [
            {"from": "Transformer-Architecture", "to": "Attention"},
            {"from": "transformer architecture", "to": "Transformer   Architecture"},
        ],
        confidence_threshold=0.6,
    )

    assert [topic.title for topic in proposed_topics] == ["Transformer Architecture", "Attention"]
    assert len(proposed_dependencies) == 1
    assert proposed_dependencies[0].to_temp_id == next(
        topic.temp_id for topic in proposed_topics if topic.title == "Attention"
    )
    assert any("self-dependency" in item.reason for item in skipped)


def test_builder_resolves_proposed_topic_against_existing_graph():
    existing = save_topic(TopicCreate(title=_title("Existing Topic"), summary="Existing summary"))
    existing_variant = existing["title"].upper().replace(" ", "-")
    new_title = _title("New Topic")

    proposed_topics, proposed_dependencies, skipped = build_topics_and_dependencies(
        [
            {"title": existing_variant, "summary": "would be duplicate", "confidence": 0.9},
            {"title": new_title, "summary": "new summary", "confidence": 0.9},
        ],
        [{"from": new_title, "to": existing_variant}],
        confidence_threshold=0.6,
        existing_topics=load_all_topics(),
    )

    assert [topic.title for topic in proposed_topics] == [new_title]
    assert len(proposed_dependencies) == 1
    assert proposed_dependencies[0].to_temp_id == existing["id"]
    assert skipped == []


def test_reshape_resolves_existing_and_duplicate_new_topics():
    selected = save_topic(TopicCreate(title=_title("Selected Topic"), summary="selected"))
    existing = save_topic(TopicCreate(title=_title("Reshape Existing"), summary="existing"))
    new_title = _title("Reshape New")
    payload = {
        "new_topics": [
            {"title": existing["title"].upper().replace(" ", "-"), "summary": "duplicate", "confidence": 0.9},
            {"title": new_title, "summary": "new", "confidence": 0.9},
            {"title": new_title.replace(" ", "-"), "summary": "duplicate new", "confidence": 0.8},
        ],
        "new_dependencies": [
            {"from": new_title.replace(" ", "-"), "to": existing["title"]},
            {"from": existing["title"], "to": existing["title"].upper().replace(" ", "-")},
        ],
        "removed_dependencies": [],
        "merges": [],
        "edits": [],
    }
    record = LLMCallRecord(
        text=json.dumps(payload),
        latency_ms=1,
        provider="mock",
        model="mock",
        input_tokens=1,
        output_tokens=1,
        tokens_estimated=False,
        estimated_cost_usd=0,
        success=True,
        operation="reshape",
    )

    with patch("app.services.reshape.call_llm_detailed", new=AsyncMock(return_value=record)):
        proposal = asyncio.run(run_reshape(topic_ids=[selected["id"]], instructions=None))

    assert [topic.title for topic in proposal.topics] == [new_title]
    assert len(proposal.dependencies) == 1
    assert proposal.dependencies[0].to_temp_id == existing["id"]
    assert any("self-dependency" in item.reason for item in proposal.skipped_dependencies)


def test_apply_rechecks_collision_created_after_proposal_generation():
    title = _title("Race Safe Topic")
    first = Proposal(
        id=f"proposal-{uuid4().hex}",
        status="pending",
        mode="ingest",
        source="test",
        topics=[ProposedTopic(temp_id="first", title=title, summary="first")],
    )
    second = Proposal(
        id=f"proposal-{uuid4().hex}",
        status="pending",
        mode="ingest",
        source="test",
        topics=[ProposedTopic(temp_id="second", title=title.upper(), summary="second")],
    )
    save_proposal(first)
    save_proposal(second)

    applied_second = apply_proposal(second.id)
    applied_first = apply_proposal(first.id)

    assert len(applied_second.created_topics) == 1
    assert applied_first.created_topics == []
    with SessionLocal() as session:
        rows = session.scalars(
            select(TopicRow).where(TopicRow.canonical_title == canonical_topic_title(title))
        ).all()
    assert len(rows) == 1


def test_apply_collapses_malformed_duplicate_topics_and_self_dependency():
    title = _title("Malformed Proposal Topic")
    proposal = Proposal(
        id=f"proposal-{uuid4().hex}",
        status="pending",
        mode="ingest",
        source="test",
        topics=[
            ProposedTopic(temp_id="one", title=title, summary="first"),
            ProposedTopic(temp_id="two", title=title.replace(" ", "-"), summary="duplicate"),
        ],
        dependencies=[ProposedDependency(from_temp_id="one", to_temp_id="two")],
    )
    save_proposal(proposal)

    result = apply_proposal(proposal.id)

    assert len(result.created_topics) == 1
    assert result.created_dependencies == []
    with SessionLocal() as session:
        rows = session.scalars(
            select(TopicRow).where(TopicRow.canonical_title == canonical_topic_title(title))
        ).all()
    assert len(rows) == 1


def test_direct_topic_creation_rejects_canonical_duplicate():
    client = TestClient(app)
    title = _title("Manual Topic")

    created = client.post("/topics", json={"title": title, "summary": "manual"})
    duplicate = client.post("/topics", json={"title": title.upper().replace(" ", "-"), "summary": "duplicate"})

    assert created.status_code == 201
    assert duplicate.status_code == 409


def test_existing_duplicate_database_backfills_without_unique_index(tmp_path: Path):
    legacy_engine = create_engine(f"sqlite:///{tmp_path / 'legacy.db'}")
    with legacy_engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE topics (
                    id VARCHAR(32) PRIMARY KEY,
                    title VARCHAR(500) NOT NULL,
                    summary TEXT NOT NULL,
                    status VARCHAR(20) NOT NULL,
                    quiz_passed BOOLEAN NOT NULL,
                    zone_id VARCHAR(32),
                    created_at DATETIME NOT NULL,
                    updated_at DATETIME NOT NULL
                )
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO topics
                    (id, title, summary, status, quiz_passed, created_at, updated_at)
                VALUES
                    ('legacy-one', 'Legacy Topic', '', 'not_started', 0, '2026-01-01', '2026-01-01'),
                    ('legacy-two', 'legacy-topic', '', 'not_started', 0, '2026-01-02', '2026-01-02')
                """
            )
        )

    ensure_topic_identity_schema(legacy_engine)
    ensure_topic_identity_schema(legacy_engine)

    with legacy_engine.connect() as connection:
        columns = {row[1] for row in connection.execute(text("PRAGMA table_info(topics)"))}
        rows = connection.execute(
            text("SELECT id, canonical_title FROM topics ORDER BY id")
        ).all()
        indexes = list(connection.execute(text("PRAGMA index_list(topics)")))

    assert "canonical_title" in columns
    assert [row[1] for row in rows] == ["legacy topic", "legacy topic"]
    assert not any(row[1] == "uq_topics_canonical_title" for row in indexes)


def test_clean_database_gets_canonical_unique_index():
    with engine.connect() as connection:
        indexes = list(connection.execute(text("PRAGMA index_list(topics)")))
    assert any(row[1] == "uq_topics_canonical_title" and row[2] for row in indexes)
