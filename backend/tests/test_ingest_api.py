"""Minimal API smoke tests for POST /ai/ingest (no real LLM calls)."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.proposal import Proposal


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_ingest_rejects_unknown_generation_strategy(client: TestClient) -> None:
    response = client.post(
        "/ai/ingest",
        json={"goal": "Learn compilers", "generation_strategy": "concept_first"},
    )
    assert response.status_code == 422


def test_ingest_rejects_closed_coverage_recovery_strategy(client: TestClient) -> None:
    response = client.post(
        "/ai/ingest",
        json={"goal": "Learn compilers", "generation_strategy": "baseline_coverage_recovery"},
    )
    assert response.status_code == 422


def test_ingest_baseline_strategy_with_mocked_service(client: TestClient) -> None:
    proposal = Proposal(
        id="test-proposal-id",
        status="pending",
        mode="ingest",
        source="goal: Learn graphs",
        topics=[],
        dependencies=[],
        generation_meta={"generation_strategy": "baseline"},
        created_at=datetime.now(timezone.utc),
    )

    with patch("app.routes.ai.run_ingest", new=AsyncMock(return_value=proposal)) as mock_run:
        response = client.post(
            "/ai/ingest",
            json={"goal": "Learn graphs", "generation_strategy": "baseline"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "test-proposal-id"
    assert body["generation_meta"]["generation_strategy"] == "baseline"
    mock_run.assert_awaited_once()
    kwargs = mock_run.await_args.kwargs
    assert kwargs["generation_strategy"] == "baseline"
    assert kwargs["goal"] == "Learn graphs"
