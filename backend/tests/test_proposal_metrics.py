from pathlib import Path

from app.evaluation.proposal_metrics import collect_proposal_metrics
from app.services.proposal_events import (
    log_proposal_applied,
    log_proposal_created,
    log_proposal_discarded,
    log_rollback,
)
from app.models.proposal import Proposal, ProposedTopic


def _proposal(pid: str, *, confidence: float = 0.85) -> Proposal:
    return Proposal(
        id=pid,
        status="pending",
        mode="ingest",
        source="test",
        topics=[ProposedTopic(temp_id="t1", title="A", summary="s", confidence=confidence)],
    )


def test_proposal_metrics_empty_log(tmp_path: Path):
    empty = tmp_path / "none.jsonl"
    result = collect_proposal_metrics(empty)
    assert result["available"] is False


def test_proposal_metrics_from_real_events(tmp_path: Path, monkeypatch):
    log = tmp_path / "events.jsonl"
    monkeypatch.setattr("app.services.proposal_events.PROPOSAL_EVENTS_PATH", log)
    monkeypatch.setenv("SYNAPSE_PROPOSAL_EVENTS_PATH", str(log))

    a = _proposal("aa", confidence=0.9)
    b = _proposal("bb", confidence=0.3)
    log_proposal_created(a)
    log_proposal_created(b)
    a.status = "applied"
    log_proposal_applied(a)
    b.status = "discarded"
    log_proposal_discarded(b)
    log_rollback("snap1")

    result = collect_proposal_metrics(log)
    assert result["available"] is True
    assert result["counts"]["proposals_created"] == 2
    assert result["counts"]["proposals_applied"] == 1
    assert result["counts"]["proposals_discarded"] == 1
    assert result["counts"]["rollbacks"] == 1
    assert result["rates"]["acceptance_rate"] == 0.5
    assert result["rates"]["rejection_rate"] == 0.5
    high = result["confidence_calibration"]["0.8–1.0"]
    low = result["confidence_calibration"]["0.2–0.4"]
    assert high["n"] == 1
    assert high["acceptance_rate"] == 1.0
    assert low["n"] == 1
    assert low["rejection_rate"] == 1.0
