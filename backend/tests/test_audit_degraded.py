import asyncio
from unittest.mock import AsyncMock, patch

from app.services.audit import run_audit


def test_audit_degraded_mode_when_llm_fails():
    topics = [
        {
            "id": "a",
            "title": "Orphan",
            "summary": "short",
            "status": "not_started",
            "resources": [],
            "quiz_passed": False,
            "zone_id": None,
        }
    ]

    async def run():
        with (
            patch("app.services.audit.load_all_topics", return_value=topics),
            patch("app.services.audit.load_dependencies", return_value=[]),
            patch("app.services.audit.call_llm", new=AsyncMock(side_effect=RuntimeError("provider down"))),
        ):
            return await run_audit()

    report = asyncio.run(run())
    assert report.status == "partial"
    assert report.semantic_analysis == "unavailable"
    assert report.semantic_error
    assert report.structural_findings
    assert any(f.type == "orphaned_topic" for f in report.structural_findings)
    assert all(f.type not in ("missing_prerequisite", "cycle_risk") for f in report.findings)


def test_audit_ok_when_llm_returns_empty_findings():
    topics = [
        {
            "id": "a",
            "title": "Root",
            "summary": "A reasonably long summary that is not thin.",
            "status": "not_started",
            "resources": [],
            "quiz_passed": False,
            "zone_id": None,
        },
        {
            "id": "b",
            "title": "Child",
            "summary": "A reasonably long summary that is not thin.",
            "status": "not_started",
            "resources": [],
            "quiz_passed": False,
            "zone_id": None,
        },
    ]
    deps = [{"from_topic_id": "b", "to_topic_id": "a"}]

    async def run():
        with (
            patch("app.services.audit.load_all_topics", return_value=topics),
            patch("app.services.audit.load_dependencies", return_value=deps),
            patch("app.services.audit.call_llm", new=AsyncMock(return_value='{"findings": []}')),
        ):
            return await run_audit()

    report = asyncio.run(run())
    assert report.status == "ok"
    assert report.semantic_analysis == "available"
    assert report.semantic_error is None
