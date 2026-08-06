"""Audit mode: read-only structural + LLM-judgment analysis of the current graph. Never
proposes or applies a mutation -- returns a diagnostic AuditReport rendered directly, not
a Proposal. One of four AI operation modes; this is the only one that touches no write
path at all, not even indirectly (no save_proposal, no topics/services mutation calls).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone

from openai import APIError

from app.models.ai_ops import AuditFinding, AuditReport
from app.prompts.audit import build_audit_prompt
from app.services.llm import call_llm
from app.services.proposal_common import parse_llm_json_object
from app.services.topics import load_all_topics, load_dependencies

logger = logging.getLogger(__name__)

_THIN_SUMMARY_MIN_WORDS = 8


def _structural_findings(topics: list[dict], dependencies: list[dict]) -> list[AuditFinding]:
    """Fast, free, purely-structural checks -- no LLM call."""
    findings: list[AuditFinding] = []

    touched: set[str] = set()
    for d in dependencies:
        touched.add(d["from_topic_id"])
        touched.add(d["to_topic_id"])

    seen_titles: dict[str, list[dict]] = defaultdict(list)
    for t in topics:
        title = str(t.get("title", "")).strip() or t["id"]
        seen_titles[title.casefold()].append(t)

        if t["id"] not in touched:
            findings.append(
                AuditFinding(
                    type="orphaned_topic",
                    topic_ids=[t["id"]],
                    detail=f"{title!r} has no dependency edges at all (neither a prerequisite of, nor requiring, anything).",
                ),
            )

        summary = str(t.get("summary", "")).strip()
        word_count = len(summary.split())
        if not summary:
            findings.append(
                AuditFinding(type="thin_topic", topic_ids=[t["id"]], detail=f"{title!r} has no summary at all."),
            )
        elif word_count < _THIN_SUMMARY_MIN_WORDS:
            findings.append(
                AuditFinding(
                    type="thin_topic",
                    topic_ids=[t["id"]],
                    detail=f"{title!r} has a very short summary ({word_count} word(s)).",
                ),
            )

    for dupes in seen_titles.values():
        if len(dupes) > 1:
            title = str(dupes[0].get("title", "")).strip() or dupes[0]["id"]
            findings.append(
                AuditFinding(
                    type="duplicate_title",
                    topic_ids=[t["id"] for t in dupes],
                    detail=f"{len(dupes)} topics share the title {title!r}.",
                ),
            )

    return findings


async def _llm_findings(topics: list[dict], dependencies: list[dict]) -> list[AuditFinding]:
    """Judgment-based checks that need semantic understanding of the topics' content, not
    just graph structure. Degrades gracefully (returns nothing) on any LLM failure --
    audit's structural findings should never be blocked by an LLM hiccup."""
    if not topics:
        return []

    title_by_id = {t["id"]: str(t.get("title", "")).strip() or t["id"] for t in topics}
    prompt = build_audit_prompt(
        topics=[{"title": title_by_id[t["id"]], "summary": str(t.get("summary", ""))} for t in topics],
        edges=[(title_by_id[d["from_topic_id"]], title_by_id[d["to_topic_id"]]) for d in dependencies],
    )

    try:
        raw = await call_llm(prompt)
        data = parse_llm_json_object(raw)
    except (RuntimeError, ValueError, APIError) as exc:
        logger.warning("Audit LLM pass failed, returning structural findings only: %s", exc)
        return []

    title_to_id = {v.casefold(): k for k, v in title_by_id.items()}
    findings: list[AuditFinding] = []
    for row in data.get("findings") or []:
        if not isinstance(row, dict):
            continue
        ftype = str(row.get("type", "")).strip()
        if ftype not in ("missing_prerequisite", "cycle_risk"):
            continue
        detail = str(row.get("detail", "")).strip()
        if not detail:
            continue
        raw_titles = row.get("topic_titles")
        titles = [str(t).strip() for t in raw_titles] if isinstance(raw_titles, list) else []
        ids = [title_to_id[t.casefold()] for t in titles if t.casefold() in title_to_id]
        findings.append(AuditFinding(type=ftype, topic_ids=ids, detail=detail))
    return findings


async def run_audit() -> AuditReport:
    """Read-only: loads topics/dependencies and returns a report. Nothing here writes."""
    topics = load_all_topics()
    dependencies = load_dependencies()

    findings = _structural_findings(topics, dependencies)
    findings.extend(await _llm_findings(topics, dependencies))

    logger.info("Audit report: %s topic(s), %s finding(s)", len(topics), len(findings))
    return AuditReport(generated_at=datetime.now(timezone.utc), total_topics=len(topics), findings=findings)
