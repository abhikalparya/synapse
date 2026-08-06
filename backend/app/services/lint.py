"""Knowledge-graph lint: schema and DAG-consistency checks over topics + dependencies."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from app.models.lint import LintResponse
from app.services.topics import load_all_topics, load_dependencies

logger = logging.getLogger(__name__)


def _topic_title(row: dict[str, Any]) -> str:
    return str(row.get("title", "")).strip() or str(row.get("id", "")).strip() or "untitled"


def _find_cycle_issues(topic_ids: set[str], dependencies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """DFS-based cycle detection; a consistency net in case dependency files were hand-edited."""
    adj: dict[str, list[str]] = defaultdict(list)
    for dep in dependencies:
        adj[dep.get("from_topic_id")].append(dep.get("to_topic_id"))

    issues: list[dict[str, Any]] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def dfs(node: str, path: list[str]) -> None:
        if node in visiting:
            cycle_start = path.index(node)
            issues.append({"type": "cycle", "topics": [*path[cycle_start:], node]})
            return
        if node in visited:
            return
        visiting.add(node)
        for nxt in adj.get(node, []):
            dfs(nxt, [*path, node])
        visiting.discard(node)
        visited.add(node)

    for tid in topic_ids:
        if tid not in visited:
            dfs(tid, [])
    return issues


def collect_lint_issues(topics: list[dict[str, Any]], dependencies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Scan loaded topic/dependency rows; return issue dicts."""
    issues: list[dict[str, Any]] = []
    topic_ids = {str(t.get("id", "")) for t in topics if t.get("id")}

    seen_titles: dict[str, list[str]] = defaultdict(list)
    for t in topics:
        title = _topic_title(t)
        seen_titles[title.casefold()].append(title)
        if not str(t.get("summary", "")).strip():
            issues.append({"type": "missing_summary", "topic": t.get("id"), "detail": title})

    for titles in seen_titles.values():
        if len(titles) > 1:
            issues.append({"type": "duplicate_title", "topics": titles})

    for dep in dependencies:
        from_id = str(dep.get("from_topic_id", ""))
        to_id = str(dep.get("to_topic_id", ""))
        if from_id == to_id:
            issues.append({"type": "self_dependency", "topic": from_id})
        if from_id not in topic_ids or to_id not in topic_ids:
            issues.append(
                {
                    "type": "orphan_dependency",
                    "detail": f"{from_id} -> {to_id} references an unknown topic",
                },
            )

    issues.extend(_find_cycle_issues(topic_ids, dependencies))
    return issues


async def run_lint() -> LintResponse:
    """Load all topics/dependencies and detect schema/DAG consistency issues."""
    topics = load_all_topics()
    dependencies = load_dependencies()
    issue_dicts = collect_lint_issues(topics, dependencies)
    logger.info("run_lint issues=%s", len(issue_dicts))
    return LintResponse.from_issue_dicts(issue_dicts)
