"""Knowledge-base lint: duplicates, tags, summaries, key points, formatting."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from openai import APIError
from pydantic import ValidationError

from app.models.lint import LintResponse
from app.prompts.lint import build_lint_suggestions_prompt
from app.services.consolidation import find_duplicate_clusters
from app.services.llm import call_llm
from app.services.tags import normalize_tag, normalize_tags_list
from app.services.wiki import load_all_wiki_pages
from app.services.wiki_schema import (
    coerce_knowledge_core,
    core_fields_differ_from_coercion,
    validate_knowledge_schema,
)

logger = logging.getLogger(__name__)

_TAG_PATTERN = re.compile(r"^[a-z0-9_]+$")
_MIN_TAGS = 2
_MIN_KEY_POINTS = 3
_MIN_BULLET_LEN = 12


def _strip_json_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _page_title(row: dict) -> str:
    t = str(row.get("title", "")).strip()
    if t:
        return t
    path = row.get("path")
    if path is not None:
        return path.stem
    return "untitled"


def _weak_key_points(raw: Any) -> tuple[bool, str | None]:
    if not isinstance(raw, list):
        return True, "key_points is not a list"
    strings = [x for x in raw if isinstance(x, str) and x.strip()]
    if len(strings) < _MIN_KEY_POINTS:
        return True, f"fewer than {_MIN_KEY_POINTS} non-empty string bullets"
    short = [s for s in strings if len(s.strip()) < _MIN_BULLET_LEN]
    if len(short) > len(strings) // 2 and short:
        return True, "many bullets are very short"
    if any(not isinstance(x, str) for x in raw if x is not None):
        return True, "non-string entries in key_points"
    return False, None


def _raw_tags_inconsistent(raw: Any) -> tuple[bool, str | None]:
    """True if raw tags are not already normalized snake lowercase tokens."""
    if raw is None:
        return False, None
    if isinstance(raw, str):
        items = [raw] if raw.strip() else []
    elif isinstance(raw, list):
        items = [str(x) for x in raw if x is not None and str(x).strip()]
    else:
        return True, "tags is not a list or string"
    for t in items:
        s = str(t).strip()
        if not s:
            continue
        n = normalize_tag(s)
        if s != n or not _TAG_PATTERN.match(n):
            return True, "tags should be lowercase snake_case tokens"
    return False, None


def _title_whitespace_issue(row: dict) -> bool:
    raw = row.get("title")
    if raw is None:
        return False
    s = str(raw)
    return s != s.strip()


def collect_lint_issues(pages: list[dict]) -> list[dict[str, Any]]:
    """Scan loaded wiki rows (with ``path``); return issue dicts."""
    issues: list[dict[str, Any]] = []

    for cluster in find_duplicate_clusters(pages):
        titles_unique = sorted(
            {_page_title(p) for p in cluster},
            key=str.casefold,
        )
        if len(titles_unique) > 1:
            issues.append({"type": "duplicate", "pages": titles_unique})
        elif len(cluster) > 1:
            t0 = titles_unique[0] if titles_unique else _page_title(cluster[0])
            issues.append(
                {
                    "type": "duplicate",
                    "pages": [t0],
                    "detail": f"{len(cluster)} wiki JSON files share this title (merge recommended)",
                },
            )

    for row in pages:
        title = _page_title(row)
        body = {k: v for k, v in row.items() if k != "path"}
        summary = str(body.get("summary", "")).strip()
        tag_count = len(normalize_tags_list(body.get("tags") or []))

        if tag_count < _MIN_TAGS:
            issues.append({"type": "missing_tags", "page": title})
        if not summary:
            issues.append({"type": "empty_summary", "page": title})

        weak, weak_detail = _weak_key_points(body.get("key_points"))
        if weak:
            issues.append(
                {
                    "type": "weak_key_points",
                    "page": title,
                    "detail": weak_detail,
                },
            )

        fmt_parts: list[str] = []
        if _title_whitespace_issue(row):
            fmt_parts.append("title has leading or trailing whitespace")
        inc_tags, tag_detail = _raw_tags_inconsistent(body.get("tags"))
        if inc_tags and tag_detail:
            fmt_parts.append(tag_detail)

        coerced = coerce_knowledge_core(dict(body))
        schema_errs = validate_knowledge_schema(coerced)
        if not summary:
            schema_errs = [e for e in schema_errs if "summary" not in e.casefold()]
        if tag_count < _MIN_TAGS:
            schema_errs = [e for e in schema_errs if "tags" not in e.casefold()]
        if weak:
            schema_errs = [e for e in schema_errs if "key_points" not in e.casefold()]
        if schema_errs:
            fmt_parts.extend(schema_errs[:8])
        if core_fields_differ_from_coercion(body, coerced):
            fmt_parts.append("core fields differ from normalized form (coercion would change file)")

        if fmt_parts:
            issues.append(
                {
                    "type": "inconsistent_formatting",
                    "page": title,
                    "detail": "; ".join(dict.fromkeys(fmt_parts)),
                },
            )

    return issues


def _parse_suggestions_payload(raw: str) -> list[dict[str, Any]]:
    cleaned = _strip_json_fences(raw)
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        return []
    sug = data.get("suggestions")
    if not isinstance(sug, list):
        return []
    return [x for x in sug if isinstance(x, dict)]


def _match_issue_key(issue: dict[str, Any]) -> tuple[str, str | None, tuple[str, ...]]:
    pages = issue.get("pages")
    if isinstance(pages, list):
        pt = tuple(sorted((str(p).strip() for p in pages if str(p).strip()), key=str.casefold))
        return (str(issue.get("type", "")), None, pt)
    return (str(issue.get("type", "")), str(issue.get("page", "")).strip() or None, ())


def _merge_suggestions(issues: list[dict[str, Any]], suggestions: list[dict[str, Any]]) -> None:
    by_key: dict[tuple, str] = {}
    for s in suggestions:
        typ = str(s.get("type", "")).strip()
        page = s.get("page")
        pages = s.get("pages")
        text = str(s.get("suggestion", "")).strip()
        if not typ or not text:
            continue
        if isinstance(pages, list) and pages:
            key = (typ, None, tuple(sorted((str(p).strip() for p in pages), key=str.casefold)))
        else:
            key = (typ, str(page).strip() if page else None, ())
        by_key[key] = text

    for issue in issues:
        key = _match_issue_key(issue)
        if key in by_key:
            issue["suggestion"] = by_key[key]


async def run_lint(*, suggest_fixes: bool = False) -> LintResponse:
    """
    Load all wiki pages, detect quality issues, and optionally attach LLM suggestions.
    """
    pages = load_all_wiki_pages()
    issue_dicts = collect_lint_issues(pages)

    if suggest_fixes and issue_dicts:
        try:
            prompt = build_lint_suggestions_prompt(issue_dicts[:45])
            raw = await call_llm(prompt)
            sug_rows = _parse_suggestions_payload(raw)
            _merge_suggestions(issue_dicts, sug_rows)
        except (
            json.JSONDecodeError,
            ValueError,
            ValidationError,
            OSError,
            RuntimeError,
            APIError,
        ) as exc:
            logger.warning("Lint LLM suggestions failed: %s", exc)

    logger.info("run_lint issues=%s suggest=%s", len(issue_dicts), suggest_fixes)
    return LintResponse.from_issue_dicts(issue_dicts)
