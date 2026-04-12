"""Load schema.json, validate wiki knowledge fields, and LLM-repair on violation."""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.models.wiki import WikiPage
from app.prompts.schema import build_wiki_schema_repair_prompt
from app.services.llm import call_llm
from app.services.tags import align_related_topics, normalize_tags_list

logger = logging.getLogger(__name__)

_SCHEMA_PATH = Path(__file__).resolve().parent.parent.parent / "schema.json"
_TAG_PATTERN = re.compile(r"^[a-z0-9_]+$")


@lru_cache(maxsize=1)
def _load_schema_raw() -> dict[str, Any]:
    try:
        text = _SCHEMA_PATH.read_text(encoding="utf-8")
    except OSError:
        logger.warning("schema.json missing at %s; using defaults", _SCHEMA_PATH)
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("schema.json invalid JSON: %s; using defaults", exc)
        return {}


def schema_title_max_length() -> int:
    raw = _load_schema_raw().get("rules") or {}
    try:
        n = int(raw.get("title_max_length", 200))
    except (TypeError, ValueError):
        return 200
    return max(32, min(500, n))


def required_knowledge_fields() -> tuple[str, ...]:
    req = _load_schema_raw().get("required")
    if isinstance(req, list) and req:
        return tuple(str(x) for x in req)
    return ("title", "summary", "key_points", "tags", "related_topics")


def _strip_json_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _to_str_list(value: Any, *, field: str, errors: list[str]) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [p.strip() for p in value.replace("\r\n", "\n").split("\n") if p.strip()]
        if len(parts) <= 1 and field == "key_points":
            return [value.strip()] if value.strip() else []
        return parts if parts else ([] if not value.strip() else [value.strip()])
    if isinstance(value, list):
        out: list[str] = []
        for i, item in enumerate(value):
            if item is None:
                continue
            if isinstance(item, str):
                s = item.strip()
                if s:
                    out.append(s)
            else:
                s = str(item).strip()
                if s:
                    out.append(s)
        return out
    errors.append(f"{field} must be a list or string (got {type(value).__name__})")
    return []


def coerce_knowledge_core(data: dict[str, Any]) -> dict[str, Any]:
    """
    Return a shallow copy with normalized core list/string shapes (no LLM).
    Preserves non-core keys (e.g. source_notes, created_at).
    """
    out = dict(data)
    errs: list[str] = []
    out["title"] = str(out.get("title", "")).strip()
    out["summary"] = str(out.get("summary", "")).strip()
    out["key_points"] = _to_str_list(out.get("key_points"), field="key_points", errors=errs)
    out["related_topics"] = _to_str_list(
        out.get("related_topics"), field="related_topics", errors=errs
    )
    raw_tags = out.get("tags")
    if isinstance(raw_tags, str):
        raw_tags = [raw_tags] if raw_tags.strip() else []
    if not isinstance(raw_tags, list):
        raw_tags = []
    out["tags"] = normalize_tags_list(raw_tags)
    for e in errs:
        logger.debug("coerce_knowledge_core: %s", e)
    return out


def core_fields_differ_from_coercion(body: dict[str, Any], coerced: dict[str, Any]) -> bool:
    """True if normalized core fields differ from the raw page (needs rewrite)."""
    for k in required_knowledge_fields():
        if body.get(k) != coerced.get(k):
            return True
    return False


def validate_knowledge_schema(data: dict[str, Any]) -> list[str]:
    """
    Validate required fields and rules from schema.json.
    Expects ``data`` without ``path``; call after ``coerce_knowledge_core``.
    """
    errors: list[str] = []
    for key in required_knowledge_fields():
        if key not in data:
            errors.append(f"missing required field: {key}")

    title = str(data.get("title", "")).strip()
    if not title:
        errors.append("title must not be blank")
    tmax = schema_title_max_length()
    if len(title) > tmax:
        errors.append(f"title exceeds concise limit ({len(title)} > {tmax} chars)")

    if "summary" in data and not str(data.get("summary", "")).strip():
        errors.append("summary must not be blank")

    kp = data.get("key_points")
    if not isinstance(kp, list):
        errors.append("key_points must be a list")
    else:
        for i, item in enumerate(kp):
            if not isinstance(item, str):
                errors.append(f"key_points[{i}] must be a string")

    tags = data.get("tags")
    if not isinstance(tags, list):
        errors.append("tags must be a list")
    else:
        for i, item in enumerate(tags):
            if not isinstance(item, str):
                errors.append(f"tags[{i}] must be a string")
            else:
                if item != item.lower():
                    errors.append(f"tags[{i}] must be lowercase")
                elif not _TAG_PATTERN.match(item):
                    errors.append(f"tags[{i}] must match lowercase snake token pattern")

    rt = data.get("related_topics")
    if not isinstance(rt, list):
        errors.append("related_topics must be a list")
    else:
        for i, item in enumerate(rt):
            if not isinstance(item, str):
                errors.append(f"related_topics[{i}] must be a string")

    return errors


def _parse_repair_response(raw: str) -> dict[str, Any]:
    cleaned = _strip_json_fences(raw)
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("repair response is not a JSON object")
    return data


def finalize_tags_and_related_inplace(data: dict[str, Any], known_titles: list[str]) -> None:
    """Normalize tags and align related_topics (same rules as wiki service)."""
    data["tags"] = normalize_tags_list(data.get("tags") or [])
    title_fold = str(data.get("title", "")).strip().casefold()
    related = align_related_topics(data.get("related_topics") or [], known_titles)
    data["related_topics"] = [x for x in related if str(x).strip().casefold() != title_fold]


async def repair_core_fields_with_llm(
    page: dict[str, Any],
    violations: list[str],
    *,
    known_page_titles: list[str] | None = None,
) -> dict[str, Any]:
    prompt = build_wiki_schema_repair_prompt(
        page, violations, known_page_titles=known_page_titles
    )
    raw = await call_llm(prompt)
    fixed = _parse_repair_response(raw)
    core = required_knowledge_fields()
    out: dict[str, Any] = {}
    for k in core:
        if k not in fixed:
            raise ValueError(f"repair JSON missing field: {k}")
        out[k] = fixed[k]
    return out


async def ensure_wiki_schema_compliant(
    data: dict[str, Any],
    *,
    known_page_titles: list[str] | None = None,
    max_llm_repairs: int = 2,
) -> dict[str, Any]:
    """
    Coerce primitives, validate against schema.json, and call the LLM to rewrite core
    fields when validation still fails. Preserves non-core keys from ``data``.
    """
    working = {k: v for k, v in data.items() if k != "path"}
    for k in required_knowledge_fields():
        if k not in working:
            working[k] = [] if k in ("key_points", "tags", "related_topics") else ""

    titles_for_prompt = known_page_titles
    attempts = 0
    while True:
        working = coerce_knowledge_core(working)
        violations = validate_knowledge_schema(working)
        if not violations:
            return working
        if attempts >= max_llm_repairs:
            raise ValueError(
                "wiki page still violates knowledge schema after repair: " + "; ".join(violations)
            )
        logger.info(
            "Wiki schema repair via LLM (attempt %s): %s",
            attempts + 1,
            "; ".join(violations[:5]) + ("..." if len(violations) > 5 else ""),
        )
        page_for_prompt = {k: working.get(k) for k in required_knowledge_fields()}
        repaired = await repair_core_fields_with_llm(
            page_for_prompt,
            violations,
            known_page_titles=titles_for_prompt,
        )
        for k in required_knowledge_fields():
            working[k] = repaired[k]
        attempts += 1


def assert_full_wiki_model(data: dict[str, Any]) -> WikiPage:
    """Final pydantic check including optional server fields."""
    return WikiPage.model_validate(data)


async def persist_validated_wiki_page(path: Path, data: dict[str, Any]) -> None:
    """Ensure schema compliance, align tags/related, validate WikiPage, write JSON."""
    from app.services.wiki import list_wiki_page_titles

    clean = {k: v for k, v in data.items() if k != "path"}
    known = list_wiki_page_titles()
    fixed = await ensure_wiki_schema_compliant(clean, known_page_titles=known)
    finalize_tags_and_related_inplace(fixed, known)
    page = assert_full_wiki_model(fixed)
    path.write_text(
        json.dumps(page.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


async def repair_invalid_wiki_files_on_disk() -> int:
    """
    Scan wiki_pages for schema violations and rewrite files in place after LLM repair.
    Returns count of files written.
    """
    from app.services.wiki import load_all_wiki_pages

    repaired = 0
    for row in load_all_wiki_pages():
        path = row.get("path")
        if not isinstance(path, Path):
            continue
        body = {k: v for k, v in row.items() if k != "path"}
        probe = coerce_knowledge_core(dict(body))
        if validate_knowledge_schema(probe) and not core_fields_differ_from_coercion(body, probe):
            continue
        await persist_validated_wiki_page(path, body)
        repaired += 1
        logger.info("Schema repair persisted for %s", path.name)
    return repaired
