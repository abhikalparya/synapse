"""LLM-based knowledge rewriting with optional on-disk version snapshots."""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import APIError
from pydantic import ValidationError

from app.models.wiki import WikiPage
from app.prompts.rewrite import build_knowledge_rewrite_prompt
from app.services.llm import call_llm
from app.services.wiki import list_wiki_page_titles, load_all_wiki_pages
from app.services.wiki_schema import persist_validated_wiki_page

logger = logging.getLogger(__name__)

_CORE_FIELDS = ("title", "summary", "key_points", "tags", "related_topics")
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_VERSIONS_DIR = _PROJECT_ROOT / "wiki_pages" / "_versions"


def _strip_json_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _parse_rewrite_core(raw: str) -> dict[str, Any]:
    cleaned = _strip_json_fences(raw)
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("rewrite output is not a JSON object")
    out: dict[str, Any] = {}
    for k in _CORE_FIELDS:
        if k not in data:
            raise ValueError(f"rewrite JSON missing field: {k}")
        out[k] = data[k]
    return out


def version_snapshots_enabled() -> bool:
    raw = os.environ.get("WIKI_VERSION_SNAPSHOTS", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def save_version_snapshot(source_path: Path, body: dict[str, Any]) -> Path | None:
    """
    Persist a copy of the current page JSON under ``wiki_pages/_versions/``.
    Returns the snapshot path, or None if disabled or on failure.
    """
    if not version_snapshots_enabled():
        return None
    try:
        _VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe_stem = source_path.stem.replace("/", "_")[:120]
        out_path = _VERSIONS_DIR / f"{safe_stem}_{stamp}.json"
        payload = {k: v for k, v in body.items() if k != "path"}
        out_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        logger.info("Saved wiki version snapshot: %s", out_path.name)
        return out_path
    except OSError as exc:
        logger.warning("Version snapshot failed for %s: %s", source_path.name, exc)
        return None


def _page_updated_sort_key(row: dict[str, Any]) -> float:
    """Older ``updated_at`` sorts first (stale pages get priority)."""
    u = row.get("updated_at") or row.get("created_at")
    if isinstance(u, datetime):
        dt = u if u.tzinfo else u.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    if isinstance(u, str) and u.strip():
        try:
            dt = datetime.fromisoformat(u.strip().replace("Z", "+00:00"))
            return dt.timestamp()
        except ValueError:
            return 0.0
    return 0.0


async def rewrite_page_core_with_llm(
    body: dict[str, Any],
    *,
    known_page_titles: list[str],
) -> dict[str, Any]:
    """Return validated core fields (title, summary, key_points, tags, related_topics)."""
    page_for_prompt = {k: body.get(k) for k in _CORE_FIELDS}
    prompt = build_knowledge_rewrite_prompt(page_for_prompt, known_page_titles=known_page_titles)
    raw = await call_llm(prompt)
    core = _parse_rewrite_core(raw)
    probe = {**{k: body.get(k) for k in body if k not in _CORE_FIELDS}, **core}
    WikiPage.model_validate({k: v for k, v in probe.items() if k != "path"})
    return core


def merge_rewrite_preserving_meta(body: dict[str, Any], core: dict[str, Any]) -> dict[str, Any]:
    """Apply rewritten core fields; keep provenance, merged_from, confidence, created_at."""
    now = datetime.now(timezone.utc)
    prior = WikiPage.model_validate({k: v for k, v in body.items() if k != "path"})
    created = prior.created_at or now
    out = {
        **body,
        **{k: core[k] for k in _CORE_FIELDS},
        "created_at": created,
        "updated_at": now,
    }
    return out


async def apply_full_rewrite_to_page(path: Path, body: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Optional snapshot, LLM rewrite of core fields, schema persist.
    Returns (ok, errors).
    """
    errs: list[str] = []
    known = list_wiki_page_titles()
    save_version_snapshot(path, body)
    try:
        core = await rewrite_page_core_with_llm(body, known_page_titles=known)
    except (json.JSONDecodeError, ValueError, ValidationError, RuntimeError, APIError) as exc:
        errs.append(f"rewrite LLM failed for {path.name}: {exc}")
        return False, errs
    merged = merge_rewrite_preserving_meta(body, core)
    try:
        await persist_validated_wiki_page(path, merged)
    except (OSError, ValueError, ValidationError, APIError) as exc:
        errs.append(f"rewrite persist failed for {path.name}: {exc}")
        return False, errs
    logger.info("Knowledge rewrite persisted: %s", path.name)
    return True, errs


def refactor_rewrite_max() -> int:
    raw = os.environ.get("REFACTOR_REWRITE_MAX", "0").strip()
    try:
        n = int(raw)
    except ValueError:
        return 0
    return max(0, min(200, n))


async def rewrite_stale_pages_batch(
    max_pages: int,
    *,
    exclude_paths: set[Path] | None = None,
) -> tuple[int, list[str]]:
    """
    Rewrite up to ``max_pages`` wiki files (oldest ``updated_at`` first).
    Skips paths in ``exclude_paths`` (e.g. pages already touched in the same refactor run).
    Returns (success_count, error_strings).
    """
    if max_pages <= 0:
        return 0, []
    excl = exclude_paths or set()
    pages = load_all_wiki_pages()
    candidates = [r for r in pages if r.get("path") not in excl]
    candidates.sort(key=_page_updated_sort_key)
    errors: list[str] = []
    ok_n = 0
    for row in candidates[:max_pages]:
        path = row.get("path")
        if not isinstance(path, Path):
            continue
        title = str(row.get("title", "")).strip()
        if not title:
            continue
        body = {k: v for k, v in row.items() if k != "path"}
        ok, errs = await apply_full_rewrite_to_page(path, body)
        errors.extend(errs)
        if ok:
            ok_n += 1
    return ok_n, errors
