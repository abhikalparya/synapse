"""Periodic refactor: merge duplicate topics and LLM-rewrite pages for clearer summaries and key points."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import APIError
from pydantic import ValidationError

from app.models.refactor import RefactorResponse
from app.models.wiki import WikiPage
from app.prompts.refactor import build_merge_duplicate_pages_prompt
from app.services.consolidation import (
    augment_summary_with_uncovered_sources,
    build_title_redirect_map,
    collect_merged_from_titles,
    find_duplicate_clusters,
    rewire_related_topics_for_merge,
    union_key_points_with_cluster,
    union_tags_with_cluster,
)
from app.services.llm import call_llm
from app.services.tags import normalize_tags_list
from app.services.wiki import load_all_wiki_pages
from app.services.rewrite import apply_full_rewrite_to_page, refactor_rewrite_max, rewrite_stale_pages_batch
from app.services.wiki_schema import persist_validated_wiki_page, repair_invalid_wiki_files_on_disk

logger = logging.getLogger(__name__)

_WEAK_SUMMARY_MAX_LEN = 120
_WEAK_SUMMARY_MIN_WORDS = 22
_MIN_TAGS = 2
_MIN_KEY_POINTS = 3
_MIN_KEY_POINT_CHARS = 12
_RECENT_MERGE_HOURS = 48


def _strip_json_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _sentence_count(text: str) -> int:
    if not str(text).strip():
        return 0
    return len(re.findall(r"[.!?]+(?:\s|$)", str(text).strip()))


def _is_weak_summary(summary: str) -> bool:
    s = str(summary or "").strip()
    if not s:
        return True
    words = len(re.findall(r"\w+", s, flags=re.UNICODE))
    if words < _WEAK_SUMMARY_MIN_WORDS:
        return True
    if len(s) < _WEAK_SUMMARY_MAX_LEN:
        return True
    if _sentence_count(s) < 2:
        return True
    return False


def _tags_sparse(tags: Any) -> bool:
    return len(normalize_tags_list(tags or [])) < _MIN_TAGS


def _parse_iso_dt(val: Any) -> datetime | None:
    if isinstance(val, datetime):
        return val if val.tzinfo else val.replace(tzinfo=timezone.utc)
    if isinstance(val, str) and val.strip():
        try:
            return datetime.fromisoformat(val.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _summary_has_repetition(summary: str) -> bool:
    s = str(summary or "").strip()
    if len(s) < 40:
        return False
    low = re.sub(r"\s+", " ", s.casefold())
    sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+", low) if len(x.strip()) > 15]
    if len(sentences) >= 2 and len(sentences) != len(set(sentences)):
        return True
    max_win = min(80, len(low))
    for size in range(max_win, 34, -1):
        for i in range(0, len(low) - size + 1):
            chunk = low[i : i + size]
            if low.find(chunk, i + size) != -1:
                return True
    return False


def _is_weak_key_points(key_points: Any) -> bool:
    if not isinstance(key_points, list):
        return True
    pts = [str(p).strip() for p in key_points if str(p).strip()]
    if len(pts) < _MIN_KEY_POINTS:
        return True
    if any(len(p) < _MIN_KEY_POINT_CHARS for p in pts):
        return True
    lows = [p.casefold() for p in pts]
    if len(lows) >= 2 and len(set(lows)) < max(1, int(len(lows) * 0.75)):
        return True
    return False


def _was_recently_merged(body: dict[str, Any]) -> bool:
    mf = body.get("merged_from") or []
    if not isinstance(mf, list) or len(mf) == 0:
        return False
    u = _parse_iso_dt(body.get("updated_at"))
    if u is None:
        return False
    delta = datetime.now(timezone.utc) - u
    return delta.total_seconds() < _RECENT_MERGE_HOURS * 3600


def _refactor_rewrite_reason(body: dict[str, Any]) -> str | None:
    """
    If non-None, the page should receive a refactor-time full LLM rewrite.
    Returns a short human-readable reason for logs (stable phrasing).
    """
    summary = str(body.get("summary", ""))
    if _is_weak_summary(summary):
        return "improved clarity"
    if _summary_has_repetition(summary):
        return "removed redundancy"
    if _is_weak_key_points(body.get("key_points")):
        return "structured key points"
    if _was_recently_merged(body):
        return "post-merge review"
    if _tags_sparse(body.get("tags")):
        return "tag backfill"
    return None


def _pick_canonical(cluster: list[dict]) -> tuple[dict, list[dict]]:
    def richness(p: dict) -> tuple[int, int, str]:
        s = str(p.get("summary") or "")
        kp = p.get("key_points") or []
        n_kp = len(kp) if isinstance(kp, list) else 0
        path_s = str(p.get("path", ""))
        return (len(s), n_kp, path_s)

    primary = max(cluster, key=richness)
    others = [p for p in cluster if p.get("path") != primary.get("path")]
    return primary, others


def _append_unique_notes(acc: list[str], items: Any) -> None:
    seen = {x.casefold() for x in acc}
    for raw in items or []:
        s = str(raw).strip()
        if not s:
            continue
        k = s.casefold()
        if k in seen:
            continue
        seen.add(k)
        acc.append(s)


def _parse_wiki_subset(raw: str) -> dict:
    cleaned = _strip_json_fences(raw)
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("LLM output is not a JSON object")
    return data


def _merge_created_at(cluster: list[dict]) -> datetime | None:
    best: datetime | None = None
    for p in cluster:
        raw = p.get("created_at")
        if raw is None:
            continue
        if isinstance(raw, datetime):
            dt = raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
        else:
            try:
                dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            except ValueError:
                continue
        if best is None or dt < best:
            best = dt
    return best


def _merge_confidence(cluster: list[dict]) -> float | None:
    vals: list[float] = []
    for p in cluster:
        c = p.get("confidence_score")
        if c is None:
            continue
        try:
            vals.append(float(c))
        except (TypeError, ValueError):
            continue
    if not vals:
        return None
    return max(0.0, min(1.0, max(vals)))


async def _merge_duplicate_cluster(cluster: list[dict]) -> tuple[bool, int, list[str], Path | None]:
    """
    Merge cluster into the richest page's file; delete other files; rewire graph links.
    Returns (persisted_ok, pages_removed, errors, canonical_path_if_ok).
    """
    errs: list[str] = []
    primary, others = _pick_canonical(cluster)
    canon_path = primary.get("path")
    if not isinstance(canon_path, Path):
        return False, 0, ["canonical page missing path"], None

    bodies: list[dict] = []
    for p in cluster:
        row = {k: v for k, v in p.items() if k != "path"}
        bodies.append(row)

    try:
        raw = await call_llm(build_merge_duplicate_pages_prompt(bodies))
        merged = _parse_wiki_subset(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        msg = f"merge LLM JSON invalid for cluster titles {[p.get('title') for p in cluster]}: {exc}"
        logger.warning(msg)
        return False, 0, [msg], None

    merged.pop("source_notes", None)
    merged.pop("created_at", None)
    merged.pop("updated_at", None)
    merged.pop("confidence_score", None)
    merged.pop("merged_from", None)
    try:
        page = WikiPage.model_validate(merged)
    except ValidationError as exc:
        msg = f"merge LLM wiki schema invalid: {exc}"
        logger.warning(msg)
        return False, 0, [msg], None

    notes: list[str] = []
    for p in cluster:
        _append_unique_notes(notes, p.get("source_notes"))

    dumped = page.model_dump()
    dumped["source_notes"] = notes
    dumped["created_at"] = _merge_created_at(cluster) or datetime.now(timezone.utc)
    dumped["updated_at"] = datetime.now(timezone.utc)
    dumped["confidence_score"] = _merge_confidence(cluster)
    dumped["merged_from"] = collect_merged_from_titles(cluster)

    cluster_summaries = [str(p.get("summary") or "").strip() for p in cluster if str(p.get("summary") or "").strip()]
    dumped["summary"] = augment_summary_with_uncovered_sources(str(dumped.get("summary", "")), cluster_summaries)
    dumped["key_points"] = union_key_points_with_cluster(dumped.get("key_points") or [], cluster)
    dumped["tags"] = union_tags_with_cluster(dumped.get("tags") or [], cluster)

    other_paths = [p["path"] for p in others if isinstance(p.get("path"), Path)]
    canonical_title = str(dumped.get("title", "")).strip()
    redirect = build_title_redirect_map(cluster, canonical_title)

    try:
        await persist_validated_wiki_page(canon_path, dumped)
    except (OSError, ValueError, ValidationError, APIError) as exc:
        errs.append(f"failed to write merged page {canon_path}: {exc}")
        return False, 0, errs, None

    removed = 0
    for op in other_paths:
        try:
            if op.is_file():
                op.unlink()
                removed += 1
        except OSError as exc:
            errs.append(f"failed to delete merged duplicate {op}: {exc}")

    n_rewired = await rewire_related_topics_for_merge(redirect)
    if n_rewired:
        logger.info(
            "Consolidation rewired related_topics on %s page(s) after merge into %s",
            n_rewired,
            canon_path.name,
        )

    try:
        refreshed = json.loads(canon_path.read_text(encoding="utf-8"))
        await persist_validated_wiki_page(canon_path, refreshed)
    except (OSError, ValueError, json.JSONDecodeError, ValidationError, APIError) as exc:
        errs.append(f"failed to re-save merged page after duplicate removal {canon_path}: {exc}")
    logger.info(
        "Refactor merged duplicate cluster into %s; removed %s file(s); merged_from=%s",
        canon_path.name,
        removed,
        len(dumped.get("merged_from") or []),
    )
    return True, removed, errs, canon_path


async def _refactor_rewrite_page(path: Path, body: dict[str, Any], reason: str) -> tuple[bool, list[str]]:
    """
    Full LLM rewrite of core fields (summary, key_points, tags, related_topics).
    Logs title and improvement reason on success.
    """
    title = str(body.get("title") or path.stem).strip() or path.stem
    ok, errs = await apply_full_rewrite_to_page(path, body)
    if ok:
        logger.info("Refactor rewritten page: %s (%s)", title, reason)
    return ok, errs


async def run_refactor() -> RefactorResponse:
    """
    Scan wiki pages, merge duplicate topics, then LLM-rewrite pages that need quality passes
    (thin or repetitive summaries, weak key points, recent merges, sparse tags) and optionally
    stale pages (REFACTOR_REWRITE_MAX).
    """
    merged_groups = 0
    pages_merged = 0
    pages_updated = 0
    pages_rewritten = 0
    errors: list[str] = []
    touched_paths: set[Path] = set()

    try:
        schema_n = await repair_invalid_wiki_files_on_disk()
        if schema_n:
            logger.info("Refactor schema pass: repaired %s wiki file(s)", schema_n)
    except (OSError, ValueError, RuntimeError, APIError) as exc:
        msg = f"schema repair pass failed: {exc}"
        logger.warning(msg)
        errors.append(msg)

    pages = load_all_wiki_pages()
    clusters = find_duplicate_clusters(pages)

    for cluster in clusters:
        ok, removed, errs, merged_primary = await _merge_duplicate_cluster(cluster)
        errors.extend(errs)
        if ok:
            merged_groups += 1
            pages_merged += removed
            if merged_primary is not None:
                try:
                    raw_data = json.loads(merged_primary.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    errors.append(f"failed to load merged page for rewrite {merged_primary.name}: {exc}")
                else:
                    merged_body = {k: v for k, v in raw_data.items() if k != "path"}
                    rw_ok, rw_errs = await _refactor_rewrite_page(
                        merged_primary,
                        merged_body,
                        "post-merge polish",
                    )
                    errors.extend(rw_errs)
                    if rw_ok:
                        pages_rewritten += 1
                        pages_updated += 1
                        touched_paths.add(merged_primary)

    pages = load_all_wiki_pages()
    for row in pages:
        path = row.get("path")
        if not isinstance(path, Path):
            continue
        if path in touched_paths:
            continue

        body = {k: v for k, v in row.items() if k != "path"}
        reason = _refactor_rewrite_reason(body)
        if reason is None:
            continue

        ok, errs = await _refactor_rewrite_page(path, body, reason)
        errors.extend(errs)
        if ok:
            pages_rewritten += 1
            pages_updated += 1
            touched_paths.add(path)

    max_rw = refactor_rewrite_max()
    if max_rw > 0:
        n_rw, rw_errs = await rewrite_stale_pages_batch(
            max_rw,
            exclude_paths=touched_paths,
        )
        pages_rewritten += n_rw
        pages_updated += n_rw
        errors.extend(rw_errs)

    logger.info(
        "Refactor run complete: merged_groups=%s pages_merged=%s pages_updated=%s pages_rewritten=%s",
        merged_groups,
        pages_merged,
        pages_updated,
        pages_rewritten,
    )
    return RefactorResponse(
        merged_groups=merged_groups,
        pages_merged=pages_merged,
        pages_updated=pages_updated,
        pages_rewritten=pages_rewritten,
        errors=errors,
    )
