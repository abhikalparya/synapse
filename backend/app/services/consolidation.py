"""
Duplicate-page detection (title string similarity) and merge helpers:
reference rewiring, merged_from metadata, and no-loss unions for key points / tags / summaries.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from openai import APIError
from pydantic import ValidationError

from app.services.tags import normalize_tags_list
from app.services.wiki import load_all_wiki_pages

logger = logging.getLogger(__name__)

_STRING_THRESHOLD = float(os.environ.get("CONSOLIDATION_STRING_THRESHOLD", "0.82"))


def _normalize_title_key(title: str) -> str:
    t = str(title).strip().casefold()
    t = re.sub(r"[^\w\s]+", " ", t, flags=re.UNICODE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def title_similarity_string(a: str, b: str) -> float:
    ka = _normalize_title_key(a)
    kb = _normalize_title_key(b)
    if not ka or not kb:
        return 0.0
    if ka == kb:
        return 1.0
    return SequenceMatcher(None, ka, kb).ratio()


def _find(parent: list[int], i: int) -> int:
    while parent[i] != i:
        i = parent[i]
    return i


def _union(parent: list[int], i: int, j: int) -> None:
    ri, rj = _find(parent, i), _find(parent, j)
    if ri != rj:
        parent[rj] = ri


def collect_merged_from_titles(cluster: list[dict]) -> list[str]:
    """Stable unique list of absorbed titles (page titles + prior merged_from)."""
    acc: list[str] = []
    for p in cluster:
        for raw in (p.get("merged_from") or []):
            s = str(raw).strip()
            if s:
                acc.append(s)
        t = str(p.get("title", "")).strip()
        if t:
            acc.append(t)
    seen: set[str] = set()
    out: list[str] = []
    for s in sorted(acc, key=str.casefold):
        k = s.casefold()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def union_key_points_with_cluster(
    llm_points: list[str],
    cluster: list[dict],
    *,
    max_points: int = 48,
) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []

    def add_seq(items: list[Any]) -> None:
        for x in items:
            t = str(x).strip()
            if not t:
                continue
            k = t.casefold()
            if k in seen:
                continue
            seen.add(k)
            out.append(t)

    add_seq(list(llm_points or []))
    for p in cluster:
        add_seq(list(p.get("key_points") or []))
    return out[:max_points]


def union_tags_with_cluster(llm_tags: list[str], cluster: list[dict], *, max_tags: int = 24) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for t in normalize_tags_list(llm_tags or []):
        k = t.casefold()
        if k in seen:
            continue
        seen.add(k)
        merged.append(t)
    for p in cluster:
        for t in normalize_tags_list(p.get("tags") or []):
            k = t.casefold()
            if k in seen:
                continue
            seen.add(k)
            merged.append(t)
    return normalize_tags_list(merged)[:max_tags]


def _summary_covers(merged: str, original: str) -> bool:
    mo, oo = merged.casefold(), original.casefold()
    if not oo or len(oo) < 16:
        return True
    if oo in mo:
        return True
    owords = set(re.findall(r"\w+", oo, flags=re.UNICODE))
    mwords = set(re.findall(r"\w+", mo, flags=re.UNICODE))
    if not owords:
        return True
    overlap = len(owords & mwords) / len(owords)
    return overlap >= 0.52


def augment_summary_with_uncovered_sources(merged_summary: str, originals: list[str]) -> str:
    """Append source summaries whose substance is not already represented (no-loss guard)."""
    base = (merged_summary or "").strip()
    extra: list[str] = []
    for raw in originals:
        s = str(raw or "").strip()
        if len(s) < 16:
            continue
        if _summary_covers(base, s):
            continue
        extra.append(s)
    if not extra:
        return base
    block = "\n\n---\n\n".join(extra)
    return (base + "\n\n[Additional merged context]\n\n" + block)[:12000]


def build_title_redirect_map(cluster: list[dict], canonical_title: str) -> dict[str, str]:
    """
    Map case-folded title string -> canonical display title for every title in the
    cluster plus prior ``merged_from`` entries (so stale graph links resolve).
    """
    ct = str(canonical_title).strip()
    redir: dict[str, str] = {}
    for p in cluster:
        for raw in (p.get("merged_from") or []):
            s = str(raw).strip()
            if s:
                redir[s.casefold()] = ct
        s = str(p.get("title", "")).strip()
        if s:
            redir[s.casefold()] = ct
    if ct:
        redir[ct.casefold()] = ct
    return redir


async def rewire_related_topics_for_merge(title_redirect: dict[str, str]) -> int:
    """
    Rewrite ``related_topics`` on every wiki page so links target ``canonical_title``
    instead of absorbed titles. Returns number of files written.
    """
    from app.services.wiki_schema import persist_validated_wiki_page

    if not title_redirect:
        return 0
    updated = 0
    for row in load_all_wiki_pages():
        path = row.get("path")
        if not isinstance(path, Path):
            continue
        body = {k: v for k, v in row.items() if k != "path"}
        related = body.get("related_topics")
        if not isinstance(related, list) or not related:
            continue
        new_rel: list[str] = []
        seen: set[str] = set()
        changed = False
        page_title = str(body.get("title", "")).strip().casefold()
        for r in related:
            s = str(r).strip()
            if not s:
                continue
            target = title_redirect.get(s.casefold(), s)
            if target.casefold() == page_title:
                changed = True
                continue
            if target != s:
                changed = True
            k = target.casefold()
            if k in seen:
                continue
            seen.add(k)
            new_rel.append(target)
        if not changed:
            continue
        body["related_topics"] = new_rel
        body["updated_at"] = datetime.now(timezone.utc)
        try:
            await persist_validated_wiki_page(path, body)
            updated += 1
        except (OSError, ValueError, ValidationError, APIError) as exc:
            logger.warning("rewire persist failed for %s: %s", path.name, exc)
    return updated


def find_duplicate_clusters(pages: list[dict]) -> list[list[dict]]:
    """
    Group pages whose titles are duplicates by normalized string similarity
    (``CONSOLIDATION_STRING_THRESHOLD``, default 0.82).
    """
    n = len(pages)
    if n < 2:
        return []

    parent = list(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            ti = str(pages[i].get("title", "")).strip()
            tj = str(pages[j].get("title", "")).strip()
            if not ti or not tj:
                continue
            if title_similarity_string(ti, tj) >= _STRING_THRESHOLD:
                _union(parent, i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        r = _find(parent, i)
        groups.setdefault(r, []).append(i)
    out: list[list[dict]] = []
    for idxs in groups.values():
        cluster = [pages[k] for k in idxs]
        if len(cluster) > 1:
            out.append(cluster)
    return out
