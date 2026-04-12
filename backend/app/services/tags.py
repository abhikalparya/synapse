"""Tag normalization, optional synonym merging, and related-topic alignment."""

import re
from typing import Iterable

# Optional canonical merges (normalized snake_case key -> preferred short tag).
_TAG_ALIASES: dict[str, str] = {
    "ai": "ai",
    "artificial_intelligence": "ai",
    "artificialintelligence": "ai",
    "machine_learning": "ml",
    "machinelearning": "ml",
    "ml": "ml",
    "nlp": "nlp",
    "natural_language_processing": "nlp",
    "natural_language": "nlp",
    "llm": "llm",
    "large_language_model": "llm",
    "large_language_models": "llm",
}


def _to_snake_token(s: str) -> str:
    t = str(s).strip().lower()
    t = re.sub(r"[\s\-–—]+", "_", t, flags=re.UNICODE)
    t = re.sub(r"_+", "_", t).strip("_")
    t = re.sub(r"[^\w_]", "", t, flags=re.UNICODE)
    return t


def normalize_tag(tag: str) -> str:
    """
    Lowercase, snake_case-ish single token, optional alias merge.
    Keeps tags short and consistent for clustering.
    """
    snake = _to_snake_token(tag)
    if not snake:
        return ""
    return _TAG_ALIASES.get(snake, snake)


def normalize_tags_list(tags: Iterable[str] | None) -> list[str]:
    """Lowercase, dedupe, preserve first-seen order; drop empties."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in tags or []:
        n = normalize_tag(str(raw))
        if not n or n in seen:
            continue
        seen.add(n)
        out.append(n)
    return out


def _title_fold_map(titles: Iterable[str]) -> dict[str, str]:
    """Map case-folded title -> one canonical title string (stable pick)."""
    by_fold: dict[str, str] = {}
    for t in sorted({str(x).strip() for x in titles if str(x).strip()}, key=lambda s: s.casefold()):
        k = t.casefold()
        if k not in by_fold:
            by_fold[k] = t
    return by_fold


def align_related_topics(
    related: Iterable[str] | None,
    known_titles: Iterable[str] | None,
) -> list[str]:
    """
    Point each related topic at an existing wiki title when case-insensitive match;
    otherwise keep trimmed original text. Dedupes preserving order.
    """
    title_by_fold = _title_fold_map(known_titles or [])
    out: list[str] = []
    seen: set[str] = set()
    for r in related or []:
        s = str(r).strip()
        if not s:
            continue
        k = s.casefold()
        if k in title_by_fold:
            canon = title_by_fold[k]
        else:
            canon = s
        if canon not in seen:
            seen.add(canon)
            out.append(canon)
    return out
