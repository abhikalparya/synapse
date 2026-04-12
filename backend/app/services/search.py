import re
from typing import Any

from app.services.tags import normalize_tags_list


def _tokenize(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 1}


def _page_search_blob(page: dict[str, Any]) -> str:
    tags = normalize_tags_list(page.get("tags"))
    parts: list[str] = [
        str(page.get("title", "")),
        str(page.get("summary", "")),
        " ".join(str(p) for p in page.get("key_points") or []),
        " ".join(tags),
        " ".join(str(r) for r in page.get("related_topics") or []),
    ]
    return " ".join(parts)


def find_relevant_pages(query: str, pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Rank wiki pages by simple keyword overlap between the query and page text.

    Each page dict must include at least: path (Path), title, summary, key_points,
    tags, related_topics. Returns a new list sorted by score descending; pages
    with score 0 are omitted.
    """
    q_tokens = _tokenize(query)
    if not q_tokens:
        return []

    scored: list[tuple[int, dict[str, Any]]] = []
    for page in pages:
        blob = _page_search_blob(page)
        b_tokens = _tokenize(blob)
        if not b_tokens:
            continue
        score = 0
        for tok in q_tokens:
            if tok in b_tokens:
                score += blob.lower().count(tok)
            title_l = str(page.get("title", "")).lower()
            if tok in title_l:
                score += 3
            for tag in normalize_tags_list(page.get("tags")):
                if tok in tag:
                    score += 2
        if score > 0:
            scored.append((score, page))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored]
