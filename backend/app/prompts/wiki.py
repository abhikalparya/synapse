"""Reusable prompts for wiki generation from raw notes."""

WIKI_JSON_SCHEMA = """You are a knowledge assistant. Turn the raw note below into a concise wiki-style entry.

Output rules:
- Respond with ONLY valid JSON (no markdown fences, no commentary before or after).
- Use this exact shape and key names:
  {
    "title": "short descriptive title",
    "summary": "2-4 sentence overview",
    "key_points": ["bullet-level fact or idea", "..."],
    "tags": ["short meaningful labels, 1-2 words each"],
    "related_topics": ["optional links to other wiki entries"]
  }
- Tags: use 3-8 items when substance allows; each tag lowercase, 1-2 words (use underscore instead of spaces), no duplicates, no near-duplicates; prefer stable vocabulary (e.g. product names, domains) over prose.
- related_topics: when the note clearly relates to an existing wiki page, use that page's EXACT title string from the provided list; otherwise use a short new topic label (no forced match).
- Do not include "source_notes", "created_at", "updated_at", or "confidence_score" (the server sets these).
- Arrays may be empty only if nothing reasonable applies; prefer at least one key_point when the note has substance.
- Keep strings factual and aligned with the note; do not invent sources or events not implied by the note."""

WIKI_USER_NOTE_PREFIX = "Raw note:\n---\n"
WIKI_USER_NOTE_SUFFIX = "\n---"


def build_wiki_generation_prompt(note: str, known_page_titles: list[str] | None = None) -> str:
    """Full prompt string passed to the LLM (instruction + note)."""
    body = note.strip()
    titles_block = ""
    if known_page_titles:
        lines = "\n".join(f"- {t}" for t in known_page_titles[:250])
        titles_block = (
            "\n\nExisting wiki page titles (prefer an EXACT copy from this list in "
            f'related_topics when the note clearly ties to that page):\n{lines}\n'
        )
    return f"{WIKI_JSON_SCHEMA}{titles_block}\n\n{WIKI_USER_NOTE_PREFIX}{body}{WIKI_USER_NOTE_SUFFIX}"
