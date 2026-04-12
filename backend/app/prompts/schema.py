"""Prompts to repair wiki pages that violate the knowledge schema."""

import json


WIKI_SCHEMA_REPAIR = """You fix a wiki page so it matches the knowledge schema.

Output rules:
- Respond with ONLY valid JSON (no markdown fences, no commentary).
- Shape exactly:
  {
    "title": "concise string",
    "summary": "string",
    "key_points": ["each item a plain string fact", "..."],
    "tags": ["lowercase_snake_tokens_only", "..."],
    "related_topics": ["string topic names", "..."]
  }
- title: short and specific (max about 200 characters).
- tags: lowercase, use underscores instead of spaces, 3-8 tags when possible, no uppercase.
- key_points: only non-empty strings; 3-8 bullets when substance allows.
- related_topics: only strings; use exact wiki titles from the provided list when clearly relevant.
- Preserve factual meaning from the input page; do not invent unrelated content."""


def build_wiki_schema_repair_prompt(
    page: dict,
    violations: list[str],
    *,
    known_page_titles: list[str] | None = None,
) -> str:
    issues = "\n".join(f"- {v}" for v in violations) if violations else "- (unspecified schema issues)"
    body = json.dumps(page, indent=2, ensure_ascii=False)
    titles_block = ""
    if known_page_titles:
        lines = "\n".join(f"- {t}" for t in known_page_titles[:250])
        titles_block = (
            "\n\nExisting wiki page titles (use an EXACT copy in related_topics when relevant):\n"
            f"{lines}\n"
        )
    return f"{WIKI_SCHEMA_REPAIR}{titles_block}\n\nViolations to fix:\n{issues}\n\nCurrent page JSON:\n{body}"
