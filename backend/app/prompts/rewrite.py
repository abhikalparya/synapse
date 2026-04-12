"""Prompts for full-page knowledge rewriting (clarity, redundancy, structure)."""

import json

KNOWLEDGE_REWRITE = """You are an editor improving a wiki knowledge page for clarity and readability.

Goals:
- Rewrite the **summary** for clear flow; remove redundancy with key points where it repeats the same facts.
- Tighten **key_points**: dedupe near-duplicates; keep every distinct fact; prefer parallel phrasing.
- Improve **structure** (ordering: broad → specific when natural).
- Keep **all factual claims** and technical meaning; do not invent sources or new facts.
- **tags**: concise lowercase tokens with underscores (merge synonyms).
- **related_topics**: only other wiki topics from the provided title list when still relevant; otherwise short labels.

Output rules:
- Respond with ONLY valid JSON (no markdown fences, no commentary).
- Shape exactly:
  {
    "title": "string (keep or slightly clarify; do not change topic)",
    "summary": "2-6 sentences",
    "key_points": ["...", "..."],
    "tags": ["...", "..."],
    "related_topics": ["...", "..."]
  }
- Do not include source_notes, merged_from, created_at, updated_at, or confidence_score."""


def build_knowledge_rewrite_prompt(page: dict, *, known_page_titles: list[str]) -> str:
    titles_block = ""
    if known_page_titles:
        lines = "\n".join(f"- {t}" for t in known_page_titles[:250])
        titles_block = (
            "\n\nExisting wiki page titles (use EXACT strings in related_topics when applicable):\n"
            f"{lines}\n"
        )
    body = json.dumps(page, indent=2, ensure_ascii=False)
    return f"{KNOWLEDGE_REWRITE}{titles_block}\n\nPage to improve:\n{body}"
