"""Prompts for periodic wiki knowledge-base refactoring."""

import json


MERGE_DUPLICATE_PAGES = """You consolidate multiple wiki pages that cover the same or nearly the same topic.

Output rules:
- Respond with ONLY valid JSON (no markdown fences, no commentary).
- Shape exactly:
  {
    "title": "one clear canonical title",
    "summary": "2-5 sentences merging facts without contradictions",
    "key_points": ["deduplicated bullets from all pages", "..."],
    "tags": ["3-8 short lowercase tokens", "underscores not spaces"],
    "related_topics": ["other wiki topics still relevant"]
  }
- Combine **every** input summary into one coherent overview: weave facts from each page; do not drop unique facts.
- **key_points**: union all important bullets from all pages; dedupe near-duplicates but keep distinct facts.
- **tags**: merge tag sets from all pages; normalize to short lowercase underscore tokens; dedupe.
- Merge overlapping facts; resolve minor conflicts by favoring the most specific, recent-sounding detail.
- related_topics: only names of topics that should remain as separate wiki pages; do not list titles of pages being merged away.
- Do not include "source_notes", "created_at", "updated_at", "confidence_score", or "merged_from" (the server sets these)."""


def build_merge_duplicate_pages_prompt(pages_json: list[dict]) -> str:
    body = json.dumps(pages_json, indent=2, ensure_ascii=False)
    return f"{MERGE_DUPLICATE_PAGES}\n\nPages to merge into one entry:\n{body}"
