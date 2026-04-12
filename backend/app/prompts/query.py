"""Prompts for RAG-style answers and self-updating wiki pages."""

import json

QUERY_ANSWER_INSTRUCTIONS = """You are Synapse, a concise knowledge assistant.

Use the provided wiki excerpts to answer the user's question. If excerpts are thin or empty, answer from general knowledge but say what is uncertain.

Rules:
- Be direct and accurate; prefer short paragraphs or bullets when helpful.
- Do not claim the wiki contains information it does not; cite themes from excerpts implicitly (no fake citations)."""


def build_query_answer_prompt(query: str, wiki_context: str) -> str:
    ctx = wiki_context.strip() or "(no matching wiki pages loaded)"
    return (
        f"{QUERY_ANSWER_INSTRUCTIONS}\n\n"
        f"Wiki excerpts:\n---\n{ctx}\n---\n\n"
        f"User question:\n{query.strip()}"
    )


WIKI_MERGE_SUMMARY_KEYS = """You refine an existing wiki entry after a new Q&A.

Output rules:
- Respond with ONLY valid JSON (no markdown fences, no commentary).
- Shape exactly:
  {"summary": "2-5 sentence overview merging prior facts with the new answer", "key_points": ["distinct bullets", "..."]}
- Preserve factual accuracy; fold in the new answer without inventing sources.
- key_points: 3-8 items when substance allows; otherwise fewer."""

MERGE_USER_TEMPLATE = """Existing wiki page JSON:
{existing_json}

User question:
{query}

New answer to integrate:
{answer}
"""


def build_wiki_merge_prompt(existing_page: dict, query: str, answer: str) -> str:
    body = json.dumps(existing_page, indent=2, ensure_ascii=False)
    return (
        f"{WIKI_MERGE_SUMMARY_KEYS}\n\n"
        + MERGE_USER_TEMPLATE.format(existing_json=body, query=query.strip(), answer=answer.strip())
    )


WIKI_FROM_QA_SCHEMA = """You create a new wiki-style entry from a user question and the assistant's answer.

Output rules:
- Respond with ONLY valid JSON (no markdown fences, no commentary).
- Shape exactly:
  {
    "title": "short descriptive title derived from the topic",
    "summary": "2-4 sentences",
    "key_points": ["bullet facts", "..."],
    "tags": ["short meaningful labels, 1-2 words each"],
    "related_topics": ["optional links to other wiki entries"]
  }
- Tags: 3-8 when substance allows; each lowercase, 1-2 words (underscores not spaces), deduplicated, consistent vocabulary.
- related_topics: when another wiki page clearly applies, use its EXACT title from the provided list; otherwise a short label.
- Do not include "source_notes", "created_at", "updated_at", or "confidence_score" (the server sets these).
- Ground content in the provided question and answer only; do not invent events."""

QA_USER_TEMPLATE = """User question:
{query}

Assistant answer:
{answer}
"""


def build_wiki_from_qa_prompt(
    query: str,
    answer: str,
    known_page_titles: list[str] | None = None,
) -> str:
    titles_block = ""
    if known_page_titles:
        lines = "\n".join(f"- {t}" for t in known_page_titles[:250])
        titles_block = (
            "\n\nExisting wiki page titles (prefer an EXACT copy in related_topics when relevant):\n"
            f"{lines}\n"
        )
    return (
        f"{WIKI_FROM_QA_SCHEMA}{titles_block}\n\n"
        + QA_USER_TEMPLATE.format(query=query.strip(), answer=answer.strip())
    )
