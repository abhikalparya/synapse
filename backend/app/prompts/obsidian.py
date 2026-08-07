"""Prompt for turning a parsed Obsidian vault into a topic + prerequisite-dependency DAG
(Phase 11 import). Reuses the same ``{"topics": [...], "dependencies": [...]}`` JSON
contract as prompts/ingest.py so the result flows through the same
proposal_common.build_topics_and_dependencies builder and the Phase 8 ingest review path
-- but the instructions differ from a plain ingest goal in one important way: a vault
already comes pre-chunked into notes, so each note is a natural topic candidate rather
than something to compress into 3-8 topics, and the vault's own [[wikilinks]] are a
strong (if untyped and undirected) hint for dependency edges that a free-text goal never
has."""

OBSIDIAN_IMPORT_JSON_SCHEMA = """You are importing an Obsidian vault into a prerequisite \
dependency graph. Below is every note in the vault, in the form:

### <note title>
<note body>
Links: <titles this note [[wikilinks]] to, if any>

Your job:
1. Decide which notes represent genuine, distinct topics worth tracking (skip notes that \
are pure indexes/table-of-contents/journal entries with no real conceptual content).
2. For each kept note, where it maps 1:1 to a topic, use that note's EXACT title as the \
topic title (case-sensitive) -- this lets the topic be traced back to its source note. \
Only invent a different title if you're merging multiple notes into one topic.
3. For each [[wikilink]] between two kept notes, decide whether it represents a genuine \
prerequisite relationship (the linking note assumes/builds on the linked note) or just a \
loose, non-prerequisite association -- wikilinks are undirected and untyped in Obsidian, \
so this judgment call is the main value you add. Only emit a dependency for genuine \
prerequisites; drop the rest.

Output rules:
- Respond with ONLY valid JSON (no markdown fences, no commentary before or after).
- Use this exact shape and key names:
  {
    "topics": [
      {"title": "...", "summary": "1-3 sentence overview", "confidence": <0-1>},
      ...
    ],
    "dependencies": [
      {"from": "<topic title that REQUIRES the other>", "to": "<topic title that is the PREREQUISITE>"},
      ...
    ]
  }
- Every "from"/"to" value MUST exactly match a "title" in "topics" (case-sensitive).
- The dependencies must form a DAG: no topic may (transitively) depend on itself.
- Unlike a from-scratch goal, do NOT artificially cap the topic count -- one topic per
  substantive note is expected and normal for a vault.
- confidence (0-1): how well-supported this note is as a standalone topic with a clear
  place in the sequence, not how important the underlying subject is in general."""


def build_obsidian_import_prompt(
    notes: list[tuple[str, str, list[str]]],
    known_topic_titles: list[str] | None = None,
) -> str:
    """``notes`` is a list of (title, body, links) tuples, already loaded from the vault."""
    blocks = []
    for title, body, links in notes:
        block = f"### {title}\n{body}"
        if links:
            block += f"\nLinks: {', '.join(links)}"
        blocks.append(block)
    vault_block = "\n\n".join(blocks)

    titles_block = ""
    if known_topic_titles:
        lines = "\n".join(f"- {t}" for t in known_topic_titles[:250])
        titles_block = f"\n\nExisting topic titles (avoid exact duplicates unless truly the same topic):\n{lines}\n"

    return f"{OBSIDIAN_IMPORT_JSON_SCHEMA}{titles_block}\n\nVault notes:\n---\n{vault_block}\n---"
