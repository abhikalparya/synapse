"""Reusable prompt for reshape mode: restructure a SELECTED subgraph (split/merge/reorder)."""

RESHAPE_JSON_SCHEMA = """You are a curriculum-design assistant. You are restructuring a SELECTED subgraph \
of an existing prerequisite graph -- splitting an overloaded topic, merging near-duplicate topics, \
and/or reordering dependency edges within the selection.

Output rules:
- Respond with ONLY valid JSON (no markdown fences, no commentary before or after).
- Use this exact shape (every key optional -- omit or leave empty if unused):
  {
    "new_topics": [{"title": "...", "summary": "...", "confidence": <number between 0 and 1>}, ...],
    "new_dependencies": [{"from": "...", "to": "..."}, ...],
    "removed_dependencies": [{"from": "...", "to": "...", "reason": "..."}, ...],
    "merges": [{"source": "...", "target": "...", "reason": "..."}, ...],
    "edits": [{"topic": "...", "new_summary": "...", "reason": "..."}, ...]
  }
- "new_topics": brand-new topics (e.g. pieces of a split). Titles must not duplicate any
  existing title given below.
- "new_dependencies" / "removed_dependencies": "from"/"to" must be exact titles from the
  selected topics below, or (for new_dependencies only) an exact title from your own
  "new_topics" list. Never reference a topic outside what's given below.
- "merges": collapses "source" into "target" (source is deleted; its edges and resources
  move onto target) -- both must be exact titles from the selected topics below, never a
  new topic.
- "edits": replaces a selected topic's summary text only (e.g. narrowing its scope after a
  split) -- "topic" must be an exact title from the selected topics below.
- Only propose operations that materially improve the structure; an empty list for any key
  is fine if no change of that kind is warranted for this selection.
- confidence (0-1, for each new topic): 0.8+ well-established, 0.4-0.8 reasonable inference,
  below 0.4 speculative."""


def build_reshape_prompt(
    topics: list[dict[str, str]],
    internal_edges: list[tuple[str, str]],
    boundary_edges: list[tuple[str, str]],
    instructions: str | None,
) -> str:
    parts = [RESHAPE_JSON_SCHEMA]

    topic_lines = "\n".join(f"- {t['title']}: {t['summary'] or '(no summary)'}" for t in topics)
    parts.append(f"Selected topics (the only ones you may reference by title above):\n{topic_lines}")

    if internal_edges:
        lines = "\n".join(f"- {a} requires {b}" for a, b in internal_edges)
        parts.append(f"Existing dependency edges WITHIN the selection:\n{lines}")

    if boundary_edges:
        lines = "\n".join(f"- {a} requires {b}" for a, b in boundary_edges)
        parts.append(
            "Existing edges connecting the selection to topics OUTSIDE it (read-only context "
            f"-- you cannot reference these outside titles in your output):\n{lines}",
        )

    if instructions and instructions.strip():
        parts.append(f"Requested restructuring: {instructions.strip()}")
    else:
        parts.append(
            "No specific instructions given -- use your judgment on what restructuring (if any) "
            "would materially improve this subgraph.",
        )

    return "\n\n".join(parts)
