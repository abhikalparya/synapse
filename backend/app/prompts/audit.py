"""Reusable prompt for the LLM-judgment half of audit mode (structural checks are pure
Python -- see services/audit.py)."""

AUDIT_JSON_SCHEMA = """You are auditing an existing prerequisite graph for structural/semantic issues. \
Given the topics (with summaries) and their dependency edges below, identify:

1. "missing_prerequisite": a topic whose summary implies it depends on background knowledge that
   is NOT captured as an explicit prerequisite edge in this graph.
2. "cycle_risk": a pair of topics whose prerequisite relationship looks ambiguous, possibly
   reversed, or otherwise risky -- e.g. two topics that seem to each require the other, or where
   the direction of an existing (or missing) edge between them is unclear.

Output rules:
- Respond with ONLY valid JSON (no markdown fences, no commentary before or after).
- Use this exact shape:
  {
    "findings": [
      {"type": "missing_prerequisite" | "cycle_risk", "topic_titles": ["..."], "detail": "..."},
      ...
    ]
  }
- "topic_titles" must be exact titles from the list given below (one topic for
  missing_prerequisite, usually two for cycle_risk).
- Only report genuine, material issues -- an empty "findings" list is a valid, good outcome if
  the graph looks sound. Do not pad the list with trivial or speculative findings."""


def build_audit_prompt(topics: list[dict[str, str]], edges: list[tuple[str, str]]) -> str:
    topic_lines = "\n".join(f"- {t['title']}: {t['summary'] or '(no summary)'}" for t in topics)
    parts = [AUDIT_JSON_SCHEMA, f"Topics:\n{topic_lines}"]
    if edges:
        edge_lines = "\n".join(f"- {a} requires {b}" for a, b in edges)
        parts.append(f"Existing dependency edges:\n{edge_lines}")
    else:
        parts.append("There are currently no dependency edges in this graph.")
    return "\n\n".join(parts)
