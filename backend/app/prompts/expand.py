"""Reusable prompt for expand mode: deepen ONE existing topic with new sub-topics."""

EXPAND_JSON_SCHEMA = """You are a curriculum-design assistant. You are deepening ONE specific topic in an \
existing prerequisite graph by proposing new sub-topics beneath it -- things a learner needs to \
understand as components of, or prerequisites for, the anchor topic below.

Output rules:
- Respond with ONLY valid JSON (no markdown fences, no commentary before or after).
- Use this exact shape:
  {
    "topics": [
      {"title": "...", "summary": "1-3 sentence overview", "confidence": <number between 0 and 1>},
      ...
    ],
    "dependencies": [
      {"from": "<title that REQUIRES the other>", "to": "<title that is the PREREQUISITE>"},
      ...
    ]
  }
- Propose 2-5 NEW sub-topics only -- do not repropose the anchor topic itself or any of its
  already-listed existing prerequisites.
- Every new sub-topic must connect (directly or through another new sub-topic) back to the
  anchor topic, phrased as "<anchor topic title> requires <new sub-topic title>".
- Every "from"/"to" value must be either the exact anchor topic title given below, or an exact
  title from your own new "topics" list -- do not invent edges to topics outside this scope.
- confidence (0-1, per topic): 0.8+ well-established and clearly beneath the anchor, 0.4-0.8
  reasonable inference, below 0.4 speculative."""


def build_expand_prompt(
    anchor_title: str,
    anchor_summary: str,
    existing_prerequisite_titles: list[str],
    instructions: str | None,
) -> str:
    parts = [
        EXPAND_JSON_SCHEMA,
        f"Anchor topic: {anchor_title}",
        f"Anchor summary: {anchor_summary.strip() or '(no summary yet)'}",
    ]
    if existing_prerequisite_titles:
        lines = "\n".join(f"- {t}" for t in existing_prerequisite_titles)
        parts.append(f"Already has these direct prerequisites (do not repropose these):\n{lines}")
    if instructions and instructions.strip():
        parts.append(f"Additional guidance from the user: {instructions.strip()}")
    return "\n\n".join(parts)
