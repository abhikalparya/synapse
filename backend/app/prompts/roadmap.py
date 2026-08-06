"""Reusable prompts for generating a topic + prerequisite-dependency DAG."""

ROADMAP_JSON_SCHEMA = """You are a curriculum-design assistant. Given the goal/content below, break it into a \
small set of discrete topics and the prerequisite relationships between them.

Output rules:
- Respond with ONLY valid JSON (no markdown fences, no commentary before or after).
- Use this exact shape and key names:
  {
    "topics": [
      {
        "title": "short topic name",
        "summary": "1-3 sentence overview of this topic",
        "confidence": <number between 0 and 1>
      },
      ...
    ],
    "dependencies": [
      {"from": "<topic title that REQUIRES the other>", "to": "<topic title that is the PREREQUISITE>"},
      ...
    ]
  }
- A dependency entry means "from" cannot be understood without "to" first; "to" is the prerequisite.
- Every "from" and "to" value MUST exactly match a "title" from the "topics" list (case-sensitive).
- The dependencies must form a DAG: no topic may (transitively) depend on itself.
- Use 3-8 topics for a focused goal; more only if the goal is genuinely broad.
- List topics roughly foundational-first, but the "dependencies" edges (not list order) define the real structure.
- Titles should be short, specific, and non-overlapping; summaries factual, with no invented specifics beyond
  what the goal/content implies.
- confidence (your calibration, 0-1, per topic):
  High (0.8-1.0): this topic and its role in the sequence is well-established, standard knowledge.
  Medium (0.4-0.8): reasonable inference from the goal/content, or a topic whose exact placement is debatable.
  Low (0.0-0.4): speculative, a stretch from the given goal/content, or you are largely guessing it belongs."""


def build_roadmap_generation_prompt(source_text: str, known_topic_titles: list[str] | None = None) -> str:
    """Full prompt string passed to the LLM (instruction + goal/content, minus already-known titles)."""
    body = source_text.strip()
    titles_block = ""
    if known_topic_titles:
        lines = "\n".join(f"- {t}" for t in known_topic_titles[:250])
        titles_block = f"\n\nExisting topic titles (avoid exact duplicates unless truly the same topic):\n{lines}\n"
    return f"{ROADMAP_JSON_SCHEMA}{titles_block}\n\nGoal / content:\n---\n{body}\n---"
