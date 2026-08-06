"""Reusable prompt for generating a closure quiz from a topic's summary and resources."""

QUIZ_JSON_SCHEMA = """You are a quiz-writing assistant. Given the topic summary and any attached resource \
excerpts below, write a short closure quiz to check whether someone has actually understood this topic.

Output rules:
- Respond with ONLY valid JSON (no markdown fences, no commentary before or after).
- Use this exact shape:
  {
    "questions": [
      {"question": "...", "choices": ["...", "...", "...", "..."], "correct_index": <0-3>},
      ...
    ]
  }
- Write 3-5 multiple-choice questions, each with exactly 4 choices and exactly one correct answer.
- correct_index is the 0-based index of the correct choice within that question's "choices" array.
- Questions should test understanding of the topic's core ideas, not trivia or wording recall; avoid
  ambiguous or trick phrasing.
- Base questions only on the summary/resources given below; do not require outside knowledge beyond
  general familiarity with the topic."""


def build_quiz_prompt(title: str, summary: str, resource_texts: list[str]) -> str:
    parts = [f"Topic: {title}"]
    if summary.strip():
        parts.append(f"Summary: {summary.strip()}")
    for i, text in enumerate(resource_texts[:5], start=1):
        excerpt = text.strip()[:2000]
        if excerpt:
            parts.append(f"Resource {i} excerpt:\n{excerpt}")
    body = "\n\n".join(parts)
    return f"{QUIZ_JSON_SCHEMA}\n\n{body}"
