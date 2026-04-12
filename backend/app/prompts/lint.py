"""Prompt for optional lint fix suggestions."""

import json


LINT_SUGGEST = """You are a knowledge-base editor. Given lint issues for wiki pages, propose one short, actionable fix per issue (imperative voice, one or two sentences).

Output rules:
- Respond with ONLY valid JSON (no markdown fences, no commentary).
- Shape exactly:
  {"suggestions": [{"type": "<same as input>", "page": "<title or null>", "pages": [<strings or null>], "suggestion": "<text>"}]}
- Match each input issue by ``type`` and ``page`` or ``pages`` (same strings as provided).
- If you cannot suggest a safe fix, use a neutral suggestion like "Review manually against schema."
- Do not invent page titles not listed in the issues."""


def build_lint_suggestions_prompt(issues: list[dict]) -> str:
    body = json.dumps(issues, indent=2, ensure_ascii=False)
    return f"{LINT_SUGGEST}\n\nLint issues:\n{body}"
