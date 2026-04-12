import json
import logging
import os
import re
import threading

from openai import AsyncOpenAI
from pydantic import ValidationError

from app.models.query import QueryLlmAnswer

logger = logging.getLogger(__name__)

QUERY_WRITEBACK_JSON_SUFFIX = """

After answering, respond with ONLY valid JSON (no markdown fences, no commentary before or after) using exactly this shape:
{"answer": "<your answer to the user>", "confidence_score": <number between 0 and 1>}

confidence_score (your calibration, 0–1):
- High (0.8–1.0): the answer is strongly supported by the wiki excerpts above, or is stable general knowledge clearly applicable.
- Medium (0.4–0.8): partial support from excerpts, reasonable inference, or thin/missing wiki context.
- Low (0.0–0.4): speculation, excerpts contradict or omit the topic, or you are largely guessing.

Put your full user-facing answer text inside the "answer" string (plain text or light markdown is fine)."""

_client_lock = threading.Lock()
_async_client: AsyncOpenAI | None = None


def _strip_json_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _build_async_openai_client() -> AsyncOpenAI:
    raw = os.environ.get("OPENAI_API_KEY", "")
    api_key = raw.strip().strip('"').strip("'")
    if not api_key:
        msg = "OPENAI_API_KEY is not set"
        logger.error(msg)
        raise RuntimeError(msg)

    org = os.environ.get("OPENAI_ORGANIZATION") or os.environ.get("OPENAI_ORG_ID")
    project = os.environ.get("OPENAI_PROJECT")
    kwargs: dict = {"api_key": api_key}
    if org:
        kwargs["organization"] = org.strip()
    if project:
        kwargs["project"] = project.strip()
    return AsyncOpenAI(**kwargs)


def _get_async_client() -> AsyncOpenAI:
    global _async_client
    with _client_lock:
        if _async_client is None:
            _async_client = _build_async_openai_client()
        return _async_client


async def close_async_openai_client() -> None:
    """Release HTTP connections; call from app lifespan shutdown."""
    global _async_client
    client: AsyncOpenAI | None = None
    with _client_lock:
        if _async_client is not None:
            client = _async_client
            _async_client = None
    if client is not None:
        await client.close()


async def call_llm(prompt: str) -> str:
    """
    Send a single prompt string to the configured chat model and return the
    assistant message text (empty string if the model returns no content).
    """
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip()
    client = _get_async_client()
    logger.debug("LLM request model=%s prompt_chars=%s", model, len(prompt))

    completion = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    choice = completion.choices[0].message
    text = (choice.content or "").strip()
    logger.debug("LLM response_chars=%s", len(text))
    return text


async def call_llm_query_answer_with_confidence(prefix_prompt: str) -> tuple[str, float]:
    """
    Run the query RAG prompt plus confidence JSON contract; return (answer, confidence).
    """
    full_prompt = prefix_prompt.strip() + QUERY_WRITEBACK_JSON_SUFFIX
    raw = await call_llm(full_prompt)
    cleaned = _strip_json_fences(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.warning("Query LLM returned non-JSON; treating as low confidence: %s", exc)
        return raw.strip() or "", 0.0
    try:
        payload = QueryLlmAnswer.model_validate(data)
    except ValidationError as exc:
        logger.warning("Query LLM JSON invalid; treating as low confidence: %s", exc)
        return (str(data.get("answer", "")).strip() if isinstance(data, dict) else "") or "", 0.0
    return (payload.answer or "").strip(), float(payload.confidence_score)
