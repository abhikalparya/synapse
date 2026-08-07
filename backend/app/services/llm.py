import json
import logging
import os
import re
import threading

from pydantic import ValidationError

from app.models.query import QueryLlmAnswer
from app.services.llm_providers.base import LLMProvider
from app.services.llm_providers.gemini_provider import GeminiProvider
from app.services.llm_providers.openai_provider import OpenAIProvider

logger = logging.getLogger(__name__)

QUERY_WRITEBACK_JSON_SUFFIX = """

After answering, respond with ONLY valid JSON (no markdown fences, no commentary before or after) using exactly this shape:
{"answer": "<your answer to the user>", "confidence_score": <number between 0 and 1>}

confidence_score (your calibration, 0–1):
- High (0.8–1.0): the answer is strongly supported by the wiki excerpts above, or is stable general knowledge clearly applicable.
- Medium (0.4–0.8): partial support from excerpts, reasonable inference, or thin/missing wiki context.
- Low (0.0–0.4): speculation, excerpts contradict or omit the topic, or you are largely guessing.

Put your full user-facing answer text inside the "answer" string (plain text or light markdown is fine)."""

_provider_lock = threading.Lock()
_provider: LLMProvider | None = None


def _clean(raw: str | None) -> str:
    return (raw or "").strip().strip('"').strip("'")


def _strip_json_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _build_provider() -> LLMProvider:
    """Construct the configured provider from env config. ``LLM_PROVIDER`` selects which
    one (default "openai" -- so a bare OPENAI_API_KEY, the pre-Phase-12 setup, keeps
    working with zero changes). Each branch raises a clear RuntimeError naming the
    specific env var it's missing, rather than failing deep inside an SDK call."""
    provider_name = (_clean(os.environ.get("LLM_PROVIDER")) or "openai").lower()

    if provider_name == "openai":
        api_key = _clean(os.environ.get("OPENAI_API_KEY"))
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        return OpenAIProvider(
            api_key=api_key,
            model=_clean(os.environ.get("OPENAI_MODEL")) or "gpt-4o-mini",
            organization=_clean(os.environ.get("OPENAI_ORGANIZATION") or os.environ.get("OPENAI_ORG_ID")) or None,
            project=_clean(os.environ.get("OPENAI_PROJECT")) or None,
        )

    if provider_name == "gemini":
        api_key = _clean(os.environ.get("GEMINI_API_KEY"))
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        return GeminiProvider(api_key=api_key, model=_clean(os.environ.get("GEMINI_MODEL")) or "gemini-2.0-flash")

    if provider_name == "openai_compatible":
        base_url = _clean(os.environ.get("OPENAI_COMPATIBLE_BASE_URL"))
        model = _clean(os.environ.get("OPENAI_COMPATIBLE_MODEL"))
        if not base_url:
            raise RuntimeError("OPENAI_COMPATIBLE_BASE_URL is not set")
        if not model:
            raise RuntimeError("OPENAI_COMPATIBLE_MODEL is not set")
        # Self-hosted/local endpoints routinely don't check the key at all; default to a
        # placeholder rather than forcing OPENAI_COMPATIBLE_API_KEY on every setup.
        api_key = _clean(os.environ.get("OPENAI_COMPATIBLE_API_KEY")) or "not-required"
        return OpenAIProvider(api_key=api_key, model=model, base_url=base_url)

    raise RuntimeError(f"Unknown LLM_PROVIDER {provider_name!r} -- expected 'openai', 'gemini', or 'openai_compatible'")


def _get_provider() -> LLMProvider:
    global _provider
    with _provider_lock:
        if _provider is None:
            _provider = _build_provider()
            logger.info("LLM provider initialized: %s", type(_provider).__name__)
        return _provider


async def close_llm_provider() -> None:
    """Release the active provider's connections; call from app lifespan shutdown."""
    global _provider
    provider: LLMProvider | None = None
    with _provider_lock:
        if _provider is not None:
            provider = _provider
            _provider = None
    if provider is not None:
        await provider.aclose()


async def call_llm(prompt: str) -> str:
    """
    Send a single prompt string to the configured provider's chat model and return the
    assistant message text (empty string if the model returns no content). Every ingest
    /expand/audit/reshape/quiz-generation/ask call in the app goes through this one
    function -- swapping providers never requires touching any of those call sites.
    """
    provider = _get_provider()
    logger.debug("LLM request provider=%s prompt_chars=%s", type(provider).__name__, len(prompt))
    text = await provider.complete(prompt)
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
