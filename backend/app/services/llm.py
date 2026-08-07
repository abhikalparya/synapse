import logging
import os
import threading

from app.services.llm_providers.base import LLMProvider
from app.services.llm_providers.gemini_provider import GeminiProvider
from app.services.llm_providers.openai_provider import OpenAIProvider
from app.services.settings import load_settings

_EXTENDED_THINKING_INSTRUCTION = (
    "Before giving your final answer, think through this step by step -- consider edge "
    "cases and alternatives rather than responding with your first instinct. Still follow "
    "every output-format rule above exactly."
)

logger = logging.getLogger(__name__)

_provider_lock = threading.Lock()
_provider: LLMProvider | None = None


def _clean(raw: str | None) -> str:
    return (raw or "").strip().strip('"').strip("'")


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


def _apply_settings(prompt: str) -> str:
    """Append the configured persona and/or extended-thinking nudge to every prompt.
    Appended rather than prepended so each call site's own output-format instructions
    (e.g. ingest/audit's "respond with ONLY valid JSON") stay first and freshest in
    context -- persona/thinking are a styling/deliberation layer on top, not a
    replacement for the structural contract."""
    settings = load_settings()
    suffix_parts: list[str] = []
    persona = (settings.get("persona") or "").strip()
    if persona:
        suffix_parts.append(persona)
    if settings.get("thinking_level") == "extended":
        suffix_parts.append(_EXTENDED_THINKING_INSTRUCTION)
    if not suffix_parts:
        return prompt
    return prompt + "\n\n" + "\n\n".join(suffix_parts)


async def call_llm(prompt: str) -> str:
    """
    Send a single prompt string to the configured provider's chat model and return the
    assistant message text (empty string if the model returns no content). Every ingest
    /expand/audit/reshape/quiz-generation/ask call in the app goes through this one
    function -- swapping providers never requires touching any of those call sites, and
    neither does applying the configured persona/thinking settings (Phase 13).
    """
    provider = _get_provider()
    full_prompt = _apply_settings(prompt)
    logger.debug("LLM request provider=%s prompt_chars=%s", type(provider).__name__, len(full_prompt))
    text = await provider.complete(full_prompt)
    logger.debug("LLM response_chars=%s", len(text))
    return text
