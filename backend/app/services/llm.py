import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Iterator

from app.db.session import DATA_DIR
from app.services.llm_providers.base import LLMProvider, LLMResult
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

_llm_operation: ContextVar[str] = ContextVar("llm_operation", default="")
_llm_sink: ContextVar[list | None] = ContextVar("llm_sink", default=None)

LLM_USAGE_LOG = DATA_DIR / "llm_usage.jsonl"


@dataclass
class LLMCallRecord:
    """One instrumented LLM call -- used by evaluation cost/latency aggregation."""

    text: str
    latency_ms: float
    provider: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    tokens_estimated: bool
    estimated_cost_usd: float | None
    success: bool
    error: str | None = None
    operation: str = ""


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
            provider_name="openai",
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
        return OpenAIProvider(api_key=api_key, model=model, base_url=base_url, provider_name="openai_compatible")

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


def reset_llm_provider() -> None:
    """Drop the cached provider so the next call rebuilds from current env (e.g. model override)."""
    global _provider
    with _provider_lock:
        _provider = None


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


@contextmanager
def llm_operation(name: str) -> Iterator[None]:
    """Label LLM calls made inside this block (ingest, expand, audit, …)."""
    token = _llm_operation.set(name)
    try:
        yield
    finally:
        _llm_operation.reset(token)


@contextmanager
def capture_llm_calls() -> Iterator[list[LLMCallRecord]]:
    """Collect ``LLMCallRecord``s for the current task (used by the evaluation runner).

    Nested captures extend the parent sink so operation-level wrappers still roll up.
    """
    records: list[LLMCallRecord] = []
    parent = _llm_sink.get()
    token = _llm_sink.set(records)
    try:
        yield records
    finally:
        _llm_sink.reset(token)
        if parent is not None:
            parent.extend(records)


def _estimate_cost_usd(model: str, input_tokens: int | None, output_tokens: int | None) -> float | None:
    """Lazy import so production LLM calls don't hard-depend on the evaluation package
    being importable in every context. Returns None when the model has no priced entry."""
    try:
        from app.evaluation.cost import estimate_cost_usd
    except Exception:
        return None
    return estimate_cost_usd(model, input_tokens, output_tokens)


def _append_usage_log(record: LLMCallRecord) -> None:
    if os.environ.get("SYNAPSE_LOG_LLM_USAGE", "1").strip().lower() in ("0", "false", "no", "off"):
        return
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **{k: v for k, v in asdict(record).items() if k != "text"},
            "response_chars": len(record.text),
        }
        with LLM_USAGE_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning("Failed to append LLM usage log: %s", exc)


def _publish_record(record: LLMCallRecord) -> None:
    sink = _llm_sink.get()
    if sink is not None:
        sink.append(record)
    _append_usage_log(record)


async def call_llm_detailed(
    prompt: str,
    *,
    temperature: float | None = None,
    seed: int | None = None,
) -> LLMCallRecord:
    """Like ``call_llm`` but returns latency/usage. Evaluation prefers this entry point."""
    operation = _llm_operation.get()
    provider = _get_provider()
    full_prompt = _apply_settings(prompt)
    temp = 0.3 if temperature is None else temperature
    started = time.perf_counter()
    try:
        result: LLMResult = await provider.complete(full_prompt, temperature=temp, seed=seed)
        latency_ms = (time.perf_counter() - started) * 1000.0
        usage = result.usage
        record = LLMCallRecord(
            text=result.text,
            latency_ms=latency_ms,
            provider=usage.provider,
            model=usage.model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            tokens_estimated=usage.estimated,
            estimated_cost_usd=_estimate_cost_usd(usage.model, usage.input_tokens, usage.output_tokens),
            success=True,
            operation=operation,
        )
        _publish_record(record)
        logger.debug(
            "LLM ok provider=%s model=%s latency_ms=%.1f in=%s out=%s estimated=%s",
            record.provider,
            record.model,
            record.latency_ms,
            record.input_tokens,
            record.output_tokens,
            record.tokens_estimated,
        )
        return record
    except Exception as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        record = LLMCallRecord(
            text="",
            latency_ms=latency_ms,
            provider=getattr(provider, "provider_name", "unknown"),
            model=getattr(provider, "model", "unknown"),
            input_tokens=None,
            output_tokens=None,
            tokens_estimated=True,
            estimated_cost_usd=None,
            success=False,
            error=str(exc),
            operation=operation,
        )
        _publish_record(record)
        raise


async def call_llm(prompt: str, *, temperature: float | None = None, seed: int | None = None) -> str:
    """
    Send a single prompt string to the configured provider's chat model and return the
    assistant message text (empty string if the model returns no content). Every ingest
    /expand/audit/reshape/quiz-generation/ask call in the app goes through this one
    function -- swapping providers never requires touching any of those call sites, and
    neither does applying the configured persona/thinking settings (Phase 13).
    """
    record = await call_llm_detailed(prompt, temperature=temperature, seed=seed)
    return record.text
