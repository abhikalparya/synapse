from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMUsage:
    """Token accounting for one completion. ``estimated`` is True when the provider did
    not return usage and the counts were derived from character length."""

    input_tokens: int | None
    output_tokens: int | None
    model: str
    provider: str
    estimated: bool = False


@dataclass
class LLMResult:
    """Normalized provider response. ``call_llm`` still returns ``text`` for existing
    call sites; evaluation and cost tracking use the rest of this record."""

    text: str
    usage: LLMUsage


class LLMProvider(ABC):
    """Common interface every concrete provider implements -- ``call_llm`` in
    services/llm.py never touches a provider SDK directly, only this."""

    model: str
    provider_name: str

    @abstractmethod
    async def complete(self, prompt: str, *, temperature: float = 0.3, seed: int | None = None) -> LLMResult:
        """Send a single prompt, return the model's text plus usage (estimated if needed)."""

    async def aclose(self) -> None:
        """Release any held connections. Default no-op; override where the SDK needs it."""
        return None


def estimate_token_count(text: str) -> int:
    """Rough fallback: ~4 characters per token. Always labeled ``estimated`` at the call site."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)
