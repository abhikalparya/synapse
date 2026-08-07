from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Common interface every concrete provider implements -- ``call_llm`` in
    services/llm.py never touches a provider SDK directly, only this."""

    @abstractmethod
    async def complete(self, prompt: str) -> str:
        """Send a single prompt, return the model's text response (empty string if none)."""

    async def aclose(self) -> None:
        """Release any held connections. Default no-op; override where the SDK needs it."""
        return None
