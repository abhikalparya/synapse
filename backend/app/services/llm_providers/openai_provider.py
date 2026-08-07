import logging

from openai import APIError, AsyncOpenAI

from app.services.llm_providers.base import LLMProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """Wraps the OpenAI chat-completions API. Also serves the generic
    OpenAI-compatible-endpoint option (self-hosted or third-party APIs that speak the
    same wire format) -- that's just this same class with ``base_url`` set."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str | None = None,
        organization: str | None = None,
        project: str | None = None,
    ) -> None:
        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        if organization:
            kwargs["organization"] = organization
        if project:
            kwargs["project"] = project
        self._client = AsyncOpenAI(**kwargs)
        self._model = model

    async def complete(self, prompt: str) -> str:
        try:
            completion = await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
        except APIError as exc:
            # Callers (routes/services) only need to know "the LLM call failed" -- they
            # shouldn't have to import an openai-specific exception type just to catch it.
            raise RuntimeError(str(exc)) from exc
        choice = completion.choices[0].message
        return (choice.content or "").strip()

    async def aclose(self) -> None:
        await self._client.close()
