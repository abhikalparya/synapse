import logging

from openai import APIError, AsyncOpenAI

from app.services.llm_providers.base import LLMProvider, LLMResult, LLMUsage, estimate_token_count

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
        provider_name: str = "openai",
    ) -> None:
        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        if organization:
            kwargs["organization"] = organization
        if project:
            kwargs["project"] = project
        self._client = AsyncOpenAI(**kwargs)
        self.model = model
        self.provider_name = provider_name
        self._model = model

    async def complete(self, prompt: str, *, temperature: float = 0.3, seed: int | None = None) -> LLMResult:
        create_kwargs: dict = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if seed is not None:
            create_kwargs["seed"] = seed
        try:
            completion = await self._client.chat.completions.create(**create_kwargs)
        except APIError as exc:
            # Callers (routes/services) only need to know "the LLM call failed" -- they
            # shouldn't have to import an openai-specific exception type just to catch it.
            raise RuntimeError(str(exc)) from exc
        choice = completion.choices[0].message
        text = (choice.content or "").strip()
        usage = getattr(completion, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
        output_tokens = getattr(usage, "completion_tokens", None) if usage is not None else None
        estimated = input_tokens is None or output_tokens is None
        if input_tokens is None:
            input_tokens = estimate_token_count(prompt)
        if output_tokens is None:
            output_tokens = estimate_token_count(text)
        return LLMResult(
            text=text,
            usage=LLMUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=self.model,
                provider=self.provider_name,
                estimated=estimated,
            ),
        )

    async def aclose(self) -> None:
        await self._client.close()
