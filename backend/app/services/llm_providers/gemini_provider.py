import logging

from google import genai
from google.genai.errors import APIError

from app.services.llm_providers.base import LLMProvider, LLMResult, LLMUsage, estimate_token_count

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    """Wraps Google's Gemini API via the google-genai SDK's async client."""

    def __init__(self, *, api_key: str, model: str) -> None:
        self._client = genai.Client(api_key=api_key)
        self.model = model
        self.provider_name = "gemini"
        self._model = model

    async def complete(self, prompt: str, *, temperature: float = 0.3, seed: int | None = None) -> LLMResult:
        try:
            response = await self._generate(prompt, temperature=temperature, seed=seed)
        except APIError as exc:
            raise RuntimeError(str(exc)) from exc
        text = (response.text or "").strip()
        meta = getattr(response, "usage_metadata", None)
        input_tokens = getattr(meta, "prompt_token_count", None) if meta is not None else None
        output_tokens = getattr(meta, "candidates_token_count", None) if meta is not None else None
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

    async def _generate(self, prompt: str, *, temperature: float, seed: int | None):
        """Prefer a temperature/seed config; fall back to the original unconfigured call."""
        try:
            from google.genai import types
        except ImportError:
            return await self._client.aio.models.generate_content(model=self._model, contents=prompt)

        config_kwargs: dict = {"temperature": temperature}
        if seed is not None:
            config_kwargs["seed"] = seed
        try:
            return await self._client.aio.models.generate_content(
                model=self._model,
                contents=prompt,
                config=types.GenerateContentConfig(**config_kwargs),
            )
        except TypeError:
            return await self._client.aio.models.generate_content(model=self._model, contents=prompt)

    async def aclose(self) -> None:
        # google-genai's Client.close() is synchronous (closes an underlying httpx client).
        self._client.close()
