import logging

from google import genai
from google.genai.errors import APIError

from app.services.llm_providers.base import LLMProvider

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    """Wraps Google's Gemini API via the google-genai SDK's async client."""

    def __init__(self, *, api_key: str, model: str) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model

    async def complete(self, prompt: str) -> str:
        try:
            response = await self._client.aio.models.generate_content(model=self._model, contents=prompt)
        except APIError as exc:
            # Same rationale as OpenAIProvider: normalize to RuntimeError so callers
            # never need to know which SDK is behind the active provider.
            raise RuntimeError(str(exc)) from exc
        return (response.text or "").strip()

    async def aclose(self) -> None:
        # google-genai's Client.close() is synchronous (closes an underlying httpx client).
        self._client.close()
