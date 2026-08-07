from typing import Literal

from pydantic import BaseModel, Field

ThinkingLevel = Literal["standard", "extended"]


class Settings(BaseModel):
    """Local, single-workspace settings for LLM behavior -- there is exactly one row of
    this, not a per-user table (Synapse has no multi-tenant concept)."""

    persona: str = Field(default="", max_length=2000, description="System-prompt prefix injected into every LLM call")
    memory_enabled: bool = Field(
        default=True,
        description="Whether prior Q&A turns for a topic are carried into new /ask calls as context",
    )
    thinking_level: ThinkingLevel = Field(
        default="standard",
        description="'extended' asks the model to reason step-by-step before answering, on every LLM call",
    )


class SettingsUpdate(BaseModel):
    persona: str | None = Field(default=None, max_length=2000)
    memory_enabled: bool | None = None
    thinking_level: ThinkingLevel | None = None
