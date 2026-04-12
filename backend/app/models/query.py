from typing import Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=16_000)


class QueryLlmAnswer(BaseModel):
    """Structured first-step LLM output for POST /query."""

    answer: str = Field(default="")
    confidence_score: float = Field(ge=0.0, le=1.0)


class QueryResponse(BaseModel):
    answer: str
    used_nodes: list[str] = Field(
        default_factory=list,
        description="Wiki page titles whose content was included in the answer context",
    )
    updated_node: str | None = Field(
        default=None,
        description="Title of the wiki page written by this query; null if skipped or unchanged",
    )
    confidence_score: float = Field(ge=0.0, le=1.0)
    wiki_action: Literal["updated", "created", "skipped"]
    wiki_file: str | None = None
