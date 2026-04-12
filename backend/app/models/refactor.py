from pydantic import BaseModel, Field


class RefactorResponse(BaseModel):
    merged_groups: int = Field(ge=0, description="Number of duplicate clusters merged")
    pages_merged: int = Field(
        ge=0,
        description="Duplicate files absorbed (files deleted after merge)",
    )
    pages_updated: int = Field(
        ge=0,
        description="Pages persisted after a refactor-time LLM pass (quality rewrite and/or stale batch)",
    )
    pages_rewritten: int = Field(
        ge=0,
        description="Pages fully rewritten via LLM (refactor conditions, post-merge, and optional stale batch)",
    )
    errors: list[str] = Field(default_factory=list, description="Non-fatal issues during refactor")
