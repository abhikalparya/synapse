from datetime import datetime

from pydantic import BaseModel, Field


class WikiPage(BaseModel):
    title: str = Field(default="", max_length=500)
    summary: str = Field(default="")
    key_points: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    related_topics: list[str] = Field(default_factory=list)
    source_notes: list[str] = Field(
        default_factory=list,
        description="Provenance: raw note filenames and/or query:<text> markers",
    )
    merged_from: list[str] = Field(
        default_factory=list,
        description="Titles (and prior merged_from) absorbed when this page was consolidated",
    )
    created_at: datetime | None = Field(
        default=None,
        description="UTC creation time; omitted on legacy pages until backfilled on write",
    )
    updated_at: datetime | None = Field(
        default=None,
        description="UTC last update time",
    )
    confidence_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Model-estimated reliability of this page's content (query write-back)",
    )


class WikiMergePatch(BaseModel):
    """LLM output shape when merging a Q&A into an existing page."""

    summary: str = Field(default="")
    key_points: list[str] = Field(default_factory=list)
