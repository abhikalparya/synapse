from pydantic import BaseModel, Field


class GenerateRoadmapRequest(BaseModel):
    goal: str | None = Field(default=None, description="A learning goal, e.g. 'learn how transformers work'")
    topics: list[str] | None = Field(default=None, description="A flat topic dump, one topic per entry")
    filenames: list[str] | None = Field(
        default=None,
        description="Ingested raw note basenames (as returned by /ingest) to use as source content",
    )
