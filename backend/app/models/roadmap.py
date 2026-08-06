from pydantic import BaseModel, Field

from app.models.topic import Dependency, Topic


class GenerateRoadmapRequest(BaseModel):
    goal: str | None = Field(default=None, description="A learning goal, e.g. 'learn how transformers work'")
    topics: list[str] | None = Field(default=None, description="A flat topic dump, one topic per entry")
    filenames: list[str] | None = Field(
        default=None,
        description="Ingested raw note basenames (as returned by /ingest) to use as source content",
    )


class SkippedDependency(BaseModel):
    from_title: str
    to_title: str
    reason: str


class GenerateRoadmapResponse(BaseModel):
    created_topics: list[Topic]
    created_dependencies: list[Dependency]
    skipped_dependencies: list[SkippedDependency] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
