from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    goal: str | None = Field(default=None, description="A learning goal, e.g. 'learn how transformers work'")
    topics: list[str] | None = Field(default=None, description="A flat topic dump, one topic per entry")
    filenames: list[str] | None = Field(
        default=None,
        description="Ingested raw note basenames (as returned by /ingest) to use as source content",
    )


class ExpandRequest(BaseModel):
    topic_id: str
    instructions: str | None = Field(default=None, description="Optional free-text guidance for the expansion")


class ReshapeRequest(BaseModel):
    topic_ids: list[str] = Field(..., min_length=1, description="The subgraph to restructure")
    instructions: str | None = Field(default=None, description="Optional free-text guidance for the restructuring")


AuditFindingType = Literal["orphaned_topic", "duplicate_title", "thin_topic", "missing_prerequisite", "cycle_risk"]


class AuditFinding(BaseModel):
    type: AuditFindingType
    topic_ids: list[str] = Field(default_factory=list)
    detail: str


class AuditReport(BaseModel):
    generated_at: datetime
    total_topics: int
    findings: list[AuditFinding] = Field(default_factory=list)
