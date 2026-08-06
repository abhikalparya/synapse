from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.models.topic import Dependency, Topic


class ProposedTopic(BaseModel):
    temp_id: str
    title: str
    summary: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    needs_review: bool = False


class ProposedDependency(BaseModel):
    from_temp_id: str
    to_temp_id: str


class SkippedProposedDependency(BaseModel):
    from_title: str
    to_title: str
    reason: str


class Proposal(BaseModel):
    id: str
    status: Literal["pending", "applied", "discarded"] = "pending"
    source: str = Field(default="", description="Human-readable description of what generated this proposal")
    topics: list[ProposedTopic] = Field(default_factory=list)
    dependencies: list[ProposedDependency] = Field(default_factory=list)
    skipped_dependencies: list[SkippedProposedDependency] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    created_at: datetime | None = None
    applied_at: datetime | None = None
    snapshot_id: str | None = Field(default=None, description="Snapshot taken immediately before this was applied")


class ApplyRequest(BaseModel):
    proposal_id: str


class ApplyResponse(BaseModel):
    proposal_id: str
    snapshot_id: str
    created_topics: list[Topic]
    created_dependencies: list[Dependency]
    skipped_dependencies: list[SkippedProposedDependency] = Field(default_factory=list)


class DiscardRequest(BaseModel):
    proposal_id: str


class RollbackRequest(BaseModel):
    snapshot_id: str | None = Field(default=None, description="Snapshot to restore; defaults to the most recent")


class RollbackResponse(BaseModel):
    snapshot_id: str
    restored_topics: int
    restored_dependencies: int
