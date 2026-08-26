from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.models.topic import Dependency, Topic

ProposalMode = Literal["ingest", "expand", "reshape"]


class ProposedTopic(BaseModel):
    temp_id: str
    title: str
    summary: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    needs_review: bool = False
    source_note_path: str | None = Field(
        default=None,
        description="Vault-relative path of the Obsidian note this topic came from (Phase 11 import only); "
        "on apply, a matching Resource is attached to the created topic pointing back to it.",
    )


class ProposedDependency(BaseModel):
    """Either side may be a temp_id (a new topic proposed alongside this) or the real id
    of an existing topic already in the graph -- expand/reshape proposals routinely
    anchor new/changed structure to topics that already exist."""

    from_temp_id: str
    to_temp_id: str


class ProposedDependencyRemoval(BaseModel):
    """Drops an existing edge; both ids must be real, already-persisted topic ids."""

    from_topic_id: str
    to_topic_id: str
    reason: str = ""


class ProposedMerge(BaseModel):
    """Merges ``source_topic_id`` into ``target_topic_id``: every dependency and resource
    on the source is rewired onto the target, then the source topic is deleted."""

    source_topic_id: str
    target_topic_id: str
    reason: str = ""


class ProposedTopicEdit(BaseModel):
    """A scoped, low-risk edit to an existing topic -- summary text only, never id,
    status, or dependencies."""

    topic_id: str
    new_summary: str
    reason: str = ""


class SkippedProposedDependency(BaseModel):
    from_title: str
    to_title: str
    reason: str


class Proposal(BaseModel):
    id: str
    status: Literal["pending", "applied", "discarded"] = "pending"
    mode: ProposalMode = "ingest"
    source: str = Field(default="", description="Human-readable description of what generated this proposal")
    topics: list[ProposedTopic] = Field(default_factory=list)
    dependencies: list[ProposedDependency] = Field(default_factory=list)
    removed_dependencies: list[ProposedDependencyRemoval] = Field(default_factory=list)
    merges: list[ProposedMerge] = Field(default_factory=list)
    edits: list[ProposedTopicEdit] = Field(default_factory=list)
    skipped_dependencies: list[SkippedProposedDependency] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    generation_meta: dict = Field(
        default_factory=dict,
        description=(
            "Structured generation observability (strategy, domain, inventory version, "
            "fallback). Not used for graph mutation."
        ),
    )
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
    removed_dependency_count: int = 0
    merged_topic_count: int = 0
    edited_topic_count: int = 0
    skipped_dependencies: list[SkippedProposedDependency] = Field(default_factory=list)


class DiscardRequest(BaseModel):
    proposal_id: str


class RollbackRequest(BaseModel):
    snapshot_id: str | None = Field(default=None, description="Snapshot to restore; defaults to the most recent")


class RollbackResponse(BaseModel):
    snapshot_id: str
    restored_topics: int
    restored_dependencies: int
