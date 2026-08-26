from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# Product API surface only. Closed experiments (concept_first, coverage_recovery)
# remain available to evaluation adapters, not the public ingest API.
GenerationStrategyLiteral = Literal[
    "baseline",
    "domain_curriculum_prior",
    "domain_prior_edge_classifier",
]


class IngestRequest(BaseModel):
    goal: str | None = Field(default=None, description="A learning goal, e.g. 'learn how transformers work'")
    topics: list[str] | None = Field(default=None, description="A flat topic dump, one topic per entry")
    filenames: list[str] | None = Field(
        default=None,
        description="Ingested raw note basenames (as returned by /ingest) to use as source content",
    )
    generation_strategy: GenerationStrategyLiteral | None = Field(
        default=None,
        description=(
            "Graph generation strategy. Default/omitted = baseline (production). "
            "Opt-in experimental: 'domain_curriculum_prior'. "
            "Experimental only: 'domain_prior_edge_classifier'."
        ),
    )
    curriculum_domain: str | None = Field(
        default=None,
        description=(
            "Optional domain for domain_curriculum_prior / domain_prior_edge_classifier. "
            "When omitted, Synapse may resolve via SYNAPSE_CURRICULUM_DOMAIN or fall back "
            "to baseline (unless require_domain_prior is true)."
        ),
    )
    require_domain_prior: bool = Field(
        default=False,
        description=(
            "When true with a domain-prior strategy, return DOMAIN_PRIOR_UNAVAILABLE / "
            "DOMAIN_UNRESOLVED instead of silently falling back to baseline."
        ),
    )


class ExpandRequest(BaseModel):
    topic_id: str
    instructions: str | None = Field(default=None, description="Optional free-text guidance for the expansion")


class ReshapeRequest(BaseModel):
    topic_ids: list[str] = Field(..., min_length=1, description="The subgraph to restructure")
    instructions: str | None = Field(default=None, description="Optional free-text guidance for the restructuring")


AuditFindingType = Literal["orphaned_topic", "duplicate_title", "thin_topic", "missing_prerequisite", "cycle_risk"]
AuditStatus = Literal["ok", "partial"]
SemanticAnalysisStatus = Literal["available", "unavailable"]


class AuditFinding(BaseModel):
    type: AuditFindingType
    topic_ids: list[str] = Field(default_factory=list)
    detail: str


class AuditReport(BaseModel):
    generated_at: datetime
    total_topics: int
    findings: list[AuditFinding] = Field(
        default_factory=list,
        description="Combined structural + semantic findings (semantic may be empty when unavailable)",
    )
    status: AuditStatus = Field(
        default="ok",
        description="'partial' when structural checks ran but semantic LLM analysis was unavailable",
    )
    semantic_analysis: SemanticAnalysisStatus = Field(
        default="available",
        description="Whether the LLM semantic pass contributed findings (or confirmed none)",
    )
    semantic_error: str | None = Field(
        default=None,
        description="Error detail when semantic_analysis is unavailable",
    )
    structural_findings: list[AuditFinding] = Field(
        default_factory=list,
        description="Deterministic structural findings only (always present when status is partial)",
    )
