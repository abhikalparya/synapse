from pydantic import BaseModel, Field


class RecentNode(BaseModel):
    id: str
    title: str
    status: str = "not_started"
    created_at: str | None = None
    updated_at: str | None = None


class KnowledgeStatsResponse(BaseModel):
    total_nodes: int = Field(ge=0)
    total_edges: int = Field(ge=0)
    recent_nodes: list[RecentNode]
