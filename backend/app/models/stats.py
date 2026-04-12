from pydantic import BaseModel, Field


class TagCount(BaseModel):
    tag: str
    count: int = Field(ge=1)


class RecentNode(BaseModel):
    title: str
    filename: str
    created_at: str | None = None
    updated_at: str | None = None
    tags: list[str] = Field(default_factory=list)


class KnowledgeStatsResponse(BaseModel):
    total_nodes: int = Field(ge=0)
    total_edges: int = Field(ge=0)
    recent_nodes: list[RecentNode]
    top_tags: list[TagCount]
