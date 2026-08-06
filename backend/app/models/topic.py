import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

TopicStatus = Literal["not_started", "in_progress", "complete"]
ResourceType = Literal["link", "document", "note"]


class Resource(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    type: ResourceType
    source_ref: str = Field(default="")
    title: str = Field(default="")


class Topic(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    title: str = Field(default="", max_length=500)
    summary: str = Field(default="")
    status: TopicStatus = Field(default="not_started")
    resources: list[Resource] = Field(default_factory=list)
    created_at: datetime | None = Field(default=None)
    updated_at: datetime | None = Field(default=None)


class Dependency(BaseModel):
    """Directed prerequisite edge: ``from_topic`` requires ``to_topic``."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    from_topic_id: str
    to_topic_id: str
    created_at: datetime | None = Field(default=None)


class TopicCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    summary: str = Field(default="")
    status: TopicStatus = Field(default="not_started")


class DependencyCreate(BaseModel):
    from_topic_id: str
    to_topic_id: str
