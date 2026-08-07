import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class Zone(BaseModel):
    """A visual/logical grouping region on the graph. Non-overlapping by design: a topic
    belongs to at most one zone at a time (``Topic.zone_id``), which is what makes the
    force-graph convex-hull rendering unambiguous."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    label: str = Field(default="", max_length=200)
    color: str | None = Field(default=None, description="Optional CSS color hint, e.g. '#8b5cf6'")
    created_at: datetime | None = Field(default=None)


class ZoneCreate(BaseModel):
    label: str = Field(..., min_length=1, max_length=200)
    color: str | None = Field(default=None)


class ZoneUpdate(BaseModel):
    label: str | None = None
    color: str | None = None
