import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

ArtifactType = Literal["note", "code_snippet", "summary", "generated_output", "qa_log"]


class Artifact(BaseModel):
    """Something a learner PRODUCED while studying a topic -- distinct from a Resource,
    which is something they studied FROM (an input). Kept out of the Topic model's own
    serialization (unlike Resource) since artifact content can be arbitrarily long;
    fetched via dedicated /topics/{id}/artifacts endpoints instead."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    topic_id: str
    type: ArtifactType
    title: str = Field(default="")
    content: str = Field(default="")
    created_at: datetime | None = Field(default=None)


class ArtifactCreate(BaseModel):
    type: ArtifactType
    title: str = Field(default="")
    content: str = Field(..., min_length=1)
