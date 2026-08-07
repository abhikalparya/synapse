import logging

from fastapi import APIRouter, HTTPException

from app.models.artifact import Artifact, ArtifactCreate
from app.services.artifacts import create_artifact, list_artifacts_for_topic
from app.services.topics import get_topic_by_id

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/topics/{topic_id}/artifacts", response_model=list[Artifact])
async def list_artifacts(topic_id: str):
    if get_topic_by_id(topic_id) is None:
        raise HTTPException(status_code=404, detail=f"No topic with id {topic_id!r}")
    return [Artifact.model_validate(row) for row in list_artifacts_for_topic(topic_id)]


@router.post("/topics/{topic_id}/artifacts", response_model=Artifact, status_code=201)
async def add_artifact(topic_id: str, body: ArtifactCreate):
    row = create_artifact(topic_id, body)
    if row is None:
        raise HTTPException(status_code=404, detail=f"No topic with id {topic_id!r}")
    return Artifact.model_validate(row)
