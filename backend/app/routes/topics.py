import logging

from fastapi import APIRouter, HTTPException

from app.models.topic import Dependency, DependencyCreate, Topic, TopicCreate
from app.services.topics import (
    DependencyCycleError,
    add_dependency,
    get_topic_by_id,
    load_all_topics,
    load_dependencies,
    save_topic,
)

router = APIRouter()
logger = logging.getLogger(__name__)


def _topic_out(row: dict) -> Topic:
    return Topic.model_validate({k: v for k, v in row.items() if k != "path"})


@router.get("/topics", response_model=list[Topic])
async def list_topics():
    return [_topic_out(row) for row in load_all_topics()]


@router.post("/topics", response_model=Topic, status_code=201)
async def create_topic(body: TopicCreate):
    row = save_topic(body)
    return _topic_out(row)


@router.get("/topics/{topic_id}", response_model=Topic)
async def get_topic(topic_id: str):
    row = get_topic_by_id(topic_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"No topic with id {topic_id!r}")
    return _topic_out(row)


@router.get("/dependencies", response_model=list[Dependency])
async def list_dependencies():
    return [Dependency.model_validate(d) for d in load_dependencies()]


@router.post("/dependencies", response_model=Dependency, status_code=201)
async def create_dependency(body: DependencyCreate):
    try:
        payload = add_dependency(body)
    except DependencyCycleError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return Dependency.model_validate(payload)
