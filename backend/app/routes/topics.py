import logging

from fastapi import APIRouter, HTTPException

from app.models.topic import Dependency, DependencyCreate, ResourceCreate, Topic, TopicCreate, TopicUpdate
from app.services.file_handler import resolve_raw_note_file
from app.services.quiz import load_quiz, quiz_gate_completion_enabled
from app.services.topics import (
    DependencyCycleError,
    add_dependency,
    attach_resource,
    get_topic_by_id,
    load_all_topics,
    load_dependencies,
    save_topic,
    update_topic,
)

router = APIRouter()
logger = logging.getLogger(__name__)


def _topic_out(row: dict) -> Topic:
    return Topic.model_validate(row)


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


@router.patch("/topics/{topic_id}", response_model=Topic)
async def patch_topic(topic_id: str, body: TopicUpdate):
    row = get_topic_by_id(topic_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"No topic with id {topic_id!r}")

    if body.status is None:
        return _topic_out(row)

    if body.status == "complete" and quiz_gate_completion_enabled():
        quiz = load_quiz(topic_id)
        if quiz is not None and not row.get("quiz_passed", False):
            raise HTTPException(
                status_code=409,
                detail="This topic has a quiz that hasn't been passed yet -- pass it before marking complete.",
            )

    updated = update_topic(topic_id, status=body.status)
    if updated is None:
        raise HTTPException(status_code=404, detail=f"No topic with id {topic_id!r}")
    return _topic_out(updated)


@router.post("/topics/{topic_id}/resources", response_model=Topic, status_code=201)
async def add_resource(topic_id: str, body: ResourceCreate):
    row = get_topic_by_id(topic_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"No topic with id {topic_id!r}")

    if body.type in ("document", "note") and resolve_raw_note_file(body.source_ref) is None:
        raise HTTPException(
            status_code=422,
            detail=f"source_ref {body.source_ref!r} is not a known ingested note (see POST /ingest)",
        )

    updated = attach_resource(topic_id, body)
    if updated is None:
        raise HTTPException(status_code=404, detail=f"No topic with id {topic_id!r}")
    return _topic_out(updated)


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
