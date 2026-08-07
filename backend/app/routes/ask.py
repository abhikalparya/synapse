import logging

from fastapi import APIRouter, HTTPException

from app.models.ask import AskRequest, AskResponse
from app.services.ask import answer_topic_question

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/topics/{topic_id}/ask", response_model=AskResponse)
async def ask_about_topic(topic_id: str, body: AskRequest):
    result = await answer_topic_question(topic_id, body.question)
    if result is None:
        raise HTTPException(status_code=404, detail=f"No topic with id {topic_id!r}")
    return AskResponse(**result)
