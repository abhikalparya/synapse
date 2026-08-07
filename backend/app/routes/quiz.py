import logging

from fastapi import APIRouter, HTTPException

from app.models.quiz import QuizPublic, QuizResult, QuizSubmission
from app.services.quiz import generate_quiz_for_topic, submit_quiz

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/topics/{topic_id}/quiz", response_model=QuizPublic)
async def generate_quiz(topic_id: str):
    """Generate a closure quiz from the topic's summary + resources; resets quiz_passed."""
    try:
        return await generate_quiz_for_topic(topic_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.warning("POST /topics/%s/quiz failed: %s", topic_id, exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/topics/{topic_id}/quiz/submit", response_model=QuizResult)
async def submit_quiz_route(topic_id: str, body: QuizSubmission):
    try:
        return submit_quiz(topic_id, body)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
