import logging

from fastapi import APIRouter, HTTPException
from openai import APIError

from app.models.refactor import RefactorResponse
from app.services.refactor import run_refactor

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/refactor", response_model=RefactorResponse)
async def trigger_refactor():
    """Merge duplicate wiki pages, LLM-rewrite merged and low-quality pages, optional stale batch (REFACTOR_REWRITE_MAX)."""
    try:
        result = await run_refactor()
    except (RuntimeError, APIError) as exc:
        logger.warning("POST /refactor failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    logger.info(
        "POST /refactor merged_groups=%s pages_merged=%s pages_updated=%s pages_rewritten=%s errors=%s",
        result.merged_groups,
        result.pages_merged,
        result.pages_updated,
        result.pages_rewritten,
        len(result.errors),
    )
    return result
