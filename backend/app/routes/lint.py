import logging

from fastapi import APIRouter

from app.models.lint import LintResponse
from app.services.lint import run_lint

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/lint", response_model=LintResponse)
async def get_knowledge_lint():
    """Run DAG/schema consistency checks over topics and dependencies."""
    result = await run_lint()
    logger.info("GET /lint issues=%s", len(result.issues))
    return result
