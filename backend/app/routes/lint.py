import logging

from fastapi import APIRouter, Query

from app.models.lint import LintResponse
from app.services.lint import run_lint

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/lint", response_model=LintResponse)
async def get_knowledge_lint(
    suggest: bool = Query(
        default=False,
        description="If true, attach optional LLM fix hints (extra latency/cost).",
    ),
):
    """Run knowledge lint rules over all wiki pages."""
    result = await run_lint(suggest_fixes=suggest)
    logger.info("GET /lint issues=%s suggest=%s", len(result.issues), suggest)
    return result
