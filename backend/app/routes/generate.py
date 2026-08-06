import logging

from fastapi import APIRouter, HTTPException
from openai import APIError

from app.models.roadmap import GenerateRoadmapRequest, GenerateRoadmapResponse
from app.services.roadmap import generate_roadmap

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/generate/roadmap", response_model=GenerateRoadmapResponse)
async def generate_roadmap_route(body: GenerateRoadmapRequest):
    """
    Turn a goal, a topic dump, and/or ingested raw notes into a Topic + Dependency DAG via
    LLM. Every proposed dependency is validated against the DAG-cycle invariant from Phase 1;
    edges that would close a cycle (or reference an unknown title) are skipped and reported
    in ``skipped_dependencies`` rather than silently stored.
    """
    try:
        return await generate_roadmap(goal=body.goal, topics=body.topics, filenames=body.filenames)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (RuntimeError, APIError) as exc:
        logger.warning("POST /generate/roadmap failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
