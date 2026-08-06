import logging

from fastapi import APIRouter, HTTPException
from openai import APIError

from app.models.proposal import Proposal
from app.models.roadmap import GenerateRoadmapRequest
from app.services.roadmap import generate_roadmap

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/generate/roadmap", response_model=Proposal)
async def generate_roadmap_route(body: GenerateRoadmapRequest):
    """
    Turn a goal, a topic dump, and/or ingested raw notes into a proposed Topic + Dependency
    DAG via LLM -- returns a pending Proposal for review. Nothing is written to the graph
    until it's committed with POST /apply.
    """
    try:
        return await generate_roadmap(goal=body.goal, topics=body.topics, filenames=body.filenames)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (RuntimeError, APIError) as exc:
        logger.warning("POST /generate/roadmap failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
