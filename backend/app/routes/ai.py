import logging

from fastapi import APIRouter, HTTPException
from openai import APIError

from app.models.ai_ops import AuditReport, ExpandRequest, IngestRequest, ReshapeRequest
from app.models.proposal import Proposal
from app.services.audit import run_audit
from app.services.expand import run_expand
from app.services.ingest import run_ingest
from app.services.reshape import run_reshape

router = APIRouter(prefix="/ai")
logger = logging.getLogger(__name__)


@router.post("/ingest", response_model=Proposal)
async def ingest(body: IngestRequest):
    """
    Turn a goal, a topic dump, and/or ingested raw notes into a proposed Topic + Dependency
    DAG via LLM -- returns a pending Proposal for review. Nothing is written to the graph
    until it's committed with POST /apply.
    """
    try:
        return await run_ingest(goal=body.goal, topics=body.topics, filenames=body.filenames)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (RuntimeError, APIError) as exc:
        logger.warning("POST /ai/ingest failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/expand", response_model=Proposal)
async def expand(body: ExpandRequest):
    """Propose new sub-topics/prerequisites beneath ONE existing topic, scoped to just that
    topic (never regenerates the rest of the graph). Returns a pending Proposal for review."""
    try:
        return await run_expand(topic_id=body.topic_id, instructions=body.instructions)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (RuntimeError, APIError) as exc:
        logger.warning("POST /ai/expand failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/reshape", response_model=Proposal)
async def reshape(body: ReshapeRequest):
    """Propose a restructuring (split/merge/reorder) of a SELECTED topic subgraph, scoped
    to just that selection. Always produces a Proposal for review -- the most invasive mode,
    never applies directly."""
    try:
        return await run_reshape(topic_ids=body.topic_ids, instructions=body.instructions)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (RuntimeError, APIError) as exc:
        logger.warning("POST /ai/reshape failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/audit", response_model=AuditReport)
async def audit():
    """Read-only analysis of the current graph -- orphaned topics, duplicate titles, thin
    summaries, missing prerequisites, cycle-risk relationships. Never mutates state, even
    indirectly; returns a diagnostic report, not a Proposal."""
    return await run_audit()
