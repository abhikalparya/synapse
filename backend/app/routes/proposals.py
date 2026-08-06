import logging

from fastapi import APIRouter, HTTPException

from app.models.proposal import (
    ApplyRequest,
    ApplyResponse,
    DiscardRequest,
    Proposal,
    RollbackRequest,
    RollbackResponse,
)
from app.services.proposals import apply_proposal, discard_proposal, list_proposals, load_proposal
from app.services.snapshots import restore_snapshot

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/proposals", response_model=list[Proposal])
async def get_proposals(status: str | None = None):
    return list_proposals(status=status)


@router.get("/proposals/{proposal_id}", response_model=Proposal)
async def get_proposal(proposal_id: str):
    proposal = load_proposal(proposal_id)
    if proposal is None:
        raise HTTPException(status_code=404, detail=f"No proposal with id {proposal_id!r}")
    return proposal


@router.post("/apply", response_model=ApplyResponse)
async def apply(body: ApplyRequest):
    """Commit a pending proposal. This is the only endpoint that can mutate the graph."""
    try:
        return apply_proposal(body.proposal_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/discard", response_model=Proposal)
async def discard(body: DiscardRequest):
    try:
        return discard_proposal(body.proposal_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/rollback", response_model=RollbackResponse)
async def rollback(body: RollbackRequest):
    """Restore the graph to a prior snapshot (defaults to the most recent apply)."""
    try:
        result = restore_snapshot(body.snapshot_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    logger.info("POST /rollback restored snapshot=%s", result["snapshot_id"])
    return result
