import logging

from fastapi import APIRouter, HTTPException, Response

from app.models.obsidian import ObsidianImportRequest
from app.models.proposal import Proposal
from app.services.obsidian import VaultNotFoundError, export_vault_zip, import_vault

router = APIRouter(prefix="/obsidian", tags=["obsidian"])
logger = logging.getLogger(__name__)


@router.post("/import", response_model=Proposal, status_code=201)
async def obsidian_import(body: ObsidianImportRequest):
    try:
        return await import_vault(body.vault_path)
    except VaultNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/export")
async def obsidian_export(scope: str | None = None):
    try:
        payload = export_vault_zip(scope)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(
        content=payload,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="synapse-export.zip"'},
    )
