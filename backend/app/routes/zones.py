import logging

from fastapi import APIRouter, HTTPException

from app.models.zone import Zone, ZoneCreate, ZoneUpdate
from app.services.zones import create_zone, delete_zone, get_zone_by_id, load_all_zones, update_zone

router = APIRouter()
logger = logging.getLogger(__name__)


def _zone_out(row: dict) -> Zone:
    return Zone.model_validate(row)


@router.get("/zones", response_model=list[Zone])
async def list_zones():
    return [_zone_out(row) for row in load_all_zones()]


@router.post("/zones", response_model=Zone, status_code=201)
async def create_zone_route(body: ZoneCreate):
    row = create_zone(body.label, body.color)
    return _zone_out(row)


@router.get("/zones/{zone_id}", response_model=Zone)
async def get_zone(zone_id: str):
    row = get_zone_by_id(zone_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"No zone with id {zone_id!r}")
    return _zone_out(row)


@router.patch("/zones/{zone_id}", response_model=Zone)
async def patch_zone(zone_id: str, body: ZoneUpdate):
    fields_set = body.model_fields_set
    updated = update_zone(
        zone_id,
        label=body.label if "label" in fields_set else None,
        color_set="color" in fields_set,
        color=body.color,
    )
    if updated is None:
        raise HTTPException(status_code=404, detail=f"No zone with id {zone_id!r}")
    return _zone_out(updated)


@router.delete("/zones/{zone_id}", status_code=204)
async def delete_zone_route(zone_id: str):
    if not delete_zone(zone_id):
        raise HTTPException(status_code=404, detail=f"No zone with id {zone_id!r}")
