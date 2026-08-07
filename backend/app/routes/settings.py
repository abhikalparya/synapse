from fastapi import APIRouter

from app.models.settings import Settings, SettingsUpdate
from app.services.settings import load_settings, update_settings

router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("", response_model=Settings)
async def get_settings():
    return Settings.model_validate(load_settings())


@router.patch("", response_model=Settings)
async def patch_settings(body: SettingsUpdate):
    fields = body.model_dump(include=body.model_fields_set)
    if not fields:
        return Settings.model_validate(load_settings())
    return Settings.model_validate(update_settings(**fields))
