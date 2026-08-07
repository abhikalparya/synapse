from pydantic import BaseModel, Field


class ObsidianImportRequest(BaseModel):
    vault_path: str = Field(..., min_length=1, description="Filesystem path to an Obsidian vault folder")
