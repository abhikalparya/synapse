from typing import Literal

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw note text to store")


class IngestResponse(BaseModel):
    status: Literal["ok", "warning"]
    path: str | None = None
    filename: str | None = None
    warnings: list[str] = Field(default_factory=list)
    file_type: str | None = Field(
        default=None,
        description="Source hint: text, txt, md, pdf, docx",
    )


class BatchIngestItem(BaseModel):
    """One file outcome from ``POST /ingest/upload/batch``."""

    filename: str
    status: Literal["ok", "warning", "error"]
    path: str | None = None
    saved_filename: str | None = None
    warnings: list[str] = Field(default_factory=list)
    file_type: str | None = None
    detail: str | None = Field(default=None, description="Error message when status is error")


class BatchIngestResponse(BaseModel):
    items: list[BatchIngestItem]
