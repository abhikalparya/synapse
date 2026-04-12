import asyncio
import logging
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from openai import APIError
from pydantic import BaseModel, Field

from app.services.file_handler import list_raw_note_files, read_raw_note, resolve_raw_note_file
from app.services.wiki import generate_wiki_from_note, list_wiki_page_titles, save_wiki_page

router = APIRouter()
logger = logging.getLogger(__name__)


def _generate_concurrency() -> int:
    raw = os.environ.get("GENERATE_CONCURRENCY", "4").strip()
    try:
        n = int(raw)
    except ValueError:
        return 4
    return max(1, min(16, n))


class GenerateFromRawRequest(BaseModel):
    """Basenames of ``*.txt`` files under ``raw_notes/`` (as returned by ingest)."""

    filenames: list[str] = Field(..., min_length=1, max_length=30)


async def _generate_one_note(path: Path, known_titles: list[str]) -> tuple[str | None, dict[str, str] | None]:
    """Returns (created_wiki_basename, None) or (None, error_dict)."""
    try:
        text = read_raw_note(path)
        if not text.strip():
            return None, {"source": path.name, "detail": "empty note skipped"}
        data = await generate_wiki_from_note(
            text,
            source_note_name=path.name,
            known_page_titles=known_titles,
        )
        out = await save_wiki_page(data)
        return out.name, None
    except (OSError, ValueError, RuntimeError, APIError) as exc:
        logger.warning("Wiki generation failed for %s: %s", path.name, exc)
        return None, {"source": path.name, "detail": str(exc)}


async def _generate_notes_parallel(note_paths: list[Path]) -> dict:
    """
    Run wiki generation for each path with bounded concurrency.
    ``processed`` counts input paths; ``created`` / ``errors`` aggregate outcomes.
    """
    if not note_paths:
        return {"processed": 0, "created": [], "errors": []}

    known_titles = list_wiki_page_titles()
    sem = asyncio.Semaphore(_generate_concurrency())

    async def guarded(path: Path) -> tuple[str | None, dict[str, str] | None]:
        async with sem:
            return await _generate_one_note(path, known_titles)

    outcomes = await asyncio.gather(*[guarded(p) for p in note_paths])
    created: list[str] = []
    errors: list[dict[str, str]] = []
    for cname, err in outcomes:
        if cname is not None:
            created.append(cname)
        if err is not None:
            errors.append(err)
    return {
        "processed": len(note_paths),
        "created": created,
        "errors": errors,
    }


@router.post("/generate")
async def generate_wiki() -> dict:
    """
    Read every .txt in raw_notes/, generate structured wiki JSON per note via LLM,
    and write JSON files under wiki_pages/. Uses bounded parallel LLM calls.
    """
    note_paths = list_raw_note_files()
    return await _generate_notes_parallel(note_paths)


@router.post("/generate/from-raw")
async def generate_from_raw_notes(body: GenerateFromRawRequest) -> dict:
    """
    Generate wiki pages only for the listed raw note basenames (faster after ingest).
    Unknown names are reported in ``errors``; if no file resolves, returns 404.
    """
    paths_ordered: list[Path] = []
    seen: set[str] = set()
    errors_pre: list[dict[str, str]] = []

    for raw in body.filenames:
        s = (raw or "").strip()
        if not s:
            errors_pre.append({"source": str(raw), "detail": "empty name"})
            continue
        p = resolve_raw_note_file(s)
        if p is None:
            errors_pre.append({"source": s, "detail": "raw note not found or invalid name"})
            continue
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        paths_ordered.append(p)

    if not paths_ordered:
        raise HTTPException(
            status_code=404,
            detail="No matching raw note files under raw_notes/ (expected ingest .txt basenames).",
        )

    result = await _generate_notes_parallel(paths_ordered)
    result["errors"] = errors_pre + result["errors"]
    result["processed"] = len(body.filenames)
    return result
