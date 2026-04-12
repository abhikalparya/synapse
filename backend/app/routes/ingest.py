import logging
from pathlib import Path
from typing import Callable

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.models.ingest import BatchIngestItem, BatchIngestResponse, IngestRequest, IngestResponse
from app.services.file_handler import save_raw_note
from app.services.parser import parse_docx, parse_md, parse_pdf, parse_txt

router = APIRouter()
logger = logging.getLogger(__name__)

_MAX_BATCH_FILES = 30


def _http_exc_detail(exc: HTTPException) -> str:
    d = exc.detail
    if isinstance(d, str):
        return d
    if isinstance(d, list):
        return "; ".join(str(x) for x in d)
    return str(d)

_EXT_HANDLERS: dict[str, Callable[[bytes], str]] = {
    ".txt": parse_txt,
    ".md": parse_md,
    ".pdf": parse_pdf,
    ".docx": parse_docx,
}


def _ingest_bytes(filename: str, raw: bytes) -> IngestResponse:
    ext = Path(filename).suffix.lower()
    if ext not in _EXT_HANDLERS:
        logger.info("Ingest upload: unsupported ext=%s filename=%s", ext, filename)
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type {ext!r}; allowed: {', '.join(sorted(_EXT_HANDLERS))}",
        )

    logger.info(
        "Ingest upload: filename=%s ext=%s bytes=%s",
        filename,
        ext,
        len(raw) if raw is not None else 0,
    )

    try:
        text = _EXT_HANDLERS[ext](raw)
    except ValueError as exc:
        logger.warning("Ingest parse failed (ValueError): %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.warning("Ingest parse failed (RuntimeError): %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.warning("Ingest parse failed: %s", exc)
        raise HTTPException(status_code=422, detail=f"Extraction failed: {exc}") from exc

    if not text or not text.strip():
        logger.info(
            "Ingest upload: empty extraction ext=%s filename=%s",
            ext,
            filename,
        )
        return IngestResponse(
            status="warning",
            warnings=["Empty extraction after parsing; nothing saved"],
            file_type=ext.lstrip("."),
        )

    try:
        path = save_raw_note(text, original_filename=filename)
    except OSError as exc:
        logger.warning("Ingest upload save failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    logger.info(
        "Ingest upload: extraction success ext=%s saved=%s",
        ext,
        path.name,
    )
    return IngestResponse(
        status="ok",
        path=str(path),
        filename=path.name,
        file_type=ext.lstrip("."),
    )


@router.post("/ingest/upload/batch", response_model=BatchIngestResponse)
async def ingest_upload_batch(
    files: list[UploadFile] = File(..., description="One or more documents: .txt, .md, .pdf, .docx"),
):
    """
    Upload multiple documents in one request. Each file is parsed and saved as a raw note;
    failures for one file do not block the rest.
    """
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required")
    if len(files) > _MAX_BATCH_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files (max {_MAX_BATCH_FILES}); split into multiple requests",
        )

    items: list[BatchIngestItem] = []
    for upload in files:
        name = (upload.filename or "upload").strip() or "upload"
        try:
            raw = await upload.read()
        except Exception as exc:
            logger.warning("Batch ingest read failed for %s: %s", name, exc)
            items.append(
                BatchIngestItem(
                    filename=name,
                    status="error",
                    detail="Could not read uploaded file",
                ),
            )
            continue

        try:
            resp = _ingest_bytes(name, raw)
        except HTTPException as exc:
            logger.info("Batch ingest rejected %s: %s", name, exc.detail)
            items.append(
                BatchIngestItem(
                    filename=name,
                    status="error",
                    detail=_http_exc_detail(exc),
                ),
            )
            continue

        items.append(
            BatchIngestItem(
                filename=name,
                status=resp.status,
                path=resp.path,
                saved_filename=resp.filename,
                warnings=list(resp.warnings or []),
                file_type=resp.file_type,
            ),
        )

    return BatchIngestResponse(items=items)


@router.post("/ingest/upload", response_model=IngestResponse)
async def ingest_upload_file(
    file: UploadFile = File(..., description="Document: .txt, .md, .pdf, or .docx"),
):
    """
    Multipart file upload (preferred). Uses Starlette/FastAPI's native parser — avoids
    ``request.form()`` issues with some proxies and clients.
    """
    filename = file.filename or "upload"
    try:
        raw = await file.read()
    except Exception as exc:
        logger.warning("Ingest upload read failed: %s", exc)
        raise HTTPException(status_code=400, detail="Could not read uploaded file") from exc

    return _ingest_bytes(filename, raw)


@router.post("/ingest", response_model=IngestResponse)
async def ingest_note(request: Request):
    """
    Ingest raw text (``application/json`` body ``{ "text": "..." }``) or legacy
    ``multipart/form-data`` field ``file`` (prefer ``POST /ingest/upload`` for files).
    """
    content_type = (request.headers.get("content-type") or "").split(";")[0].strip().lower()

    if content_type == "application/json":
        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Invalid JSON body") from exc
        try:
            data = IngestRequest.model_validate(body)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        text = data.text.strip()
        if not text:
            logger.info("Ingest JSON: empty text after strip")
            return IngestResponse(
                status="warning",
                warnings=["Empty text after trim; nothing saved"],
                file_type="text",
            )

        try:
            path = save_raw_note(data.text)
        except OSError as exc:
            logger.warning("Ingest JSON save failed: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        logger.info("Ingest JSON: success filename=%s", path.name)
        return IngestResponse(
            status="ok",
            path=str(path),
            filename=path.name,
            file_type="text",
        )

    if content_type == "multipart/form-data":
        try:
            form = await request.form()
        except Exception as exc:
            logger.warning("Ingest multipart parse failed: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid multipart form. Use POST /ingest/upload for file uploads "
                    f"(parse error: {exc!r})"
                ),
            ) from exc

        upload = form.get("file")
        if upload is None:
            raise HTTPException(
                status_code=400,
                detail="Missing form field 'file' for multipart ingest",
            )
        if not isinstance(upload, UploadFile):
            raise HTTPException(
                status_code=400,
                detail="Form field 'file' must be a file upload",
            )
        filename = upload.filename or "upload"
        try:
            raw = await upload.read()
        except Exception as exc:
            logger.warning("Ingest upload read failed: %s", exc)
            raise HTTPException(status_code=400, detail="Could not read uploaded file") from exc

        return _ingest_bytes(filename, raw)

    raise HTTPException(
        status_code=415,
        detail="Unsupported Content-Type; use application/json, multipart/form-data, or POST /ingest/upload",
    )
