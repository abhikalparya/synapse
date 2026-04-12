import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_NOTES_DIR = _PROJECT_ROOT / "raw_notes"


def _ensure_raw_notes_dir() -> None:
    RAW_NOTES_DIR.mkdir(parents=True, exist_ok=True)


def _slugify_stem(name: str) -> str:
    base = (name or "upload").strip().lower()
    base = re.sub(r"[^\w\s-]", "", base, flags=re.UNICODE)
    base = re.sub(r"[-\s]+", "-", base).strip("-")
    return base[:80] if base else "upload"


def list_raw_note_files() -> list[Path]:
    """Return sorted paths to .txt notes under raw_notes/ (excludes non-txt)."""
    _ensure_raw_notes_dir()
    paths = sorted(p for p in RAW_NOTES_DIR.glob("*.txt") if p.is_file())
    return paths


def read_raw_note(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def resolve_raw_note_file(filename: str) -> Path | None:
    """
    Return a resolved path under ``raw_notes/`` for a bare ``*.txt`` basename, or None.
    Rejects path components (traversal) and missing files.
    """
    s = (filename or "").strip()
    if not s or "/" in s or "\\" in s or s.startswith("."):
        return None
    name = Path(s).name
    if name != s or not name.endswith(".txt"):
        return None
    p = (RAW_NOTES_DIR / name).resolve()
    try:
        p.relative_to(RAW_NOTES_DIR.resolve())
    except ValueError:
        return None
    if not p.is_file():
        return None
    return p


def save_raw_note(content: str, *, original_filename: str | None = None) -> Path:
    """
    Write content to a new .txt under raw_notes/.

    If ``original_filename`` is set (e.g. ``report.pdf``), the note filename is
    ``{slug-stem}_{timestamp}_{random}.txt``; otherwise ``{timestamp}_{random}.txt``.
    """
    _ensure_raw_notes_dir()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    if original_filename:
        stem = _slugify_stem(Path(original_filename).stem)
        fname = f"{stem}_{stamp}_{suffix}.txt"
    else:
        fname = f"{stamp}_{suffix}.txt"
    path = RAW_NOTES_DIR / fname
    path.write_text(content, encoding="utf-8")
    logger.info("Saved raw note: %s", path)
    return path
