import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from app.models.wiki import WikiMergePatch, WikiPage
from app.prompts.query import build_wiki_from_qa_prompt, build_wiki_merge_prompt
from app.prompts.wiki import build_wiki_generation_prompt
from app.services.llm import call_llm
from app.services.tags import align_related_topics, normalize_tags_list
from app.services.wiki_schema import ensure_wiki_schema_compliant

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WIKI_PAGES_DIR = _PROJECT_ROOT / "wiki_pages"


def _ensure_wiki_pages_dir() -> None:
    WIKI_PAGES_DIR.mkdir(parents=True, exist_ok=True)


def _strip_json_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _parse_wiki_json(raw: str) -> dict:
    cleaned = _strip_json_fences(raw)
    return json.loads(cleaned)


def _parse_merge_patch(raw: str) -> WikiMergePatch:
    cleaned = _strip_json_fences(raw)
    data = json.loads(cleaned)
    return WikiMergePatch.model_validate(data)


def _append_unique(sources: list[str], entry: str) -> list[str]:
    if entry in sources:
        return sources
    out = list(sources)
    out.append(entry)
    return out


def load_all_wiki_pages() -> list[dict]:
    """Load every *.json under wiki_pages/ as dicts with a Path ``path`` field."""
    _ensure_wiki_pages_dir()
    records: list[dict] = []
    for path in sorted(WIKI_PAGES_DIR.glob("*.json")):
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Skipping wiki page %s: %s", path.name, exc)
            continue
        if not isinstance(data, dict):
            logger.warning("Skipping non-object wiki JSON: %s", path.name)
            continue
        row = dict(data)
        row["path"] = path
        records.append(row)
    return records


def list_wiki_page_titles() -> list[str]:
    """Distinct non-empty wiki titles, stable sort for prompts."""
    titles: list[str] = []
    seen: set[str] = set()
    for row in load_all_wiki_pages():
        t = str(row.get("title", "")).strip()
        if t and t not in seen:
            seen.add(t)
            titles.append(t)
    return sorted(titles, key=str.casefold)


def _finalize_tags_and_related(data: dict, *, known_page_titles: list[str]) -> None:
    data["tags"] = normalize_tags_list(data.get("tags") or [])
    title_fold = str(data.get("title", "")).strip().casefold()
    related = align_related_topics(data.get("related_topics") or [], known_page_titles)
    data["related_topics"] = [x for x in related if str(x).strip().casefold() != title_fold]


async def update_wiki_page(path: Path, query: str, answer: str, *, confidence_score: float) -> None:
    """
    Merge the Q&A into an existing page by refreshing ``summary`` and ``key_points``
    via the LLM, then persist to the same file.
    """
    raw_text = path.read_text(encoding="utf-8")
    existing = json.loads(raw_text)
    if not isinstance(existing, dict):
        raise ValueError(f"Wiki file is not a JSON object: {path}")
    body = {k: v for k, v in existing.items() if k != "path"}

    merge_prompt = build_wiki_merge_prompt(body, query, answer)
    merge_raw = await call_llm(merge_prompt)
    try:
        patch = _parse_merge_patch(merge_raw)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"LLM merge output is invalid: {exc}") from exc

    new_summary = patch.summary.strip() or str(body.get("summary", ""))
    raw_points = patch.key_points if patch.key_points else list(body.get("key_points") or [])
    new_points = [str(p).strip() for p in raw_points if str(p).strip()]

    prior_sources = [str(s).strip() for s in (body.get("source_notes") or []) if str(s).strip()]
    query_marker = f"query:{query.strip()}"
    source_notes = _append_unique(prior_sources, query_marker)

    now = datetime.now(timezone.utc)
    prior = WikiPage.model_validate(body)
    created_at = prior.created_at or now

    merged = {
        **body,
        "summary": new_summary,
        "key_points": new_points,
        "source_notes": source_notes,
        "created_at": created_at,
        "updated_at": now,
        "confidence_score": confidence_score,
    }
    known = list_wiki_page_titles()
    merged = await ensure_wiki_schema_compliant(merged, known_page_titles=known)
    _finalize_tags_and_related(merged, known_page_titles=known)
    page = WikiPage.model_validate(merged)
    path.write_text(
        json.dumps(page.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Updated wiki page: %s", path)


async def create_new_page_from_query(query: str, answer: str, *, confidence_score: float) -> tuple[Path, str]:
    """Create a wiki JSON file from the query; returns ``(path, page_title)``."""
    known = list_wiki_page_titles()
    prompt = build_wiki_from_qa_prompt(query, answer, known_page_titles=known)
    raw = await call_llm(prompt)
    try:
        data = _parse_wiki_json(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM output is not valid JSON: {exc}") from exc
    data.pop("source_notes", None)
    data.pop("created_at", None)
    data.pop("updated_at", None)
    data.pop("confidence_score", None)
    fixed = await ensure_wiki_schema_compliant(data, known_page_titles=known)
    try:
        page = WikiPage.model_validate(fixed)
    except ValidationError as exc:
        raise ValueError(f"LLM JSON does not match wiki schema: {exc}") from exc
    dumped = page.model_dump()
    dumped["source_notes"] = _append_unique([], f"query:{query.strip()}")
    dumped["confidence_score"] = confidence_score
    path = await save_wiki_page(dumped)
    title = str(dumped.get("title", "")).strip() or path.stem
    return path, title


async def generate_wiki_from_note(
    note: str,
    *,
    source_note_name: str | None = None,
    known_page_titles: list[str] | None = None,
) -> dict:
    """Call the LLM and return a validated wiki page dict with optional raw-note provenance."""
    prompt = build_wiki_generation_prompt(note, known_page_titles=known_page_titles)
    raw = await call_llm(prompt)
    try:
        data = _parse_wiki_json(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM output is not valid JSON: {exc}") from exc
    data.pop("source_notes", None)
    data.pop("created_at", None)
    data.pop("updated_at", None)
    data.pop("confidence_score", None)
    fixed = await ensure_wiki_schema_compliant(
        data,
        known_page_titles=known_page_titles or [],
    )
    try:
        page = WikiPage.model_validate(fixed)
    except ValidationError as exc:
        raise ValueError(f"LLM JSON does not match wiki schema: {exc}") from exc
    dumped = page.model_dump()
    if source_note_name:
        name = source_note_name.strip()
        if name:
            dumped["source_notes"] = _append_unique(list(dumped.get("source_notes") or []), name)
    _finalize_tags_and_related(dumped, known_page_titles=known_page_titles or [])
    return dumped


def _slugify_title(title: str) -> str:
    base = (title or "untitled").lower().strip()
    base = re.sub(r"[^\w\s-]", "", base, flags=re.UNICODE)
    base = re.sub(r"[-\s]+", "-", base).strip("-")
    return base[:80] if base else "untitled"


async def save_wiki_page(data: dict) -> Path:
    """Persist wiki JSON under wiki_pages/; logs creation."""
    _ensure_wiki_pages_dir()
    now = datetime.now(timezone.utc)
    d = {k: v for k, v in data.items() if k != "path"}
    known = list_wiki_page_titles()
    d = await ensure_wiki_schema_compliant(d, known_page_titles=known)
    _finalize_tags_and_related(d, known_page_titles=known)
    if d.get("created_at") is None:
        d["created_at"] = now
    d["updated_at"] = now
    page = WikiPage.model_validate(d)
    payload = page.model_dump(mode="json")
    stamp = now.strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    slug = _slugify_title(str(payload.get("title", "")))
    path = WIKI_PAGES_DIR / f"{slug}_{stamp}_{suffix}.json"
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Created wiki page: %s", path)
    return path
