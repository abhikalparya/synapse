"""Generative roadmap creation: goal / topic-dump / ingested notes -> Topic + Dependency DAG via LLM."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import ValidationError

from app.models.roadmap import GenerateRoadmapResponse, SkippedDependency
from app.models.topic import Dependency, DependencyCreate, Topic, TopicCreate
from app.prompts.roadmap import build_roadmap_generation_prompt
from app.services.file_handler import read_raw_note, resolve_raw_note_file
from app.services.llm import call_llm
from app.services.topics import DependencyCycleError, add_dependency, load_all_topics, save_topic

logger = logging.getLogger(__name__)


def _strip_json_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _build_source_text(
    goal: str | None,
    topics: list[str] | None,
    filenames: list[str] | None,
) -> tuple[str, list[str]]:
    """Combine goal / topic-dump / raw-note text into one prompt source; returns (text, errors)."""
    parts: list[str] = []
    errors: list[str] = []

    if goal and goal.strip():
        parts.append(f"Goal: {goal.strip()}")

    if topics:
        dump = "\n".join(f"- {t.strip()}" for t in topics if t.strip())
        if dump:
            parts.append(f"Topic dump:\n{dump}")

    for name in filenames or []:
        path = resolve_raw_note_file(name)
        if path is None:
            errors.append(f"raw note not found: {name}")
            continue
        try:
            text = read_raw_note(path)
        except OSError as exc:
            errors.append(f"failed to read {name}: {exc}")
            continue
        if text.strip():
            parts.append(f"Note ({name}):\n{text.strip()}")

    return "\n\n".join(parts), errors


def _parse_roadmap_json(raw: str) -> dict[str, Any]:
    cleaned = _strip_json_fences(raw)
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("LLM output is not a JSON object")
    return data


async def generate_roadmap(
    *,
    goal: str | None,
    topics: list[str] | None,
    filenames: list[str] | None,
) -> GenerateRoadmapResponse:
    """
    Call the LLM for a topic + dependency DAG from the given source(s), persist the topics,
    then add each dependency through the Phase 1 cycle check -- edges that would close a
    cycle (or reference an unknown title) are skipped and reported, never silently dropped.
    """
    source_text, source_errors = _build_source_text(goal, topics, filenames)
    if not source_text.strip():
        raise ValueError("Provide at least one of: goal, topics, filenames (with resolvable content)")

    known_titles = sorted({str(r.get("title", "")).strip() for r in load_all_topics() if r.get("title")})
    prompt = build_roadmap_generation_prompt(source_text, known_topic_titles=known_titles)
    raw = await call_llm(prompt)
    data = _parse_roadmap_json(raw)

    raw_topics = data.get("topics")
    raw_deps = data.get("dependencies")
    if not isinstance(raw_topics, list) or not raw_topics:
        raise ValueError("LLM response did not include a non-empty 'topics' list")
    if not isinstance(raw_deps, list):
        raw_deps = []

    created_topics: list[Topic] = []
    title_to_id: dict[str, str] = {}
    errors: list[str] = list(source_errors)

    for row in raw_topics:
        if not isinstance(row, dict):
            continue
        title = str(row.get("title", "")).strip()
        if not title:
            continue
        summary = str(row.get("summary", "")).strip()
        try:
            create = TopicCreate(title=title, summary=summary)
        except ValidationError as exc:
            errors.append(f"invalid topic {title!r}: {exc}")
            continue
        stored = save_topic(create)
        topic = Topic.model_validate({k: v for k, v in stored.items() if k != "path"})
        created_topics.append(topic)
        title_to_id[title.casefold()] = topic.id

    created_dependencies: list[Dependency] = []
    skipped_dependencies: list[SkippedDependency] = []

    for row in raw_deps:
        if not isinstance(row, dict):
            continue
        from_title = str(row.get("from", "")).strip()
        to_title = str(row.get("to", "")).strip()
        from_id = title_to_id.get(from_title.casefold())
        to_id = title_to_id.get(to_title.casefold())
        if from_id is None or to_id is None:
            skipped_dependencies.append(
                SkippedDependency(from_title=from_title, to_title=to_title, reason="unknown topic reference"),
            )
            continue
        try:
            payload = add_dependency(DependencyCreate(from_topic_id=from_id, to_topic_id=to_id))
        except (DependencyCycleError, ValueError) as exc:
            skipped_dependencies.append(SkippedDependency(from_title=from_title, to_title=to_title, reason=str(exc)))
            continue
        created_dependencies.append(Dependency.model_validate(payload))

    logger.info(
        "Roadmap generated: topics=%s dependencies=%s skipped=%s",
        len(created_topics),
        len(created_dependencies),
        len(skipped_dependencies),
    )
    return GenerateRoadmapResponse(
        created_topics=created_topics,
        created_dependencies=created_dependencies,
        skipped_dependencies=skipped_dependencies,
        errors=errors,
    )
