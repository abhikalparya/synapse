"""Baseline graph generators: linear roadmap and direct LLM dependency graph.

Neither baseline runs Synapse's proposal validation / DAG filter. The direct baseline
intentionally keeps self-loops, cycles, and unknown-title edges so validity metrics
reflect raw LLM output.
"""

from __future__ import annotations

import re
from typing import Any

from app.evaluation.failure_analysis import classify_llm_exception
from app.evaluation.schemas import EvalExample, GeneratedGraph
from app.services.llm import call_llm_detailed, llm_operation
from app.services.proposal_common import parse_llm_json_object, strip_json_fences

LINEAR_ROADMAP_PROMPT = """You are a curriculum-design assistant. Given the learning goal below, \
produce a simple ordered learning roadmap as a numbered list.

Output rules:
- Respond with ONLY a numbered list (1. 2. 3. …). No JSON, no markdown fences, no commentary.
- Each line is one short topic title.
- Order topics from foundational prerequisites first to the final goal last.
- Use 4-8 topics for a focused goal.
- Do not invent credentials, URLs, or personal advice.

Goal / content:
---
{source}
---
"""

_NUMBERED_RE = re.compile(r"^\s*(?:\d+[\.)]|[-*•])\s+(.+?)\s*$")


def build_source_text(example: EvalExample) -> str:
    parts = [f"Goal: {example.goal.strip()}"]
    if example.input_notes and example.input_notes.strip():
        parts.append(f"Notes:\n{example.input_notes.strip()}")
    return "\n\n".join(parts)


def parse_linear_roadmap(text: str) -> list[str]:
    cleaned = strip_json_fences(text)
    topics: list[str] = []
    for line in cleaned.splitlines():
        m = _NUMBERED_RE.match(line)
        if not m:
            continue
        title = m.group(1).strip().strip("*_`# ")
        if title:
            topics.append(title)
    if not topics:
        raise ValueError("Could not parse a numbered/bulleted roadmap from LLM output")
    return topics


def linear_topics_to_dependencies(topics: list[str]) -> list[tuple[str, str]]:
    """Convert ordered roadmap A, B, C into Synapse edges B→A, C→B (later requires earlier)."""
    deps: list[tuple[str, str]] = []
    for i in range(1, len(topics)):
        deps.append((topics[i], topics[i - 1]))
    return deps


def parse_direct_dependency_graph(raw: str) -> GeneratedGraph:
    """Parse ingest-shaped JSON without DAG validation or unknown-title filtering."""
    data = parse_llm_json_object(raw)
    raw_topics = data.get("topics")
    raw_deps = data.get("dependencies")
    if not isinstance(raw_topics, list) or not raw_topics:
        raise ValueError("LLM response did not include a non-empty 'topics' list")
    if not isinstance(raw_deps, list):
        raw_deps = []

    topics: list[str] = []
    confidences: list[float] = []
    for row in raw_topics:
        if not isinstance(row, dict):
            continue
        title = str(row.get("title", "")).strip()
        if not title:
            continue
        topics.append(title)
        try:
            confidences.append(max(0.0, min(1.0, float(row.get("confidence", 0.5)))))
        except (TypeError, ValueError):
            confidences.append(0.5)

    if not topics:
        raise ValueError("LLM response contained no well-formed topics")

    dependencies: list[tuple[str, str]] = []
    for row in raw_deps:
        if not isinstance(row, dict):
            continue
        frm = str(row.get("from", "")).strip()
        to = str(row.get("to", "")).strip()
        if not frm or not to:
            continue
        dependencies.append((frm, to))

    return GeneratedGraph(
        topics=topics,
        dependencies=dependencies,
        topic_confidences=confidences,
        raw_response=raw,
        parse_ok=True,
    )


async def run_linear_baseline(
    example: EvalExample,
    *,
    temperature: float = 0.0,
    seed: int | None = 42,
) -> tuple[GeneratedGraph, dict[str, Any]]:
    """Baseline A: ordered roadmap -> chain DAG. No Synapse validation."""
    prompt = LINEAR_ROADMAP_PROMPT.format(source=build_source_text(example))
    try:
        with llm_operation("eval_linear"):
            record = await call_llm_detailed(prompt, temperature=temperature, seed=seed)
        topics = parse_linear_roadmap(record.text)
        graph = GeneratedGraph(
            topics=topics,
            dependencies=linear_topics_to_dependencies(topics),
            raw_response=record.text,
            parse_ok=True,
        )
        meta = {
            "llm_latency_ms": record.latency_ms,
            "input_tokens": record.input_tokens,
            "output_tokens": record.output_tokens,
            "tokens_estimated": record.tokens_estimated,
            "cost_usd": record.estimated_cost_usd,
            "model": record.model,
            "provider": record.provider,
        }
        return graph, meta
    except Exception as exc:
        category = classify_llm_exception(exc)
        graph = GeneratedGraph(
            topics=[],
            dependencies=[],
            parse_ok=False,
            error=str(exc),
            error_category=category,
        )
        return graph, {
            "llm_latency_ms": 0.0,
            "input_tokens": None,
            "output_tokens": None,
            "tokens_estimated": True,
            "cost_usd": None,
            "error": str(exc),
            "error_category": category,
        }


async def generate_direct_llm_raw(
    example: EvalExample,
    *,
    temperature: float = 0.0,
    seed: int | None = 42,
    prompt_variant: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Shared LLM call for Direct and Synapse (equivalent inputs; post-processing differs)."""
    from app.prompts.ingest import build_ingest_prompt, prompt_metadata

    meta_prompt = prompt_metadata(prompt_variant)
    prompt = build_ingest_prompt(
        build_source_text(example),
        known_topic_titles=[],
        variant=prompt_variant,
    )
    with llm_operation("eval_graph_json"):
        record = await call_llm_detailed(prompt, temperature=temperature, seed=seed)
    meta = {
        "llm_latency_ms": record.latency_ms,
        "input_tokens": record.input_tokens,
        "output_tokens": record.output_tokens,
        "tokens_estimated": record.tokens_estimated,
        "cost_usd": record.estimated_cost_usd,
        "model": record.model,
        "provider": record.provider,
        "raw": record.text,
        **meta_prompt,
    }
    return record.text, meta


def run_direct_from_raw(raw: str) -> GeneratedGraph:
    try:
        return parse_direct_dependency_graph(raw)
    except Exception as exc:
        return GeneratedGraph(
            topics=[],
            dependencies=[],
            raw_response=raw,
            parse_ok=False,
            error=str(exc),
            error_category=classify_llm_exception(exc),
        )
