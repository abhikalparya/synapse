"""Domain curriculum prior generation pipeline (experimental, opt-in).

goal → domain inventory → closed-world concept selection → dependency generation
→ existing DAG validation (build_topics_and_dependencies)

Production default remains baseline. NEW_CONCEPT_COUNT relative to inventory must be 0.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.evaluation.cost import estimate_cost_usd
from app.curriculum.inventory import (
    DEFAULT_CURRICULUM_DIR,
    DomainInventory,
    load_domain_inventory,
    load_experiment_config,
    resolve_domain_for_case,
)
from app.curriculum.selection import (
    SelectionResult,
    concepts_to_topic_dicts,
    parse_and_validate_selection,
)
from app.prompts.domain_curriculum_prior import (
    build_dependency_prompt,
    build_selection_prompt,
    domain_curriculum_prior_metadata,
)
from app.services.llm import call_llm_detailed, llm_operation
from app.services.proposal_common import (
    build_topics_and_dependencies,
    parse_llm_json_object,
    review_confidence_threshold,
)


@dataclass
class CurriculumPriorTimings:
    selection_ms: float = 0.0
    dependency_ms: float = 0.0
    validation_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.selection_ms + self.dependency_ms + self.validation_ms


@dataclass
class CurriculumPriorResult:
    topics: list[dict[str, Any]] = field(default_factory=list)
    dependencies: list[dict[str, str]] = field(default_factory=list)
    skipped_dependencies: list[Any] = field(default_factory=list)
    domain: str = ""
    inventory_version: str = ""
    inventory_hash: str = ""
    inventory_size: int = 0
    selection: SelectionResult | None = None
    timings: CurriculumPriorTimings = field(default_factory=CurriculumPriorTimings)
    input_tokens: int = 0
    output_tokens: int = 0
    tokens_estimated: bool = True
    cost_usd: float | None = None
    estimated_cost_usd: float | None = None
    model: str | None = None
    provider: str | None = None
    parse_ok: bool = True
    status: str = "ok"
    errors: list[str] = field(default_factory=list)
    selection_raw: str = ""
    dependency_raw: str = ""
    prompt_meta: dict[str, str] = field(default_factory=dict)
    new_concept_count: int = 0
    fallback_reason: str | None = None

    def to_meta(self) -> dict[str, Any]:
        sel = self.selection
        if self.estimated_cost_usd is None:
            if self.cost_usd is not None:
                self.estimated_cost_usd = self.cost_usd
            else:
                self.estimated_cost_usd = estimate_cost_usd(
                    self.model or "gpt-4o-mini", self.input_tokens, self.output_tokens
                )
        return {
            **self.prompt_meta,
            "generation_strategy": "domain_curriculum_prior",
            "domain": self.domain,
            "inventory_version": self.inventory_version,
            "inventory_hash": self.inventory_hash,
            "inventory_size": self.inventory_size,
            "selected_concept_ids": [s.concept_id for s in (sel.selected if sel else [])],
            "selected_titles": [s.title for s in (sel.selected if sel else [])],
            "selected_concept_count": len(sel.selected) if sel else 0,
            "selection_unknown_id_count": sel.unknown_selection_count if sel else 0,
            "selection_invalid_id_count": (
                (sel.unknown_selection_count if sel else 0)
                + (sel.out_of_scope_selection_count if sel else 0)
            ),
            "unknown_selection_count": sel.unknown_selection_count if sel else 0,
            "out_of_scope_selection_count": sel.out_of_scope_selection_count if sel else 0,
            "rejected_arbitrary_titles": list(sel.rejected_arbitrary_titles) if sel else [],
            "selection_truncated": bool(sel.truncated) if sel else False,
            "new_concept_count": self.new_concept_count,
            "status": self.status,
            "errors": list(self.errors),
            "fallback_reason": self.fallback_reason,
            "selection_latency": self.timings.selection_ms,
            "dependency_latency": self.timings.dependency_ms,
            "total_latency": self.timings.total_ms,
            "stage_latency_ms": {
                "selection": self.timings.selection_ms,
                "dependency_generation": self.timings.dependency_ms,
                "validation": self.timings.validation_ms,
                "total": self.timings.total_ms,
            },
            "llm_latency_ms": self.timings.selection_ms + self.timings.dependency_ms,
            "estimated_input_tokens": self.input_tokens,
            "estimated_output_tokens": self.output_tokens,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "tokens_estimated": self.tokens_estimated,
            "cost_usd": self.cost_usd,
            "estimated_cost": self.estimated_cost_usd,
            "estimated_cost_usd": self.estimated_cost_usd,
            "model": self.model,
            "provider": self.provider,
        }


def _accumulate(result: CurriculumPriorResult, record: Any) -> None:
    result.input_tokens += int(getattr(record, "input_tokens", 0) or 0)
    result.output_tokens += int(getattr(record, "output_tokens", 0) or 0)
    result.tokens_estimated = bool(
        result.tokens_estimated and getattr(record, "tokens_estimated", True)
    )
    if result.model is None:
        result.model = getattr(record, "model", None)
    if result.provider is None:
        result.provider = getattr(record, "provider", None)
    cost = getattr(record, "cost_usd", None)
    if cost is not None:
        result.cost_usd = (result.cost_usd or 0.0) + float(cost)


def _goal_from_source(source_text: str) -> str:
    for line in source_text.splitlines():
        if line.casefold().startswith("goal:"):
            return line.split(":", 1)[1].strip()
    return source_text.strip()


def _parse_deps(raw: str, allowed_titles: set[str]) -> list[dict[str, Any]]:
    try:
        payload = parse_llm_json_object(raw)
    except Exception:
        return []
    rows = payload.get("dependencies") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    allowed_cf = {t.casefold(): t for t in allowed_titles}
    for row in rows:
        if not isinstance(row, dict):
            continue
        frm = str(row.get("from") or "").strip()
        to = str(row.get("to") or "").strip()
        if not frm or not to:
            continue
        frm_c = allowed_cf.get(frm.casefold())
        to_c = allowed_cf.get(to.casefold())
        if frm_c is None or to_c is None:
            continue
        try:
            conf = float(row.get("confidence", 0.7))
        except (TypeError, ValueError):
            conf = 0.7
        out.append({"from": frm_c, "to": to_c, "confidence": conf})
    return out


async def run_domain_curriculum_prior_pipeline(
    source_text: str,
    *,
    domain: str,
    curriculum_dir: str | Path | None = None,
    temperature: float = 0.0,
    seed: int | None = 42,
) -> CurriculumPriorResult:
    """Closed-world selection + constrained dependency generation."""
    root = Path(curriculum_dir) if curriculum_dir else DEFAULT_CURRICULUM_DIR
    cfg = load_experiment_config(root)
    max_selected = int(cfg.get("max_selected_concepts") or 8)
    max_required = int(cfg.get("max_required_concepts") or 8)

    inventory = load_domain_inventory(domain, curriculum_dir=root)
    result = CurriculumPriorResult(
        domain=domain,
        inventory_version=inventory.version,
        inventory_hash=inventory.content_hash,
        inventory_size=inventory.size(),
        prompt_meta=domain_curriculum_prior_metadata(domain, inventory.version),
    )
    if not source_text.strip():
        result.parse_ok = False
        result.status = "partial"
        result.errors.append("empty source text")
        return result

    goal = _goal_from_source(source_text)
    selection_prompt = build_selection_prompt(goal, inventory, max_required=max_required)
    try:
        with llm_operation("domain_curriculum_prior_selection"):
            t0 = time.perf_counter()
            sel_record = await call_llm_detailed(
                selection_prompt, temperature=temperature, seed=seed
            )
            result.timings.selection_ms = (time.perf_counter() - t0) * 1000.0
        result.selection_raw = sel_record.text
        _accumulate(result, sel_record)
    except Exception as exc:
        result.parse_ok = False
        result.status = "partial"
        result.errors.append(f"selection failed: {exc}")
        return result

    selection = parse_and_validate_selection(
        result.selection_raw,
        inventory,
        max_required=max_required,
        max_selected=max_selected,
    )
    result.selection = selection
    result.new_concept_count = 0  # enforced: only inventory IDs accepted
    if not selection.selected:
        result.parse_ok = False
        result.status = "partial"
        result.errors.append("no valid REQUIRED concepts selected from inventory")
        return result

    topic_dicts = concepts_to_topic_dicts(selection.selected, inventory)
    titles = [t["title"] for t in topic_dicts]
    by_id = inventory.by_id()
    details = [(s.title, by_id[s.concept_id].description) for s in selection.selected]
    dep_prompt = build_dependency_prompt(goal, titles, concept_details=details)
    try:
        with llm_operation("domain_curriculum_prior_dependencies"):
            t1 = time.perf_counter()
            dep_record = await call_llm_detailed(
                dep_prompt,
                temperature=temperature,
                seed=None if seed is None else seed + 1,
            )
            result.timings.dependency_ms = (time.perf_counter() - t1) * 1000.0
        result.dependency_raw = dep_record.text
        _accumulate(result, dep_record)
        raw_deps = _parse_deps(dep_record.text, set(titles))
    except Exception as exc:
        result.parse_ok = False
        result.status = "partial"
        result.errors.append(f"dependency generation failed: {exc}")
        # Still return selected topics without deps
        result.topics = topic_dicts
        return result

    t2 = time.perf_counter()
    proposed_topics, proposed_deps, skipped = build_topics_and_dependencies(
        topic_dicts,
        raw_deps,
        confidence_threshold=review_confidence_threshold(),
    )
    result.timings.validation_ms = (time.perf_counter() - t2) * 1000.0
    id_to_title = {t.temp_id: t.title for t in proposed_topics}
    result.dependencies = []
    for d in proposed_deps:
        frm = id_to_title.get(d.from_temp_id)
        to = id_to_title.get(d.to_temp_id)
        if frm and to:
            result.dependencies.append({"from": frm, "to": to})
    result.topics = [
        {
            "title": t.title,
            "summary": t.summary,
            "confidence": t.confidence,
            "temp_id": t.temp_id,
        }
        for t in proposed_topics
    ]
    result.skipped_dependencies = skipped
    # Sanity: every topic title must be an inventory title
    inv_titles = {c.title.casefold() for c in inventory.concepts}
    invented = [t["title"] for t in result.topics if t["title"].casefold() not in inv_titles]
    result.new_concept_count = len(invented)
    if invented:
        result.errors.append(f"invented titles leaked: {invented}")
        result.parse_ok = False
        result.status = "partial"
    return result
