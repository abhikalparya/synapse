"""Domain-prior + closed-world edge classification pipeline (experimental).

Reuses domain inventory concept selection, then classifies directed candidate pairs
instead of free-form dependency generation.

Production default remains baseline. Inventories are not modified.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.curriculum.edge_candidates import (
    ClassificationParseResult,
    batch_candidate_pairs,
    generate_candidate_pairs,
    merge_classification_results,
    parse_classification_response,
)
from app.curriculum.inventory import (
    DEFAULT_CURRICULUM_DIR,
    load_domain_inventory,
    load_experiment_config,
)
from app.curriculum.selection import (
    SelectedConcept,
    concepts_to_topic_dicts,
    parse_and_validate_selection,
)
from app.evaluation.cost import estimate_cost_usd
from app.prompts.domain_curriculum_prior import build_selection_prompt
from app.prompts.domain_prior_edge_classifier import (
    build_edge_classification_prompt,
    edge_classifier_metadata,
    resolve_edge_classifier_prompt_variant,
)
from app.services.llm import call_llm_detailed, llm_operation
from app.services.proposal_common import (
    build_topics_and_dependencies,
    review_confidence_threshold,
)


def _accumulate(result: EdgeClassifierResult, record: Any) -> None:
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


@dataclass
class EdgeClassifierTimings:
    selection_ms: float = 0.0
    classification_ms: float = 0.0
    validation_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.selection_ms + self.classification_ms + self.validation_ms


@dataclass
class EdgeClassifierResult:
    topics: list[dict[str, Any]] = field(default_factory=list)
    dependencies: list[dict[str, str]] = field(default_factory=list)
    skipped_dependencies: list[Any] = field(default_factory=list)
    domain: str = ""
    inventory_version: str = ""
    inventory_size: int = 0
    selection_raw: str = ""
    classification_raw_batches: list[str] = field(default_factory=list)
    candidate_meta: dict[str, Any] = field(default_factory=dict)
    classification: ClassificationParseResult | None = None
    timings: EdgeClassifierTimings = field(default_factory=EdgeClassifierTimings)
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
    prompt_meta: dict[str, str] = field(default_factory=dict)
    new_concept_count: int = 0
    batch_count: int = 0
    pairs_per_batch: int = 0
    candidate_required_edges: int = 0
    accepted_required_edges: int = 0
    cycle_rejected_edges: int = 0
    selected_titles: list[str] = field(default_factory=list)
    selected_ids: list[str] = field(default_factory=list)
    pair_decisions: list[dict[str, str]] = field(default_factory=list)
    prompt_variant: str = "edge_classifier_baseline"
    selection_input_tokens: int = 0
    selection_output_tokens: int = 0
    classification_input_tokens: int = 0
    classification_output_tokens: int = 0

    def to_meta(self) -> dict[str, Any]:
        cls = self.classification
        return {
            **self.prompt_meta,
            "generation_strategy": "domain_prior_edge_classifier",
            "prompt_variant": self.prompt_variant,
            "edge_classifier_prompt_variant": self.prompt_variant,
            "domain": self.domain,
            "inventory_version": self.inventory_version,
            "inventory_size": self.inventory_size,
            "selected_concept_ids": list(self.selected_ids),
            "selected_titles": list(self.selected_titles),
            "selected_concept_count": len(self.selected_ids),
            "candidate_meta": dict(self.candidate_meta),
            "batch_count": self.batch_count,
            "pairs_per_batch": self.pairs_per_batch,
            "predicted_required_pair_count": self.candidate_required_edges,
            "accepted_required_edges": self.accepted_required_edges,
            "cycle_rejected_edges": self.cycle_rejected_edges,
            "uncertain_count": cls.uncertain_count if cls else 0,
            "unknown_id_rate_inputs": len(cls.rejected_unknown_ids) if cls else 0,
            "rejected_non_candidate_count": len(cls.rejected_non_candidate) if cls else 0,
            "duplicate_decision_count": cls.duplicate_decision_count if cls else 0,
            "new_concept_count": self.new_concept_count,
            "pair_decisions": list(self.pair_decisions),
            "status": self.status,
            "errors": list(self.errors),
            "stage_latency_ms": {
                "selection": self.timings.selection_ms,
                "edge_classification": self.timings.classification_ms,
                "validation": self.timings.validation_ms,
                "total": self.timings.total_ms,
            },
            "llm_latency_ms": self.timings.selection_ms + self.timings.classification_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "selection_input_tokens": self.selection_input_tokens,
            "selection_output_tokens": self.selection_output_tokens,
            "classification_input_tokens": self.classification_input_tokens,
            "classification_output_tokens": self.classification_output_tokens,
            "tokens_estimated": self.tokens_estimated,
            "cost_usd": self.cost_usd,
            "estimated_cost_usd": self.estimated_cost_usd,
            "model": self.model,
            "provider": self.provider,
        }


def _estimate(result: EdgeClassifierResult) -> None:
    if result.cost_usd is not None:
        result.estimated_cost_usd = result.cost_usd
        return
    model = result.model or "gpt-4o-mini"
    est = estimate_cost_usd(model, result.input_tokens, result.output_tokens)
    result.estimated_cost_usd = est


def _finalize_graph(
    result: EdgeClassifierResult,
    *,
    topic_dicts: list[dict[str, Any]],
    classification: ClassificationParseResult,
    id_to_title: dict[str, str],
    inventory,
) -> None:
    result.classification = classification
    result.pair_decisions = [
        {
            "from_id": d.from_id,
            "to_id": d.to_id,
            "from_title": id_to_title.get(d.from_id, ""),
            "to_title": id_to_title.get(d.to_id, ""),
            "decision": d.decision,
        }
        for d in classification.decisions
    ]
    result.candidate_required_edges = len(classification.required_edges)
    raw_deps = [
        {"from": frm, "to": to, "confidence": 0.75}
        for frm, to in classification.required_edges
    ]
    t2 = time.perf_counter()
    proposed_topics, proposed_deps, skipped = build_topics_and_dependencies(
        topic_dicts,
        raw_deps,
        confidence_threshold=review_confidence_threshold(),
    )
    result.timings.validation_ms = (time.perf_counter() - t2) * 1000.0
    id_map = {t.temp_id: t.title for t in proposed_topics}
    accepted: list[dict[str, str]] = []
    for d in proposed_deps:
        frm = id_map.get(d.from_temp_id)
        to = id_map.get(d.to_temp_id)
        if frm and to:
            accepted.append({"from": frm, "to": to})
    result.dependencies = accepted
    result.accepted_required_edges = len(accepted)
    result.cycle_rejected_edges = sum(
        1
        for s in skipped
        if "cycle" in str(getattr(s, "reason", "") or "").casefold()
    )
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
    inv_titles = {c.title.casefold() for c in inventory.concepts}
    invented = [t["title"] for t in result.topics if t["title"].casefold() not in inv_titles]
    result.new_concept_count = len(invented)
    if invented:
        result.errors.append(f"invented titles leaked: {invented}")
        result.parse_ok = False
        result.status = "partial"
    if classification.rejected_unknown_ids:
        result.errors.append(
            f"rejected unknown ids: {sorted(set(classification.rejected_unknown_ids))}"
        )
    if classification.rejected_non_candidate:
        result.errors.append(
            f"rejected non-candidate pairs: {len(classification.rejected_non_candidate)}"
        )
    _estimate(result)


async def _classify_batches(
    result: EdgeClassifierResult,
    *,
    goal: str,
    selected: list[SelectedConcept],
    descriptions: dict[str, str],
    pairs: list,
    pairs_per_batch: int,
    temperature: float,
    seed: int | None,
    resolved_variant: str,
) -> list[ClassificationParseResult]:
    batches = batch_candidate_pairs(pairs, pairs_per_batch=pairs_per_batch)
    result.batch_count = len(batches)
    id_to_title = {s.concept_id: s.title for s in selected}
    parse_parts: list[ClassificationParseResult] = []
    with llm_operation("domain_prior_edge_classifier_classify"):
        t1 = time.perf_counter()
        for bi, batch in enumerate(batches):
            prompt = build_edge_classification_prompt(
                goal,
                selected,
                batch,
                concept_descriptions=descriptions,
                variant=resolved_variant,
            )
            before_in, before_out = result.input_tokens, result.output_tokens
            rec = await call_llm_detailed(
                prompt,
                temperature=temperature,
                seed=None if seed is None else seed + 1 + bi,
            )
            result.classification_raw_batches.append(rec.text)
            _accumulate(result, rec)
            result.classification_input_tokens += result.input_tokens - before_in
            result.classification_output_tokens += result.output_tokens - before_out
            parse_parts.append(
                parse_classification_response(rec.text, batch, id_to_title=id_to_title)
            )
        result.timings.classification_ms = (time.perf_counter() - t1) * 1000.0
    return parse_parts


async def run_domain_prior_edge_classifier_pipeline(
    source_text: str,
    *,
    domain: str,
    curriculum_dir: str | Path | None = None,
    temperature: float = 0.0,
    seed: int | None = 42,
    prompt_variant: str | None = None,
) -> EdgeClassifierResult:
    root = Path(curriculum_dir) if curriculum_dir else DEFAULT_CURRICULUM_DIR
    cfg = load_experiment_config(root)
    max_selected = int(cfg.get("max_selected_concepts") or 8)
    max_required = int(cfg.get("max_required_concepts") or 8)
    edge_cfg = cfg.get("edge_classifier") or {}
    pairs_per_batch = int(edge_cfg.get("pairs_per_batch") or 64)
    max_pairs = edge_cfg.get("max_candidate_pairs")
    max_candidate_pairs = int(max_pairs) if max_pairs is not None else None
    resolved_variant = resolve_edge_classifier_prompt_variant(prompt_variant)

    inventory = load_domain_inventory(domain, curriculum_dir=root)
    result = EdgeClassifierResult(
        domain=domain,
        inventory_version=inventory.version,
        inventory_size=inventory.size(),
        prompt_meta=edge_classifier_metadata(
            domain, inventory.version, variant=resolved_variant
        ),
        pairs_per_batch=pairs_per_batch,
        prompt_variant=resolved_variant,
    )
    if not source_text.strip():
        result.parse_ok = False
        result.status = "partial"
        result.errors.append("empty source text")
        return result

    goal = _goal_from_source(source_text)
    selection_prompt = build_selection_prompt(goal, inventory, max_required=max_required)
    try:
        with llm_operation("domain_prior_edge_classifier_selection"):
            t0 = time.perf_counter()
            sel_record = await call_llm_detailed(
                selection_prompt, temperature=temperature, seed=seed
            )
            result.timings.selection_ms = (time.perf_counter() - t0) * 1000.0
        result.selection_raw = sel_record.text
        before_in, before_out = result.input_tokens, result.output_tokens
        _accumulate(result, sel_record)
        result.selection_input_tokens = result.input_tokens - before_in
        result.selection_output_tokens = result.output_tokens - before_out
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
    if not selection.selected:
        result.parse_ok = False
        result.status = "partial"
        result.errors.append("no valid REQUIRED concepts selected from inventory")
        return result

    result.selected_ids = [s.concept_id for s in selection.selected]
    result.selected_titles = [s.title for s in selection.selected]
    topic_dicts = concepts_to_topic_dicts(selection.selected, inventory)
    id_to_title = {s.concept_id: s.title for s in selection.selected}
    descriptions = {
        s.concept_id: inventory.by_id()[s.concept_id].description for s in selection.selected
    }
    pairs, cand_meta = generate_candidate_pairs(
        selection.selected, max_candidate_pairs=max_candidate_pairs
    )
    result.candidate_meta = cand_meta
    if cand_meta.get("truncated"):
        result.errors.append(
            f"candidate space truncated: evaluated={cand_meta['candidate_pairs_evaluated']} "
            f"omitted={cand_meta['candidate_pairs_omitted']}"
        )

    try:
        parse_parts = await _classify_batches(
            result,
            goal=goal,
            selected=selection.selected,
            descriptions=descriptions,
            pairs=pairs,
            pairs_per_batch=pairs_per_batch,
            temperature=temperature,
            seed=seed,
            resolved_variant=resolved_variant,
        )
    except Exception as exc:
        result.parse_ok = False
        result.status = "partial"
        result.errors.append(f"edge classification failed: {exc}")
        result.topics = topic_dicts
        _estimate(result)
        return result

    classification = merge_classification_results(parse_parts)
    _finalize_graph(
        result,
        topic_dicts=topic_dicts,
        classification=classification,
        id_to_title=id_to_title,
        inventory=inventory,
    )
    return result


async def classify_with_frozen_selection(
    source_text: str,
    *,
    domain: str,
    selected_ids: list[str],
    selection_raw: str,
    selection_ms: float,
    selection_input_tokens: int,
    selection_output_tokens: int,
    model: str | None,
    provider: str | None,
    curriculum_dir: str | Path | None = None,
    temperature: float = 0.0,
    seed: int | None = 42,
    prompt_variant: str | None = None,
) -> EdgeClassifierResult:
    """Classify pairs for a frozen inventory selection (no second selection LLM call)."""
    root = Path(curriculum_dir) if curriculum_dir else DEFAULT_CURRICULUM_DIR
    cfg = load_experiment_config(root)
    edge_cfg = cfg.get("edge_classifier") or {}
    pairs_per_batch = int(edge_cfg.get("pairs_per_batch") or 64)
    max_pairs = edge_cfg.get("max_candidate_pairs")
    max_candidate_pairs = int(max_pairs) if max_pairs is not None else None
    resolved_variant = resolve_edge_classifier_prompt_variant(prompt_variant)

    inventory = load_domain_inventory(domain, curriculum_dir=root)
    by_id = inventory.by_id()
    selected = [
        SelectedConcept(concept_id=cid, title=by_id[cid].title, kind="REQUIRED")
        for cid in selected_ids
        if cid in by_id
    ]
    result = EdgeClassifierResult(
        domain=domain,
        inventory_version=inventory.version,
        inventory_size=inventory.size(),
        prompt_meta=edge_classifier_metadata(
            domain, inventory.version, variant=resolved_variant
        ),
        pairs_per_batch=pairs_per_batch,
        prompt_variant=resolved_variant,
        selection_raw=selection_raw,
        selected_ids=[s.concept_id for s in selected],
        selected_titles=[s.title for s in selected],
        model=model,
        provider=provider,
        selection_input_tokens=selection_input_tokens,
        selection_output_tokens=selection_output_tokens,
        input_tokens=selection_input_tokens,
        output_tokens=selection_output_tokens,
    )
    result.prompt_meta["selection_shared"] = "true"
    result.timings.selection_ms = selection_ms
    if not selected:
        result.parse_ok = False
        result.status = "partial"
        result.errors.append("frozen selection empty")
        return result

    goal = _goal_from_source(source_text)
    topic_dicts = concepts_to_topic_dicts(selected, inventory)
    id_to_title = {s.concept_id: s.title for s in selected}
    descriptions = {s.concept_id: by_id[s.concept_id].description for s in selected}
    pairs, cand_meta = generate_candidate_pairs(
        selected, max_candidate_pairs=max_candidate_pairs
    )
    result.candidate_meta = cand_meta

    try:
        parse_parts = await _classify_batches(
            result,
            goal=goal,
            selected=selected,
            descriptions=descriptions,
            pairs=pairs,
            pairs_per_batch=pairs_per_batch,
            temperature=temperature,
            seed=seed,
            resolved_variant=resolved_variant,
        )
    except Exception as exc:
        result.parse_ok = False
        result.status = "partial"
        result.errors.append(f"edge classification failed: {exc}")
        result.topics = topic_dicts
        _estimate(result)
        return result

    classification = merge_classification_results(parse_parts)
    _finalize_graph(
        result,
        topic_dicts=topic_dicts,
        classification=classification,
        id_to_title=id_to_title,
        inventory=inventory,
    )
    return result
