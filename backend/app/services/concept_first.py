"""Concept-First experimental graph generation pipeline (EVALUATION-ONLY).

Learning goal
  → Stage 1: candidate concept generation (LLM)
  → Stage 2: deterministic concept normalization
  → Stage 3: dependency generation constrained to inventory (LLM)
  → Existing Synapse DAG validation (``build_topics_and_dependencies``)

Closed experiment — not accepted by product ingest. Use evaluation adapters
(``resolve_evaluation_generation_strategy`` / ``run_concept_first``) for historical
reproducibility only. Production default remains baseline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from app.prompts.concept_first import (
    build_concept_generation_prompt,
    build_dependency_generation_prompt,
    concept_first_prompt_metadata,
)
from app.services.concept_normalization import (
    CandidateConcept,
    NormalizationResult,
    normalize_concepts,
)
from app.services.inventory_pruning import PruneResult, prune_inventory
from app.services.llm import call_llm_detailed, llm_operation
from app.services.proposal_common import (
    build_topics_and_dependencies,
    parse_llm_json_object,
    review_confidence_threshold,
)

@dataclass
class ConceptFirstStageTimings:
    concept_generation_ms: float = 0.0
    normalization_ms: float = 0.0
    dependency_generation_ms: float = 0.0
    validation_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return (
            self.concept_generation_ms
            + self.normalization_ms
            + self.dependency_generation_ms
            + self.validation_ms
        )


@dataclass
class ConceptFirstResult:
    """Structured output of the Concept-First pipeline (eval + ingest)."""

    topics: list[dict[str, Any]] = field(default_factory=list)
    dependencies: list[dict[str, str]] = field(default_factory=list)
    skipped_dependencies: list[Any] = field(default_factory=list)
    candidate_concepts: list[dict[str, Any]] = field(default_factory=list)
    normalization: NormalizationResult | None = None
    pruning: PruneResult | None = None
    timings: ConceptFirstStageTimings = field(default_factory=ConceptFirstStageTimings)
    input_tokens: int = 0
    output_tokens: int = 0
    tokens_estimated: bool = True
    cost_usd: float | None = None
    model: str | None = None
    provider: str | None = None
    parse_ok: bool = True
    status: str = "ok"  # ok | partial
    semantic_analysis: str = "available"  # available | unavailable
    errors: list[str] = field(default_factory=list)
    concept_raw: str = ""
    dependency_raw: str = ""
    prompt_meta: dict[str, str] = field(default_factory=dict)
    enable_pruning: bool = False

    def inventory_titles(self) -> list[str]:
        if self.pruning is not None and self.enable_pruning:
            return list(self.pruning.kept_concepts)
        if self.normalization is None:
            return []
        return [c.title for c in self.normalization.inventory]

    def to_meta(self) -> dict[str, Any]:
        norm = self.normalization.to_dict() if self.normalization else None
        prune = self.pruning.to_dict() if self.pruning else None
        strategy = "concept_first_pruned" if self.enable_pruning else "concept_first"
        return {
            **self.prompt_meta,
            "generation_strategy": strategy,
            "enable_pruning": self.enable_pruning,
            "status": self.status,
            "semantic_analysis": self.semantic_analysis,
            "errors": list(self.errors),
            "candidate_concepts": list(self.candidate_concepts),
            "normalized_inventory": (
                [c.title for c in self.normalization.inventory] if self.normalization else []
            ),
            "pruned_inventory": list(self.pruning.kept_concepts) if self.pruning else None,
            "normalization": norm,
            "pruning": prune,
            "stage_latency_ms": {
                "concept_generation": self.timings.concept_generation_ms,
                "normalization": self.timings.normalization_ms,
                "dependency_generation": self.timings.dependency_generation_ms,
                "validation": self.timings.validation_ms,
                "total": self.timings.total_ms,
            },
            "llm_latency_ms": (
                self.timings.concept_generation_ms + self.timings.dependency_generation_ms
            ),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "tokens_estimated": self.tokens_estimated,
            "cost_usd": self.cost_usd,
            "model": self.model,
            "provider": self.provider,
            "concept_raw": self.concept_raw,
            "dependency_raw": self.dependency_raw,
        }

def _parse_concepts(raw: str) -> list[CandidateConcept]:
    data = parse_llm_json_object(raw)
    rows = data.get("concepts")
    if not isinstance(rows, list) or not rows:
        # Tolerate accidental topics-shaped output without inventing deps.
        rows = data.get("topics")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Concept generation did not include a non-empty 'concepts' list")
    out: list[CandidateConcept] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        title = str(row.get("title", "")).strip()
        if not title:
            continue
        out.append(
            CandidateConcept(
                title=title,
                description=str(row.get("description") or row.get("summary") or "").strip(),
                reason=str(row.get("reason") or "").strip(),
            )
        )
    if not out:
        raise ValueError("Concept generation contained no well-formed concepts")
    return out


def _parse_dependencies(raw: str) -> list[dict[str, str]]:
    data = parse_llm_json_object(raw)
    rows = data.get("dependencies")
    if not isinstance(rows, list):
        raise ValueError("Dependency generation did not include a 'dependencies' list")
    out: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        frm = str(row.get("from", "")).strip()
        to = str(row.get("to", "")).strip()
        if frm and to:
            out.append({"from": frm, "to": to})
    return out


def _filter_deps_to_inventory(
    deps: list[dict[str, str]],
    inventory_titles: list[str],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Drop edges that introduce titles outside the inventory (case-sensitive match)."""
    allowed = set(inventory_titles)
    kept: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []
    for d in deps:
        if d["from"] in allowed and d["to"] in allowed:
            kept.append(d)
        else:
            rejected.append({**d, "reason": "new or unknown concept title outside inventory"})
    return kept, rejected


def _accumulate_usage(result: ConceptFirstResult, record: Any) -> None:
    result.input_tokens += int(getattr(record, "input_tokens", 0) or 0)
    result.output_tokens += int(getattr(record, "output_tokens", 0) or 0)
    result.tokens_estimated = bool(
        result.tokens_estimated and getattr(record, "tokens_estimated", True)
    )
    cost = getattr(record, "estimated_cost_usd", None)
    if cost is not None:
        result.cost_usd = (result.cost_usd or 0.0) + float(cost)
    if result.model is None:
        result.model = getattr(record, "model", None)
    if result.provider is None:
        result.provider = getattr(record, "provider", None)


def _objective_from_source(source_text: str) -> str:
    for line in source_text.splitlines():
        if line.casefold().startswith("goal:"):
            return line.split(":", 1)[1].strip()
    return source_text.strip()


async def run_concept_first_pipeline(
    source_text: str,
    *,
    temperature: float = 0.0,
    seed: int | None = 42,
    enable_pruning: bool = False,
    prune_config_name: str = "combined_conservative",
) -> ConceptFirstResult:
    """Run the Concept-First pipeline.

    When ``enable_pruning`` is True (strategy ``concept_first_pruned``), applies
    deterministic inventory pruning after normalization and before dependency generation.
    Default Concept-First leaves the normalized inventory unchanged.
    """
    result = ConceptFirstResult(
        prompt_meta=concept_first_prompt_metadata(),
        enable_pruning=enable_pruning,
    )
    if enable_pruning:
        result.prompt_meta = {
            **result.prompt_meta,
            "generation_strategy": "concept_first_pruned",
            "prune_config": prune_config_name,
        }
    if not source_text.strip():
        result.parse_ok = False
        result.status = "partial"
        result.semantic_analysis = "unavailable"
        result.errors.append("empty source text")
        return result

    # --- Stage 1: concept generation ---
    concept_prompt = build_concept_generation_prompt(source_text)
    try:
        with llm_operation("concept_first_concepts"):
            t0 = time.perf_counter()
            concept_record = await call_llm_detailed(
                concept_prompt,
                temperature=temperature,
                seed=seed,
            )
            result.timings.concept_generation_ms = (time.perf_counter() - t0) * 1000.0
        result.concept_raw = concept_record.text
        _accumulate_usage(result, concept_record)
        candidates = _parse_concepts(concept_record.text)
    except Exception as exc:
        result.parse_ok = False
        result.status = "partial"
        result.semantic_analysis = "unavailable"
        result.errors.append(f"concept generation failed: {exc}")
        return result

    result.candidate_concepts = [
        {"title": c.title, "description": c.description, "reason": c.reason} for c in candidates
    ]

    # --- Stage 2: deterministic normalization ---
    t1 = time.perf_counter()
    norm = normalize_concepts(candidates)
    result.timings.normalization_ms = (time.perf_counter() - t1) * 1000.0
    result.normalization = norm
    inventory = list(norm.inventory)
    if not inventory:
        result.parse_ok = False
        result.status = "partial"
        result.semantic_analysis = "unavailable"
        result.errors.append("normalization produced an empty concept inventory")
        return result

    # --- Optional Stage 2b: deterministic inventory pruning (explicit opt-in) ---
    by_title = {c.title: c for c in inventory}
    titles = [c.title for c in inventory]
    if enable_pruning:
        prune = prune_inventory(
            titles,
            _objective_from_source(source_text),
            config_name=prune_config_name,
        )
        result.pruning = prune
        titles = list(prune.kept_concepts)
        if prune.fallback_to_original_inventory:
            result.errors.append("pruning FALLBACK_TO_ORIGINAL_INVENTORY")
        inventory = [by_title[t] for t in titles if t in by_title]
        # Titles kept that somehow aren't in by_title (shouldn't happen)
        for t in titles:
            if t not in by_title:
                inventory.append(CandidateConcept(title=t))

    raw_topics = [
        {
            "title": c.title,
            "summary": c.description or c.reason or f"Concept: {c.title}",
            "confidence": 0.7,
        }
        for c in inventory
    ]

    # --- Stage 3: dependency generation constrained to inventory ---
    dep_prompt = build_dependency_generation_prompt(source_text, titles)
    try:
        with llm_operation("concept_first_dependencies"):
            t2 = time.perf_counter()
            dep_record = await call_llm_detailed(
                dep_prompt,
                temperature=temperature,
                seed=None if seed is None else seed + 17,
            )
            result.timings.dependency_generation_ms = (time.perf_counter() - t2) * 1000.0
        result.dependency_raw = dep_record.text
        _accumulate_usage(result, dep_record)
        raw_deps = _parse_dependencies(dep_record.text)
    except Exception as exc:
        result.status = "partial"
        result.semantic_analysis = "unavailable"
        result.errors.append(f"dependency generation failed: {exc}")
        result.topics = raw_topics
        result.dependencies = []
        # Still expose inventory; do not claim a complete graph.
        result.parse_ok = True
        return result

    kept_deps, rejected_new = _filter_deps_to_inventory(raw_deps, titles)
    for r in rejected_new:
        result.errors.append(
            f"dropped dependency outside inventory: {r.get('from')} → {r.get('to')}"
        )

    # --- Existing Synapse DAG validation ---
    t3 = time.perf_counter()
    proposed_topics, proposed_dependencies, skipped = build_topics_and_dependencies(
        raw_topics,
        kept_deps,
        confidence_threshold=review_confidence_threshold(),
    )
    result.timings.validation_ms = (time.perf_counter() - t3) * 1000.0

    id_to_title = {t.temp_id: t.title for t in proposed_topics}
    deps: list[dict[str, str]] = []
    for d in proposed_dependencies:
        frm = id_to_title.get(d.from_temp_id)
        to = id_to_title.get(d.to_temp_id)
        if frm and to:
            deps.append({"from": frm, "to": to})

    result.topics = [
        {
            "title": t.title,
            "summary": t.summary,
            "confidence": t.confidence,
            "temp_id": t.temp_id,
            "needs_review": t.needs_review,
        }
        for t in proposed_topics
    ]
    result.dependencies = deps
    result.skipped_dependencies = skipped
    for s in skipped:
        result.errors.append(f"skipped dependency: {s.from_title} → {s.to_title} ({s.reason})")
    if rejected_new:
        result.status = "partial" if result.status == "ok" else result.status
    result.parse_ok = True
    return result
