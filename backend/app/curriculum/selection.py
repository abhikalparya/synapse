"""Closed-world concept selection parsing and ranking (no invention)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.curriculum.inventory import CurriculumConcept, DomainInventory
from app.services.proposal_common import parse_llm_json_object


ALLOWED_KINDS = frozenset({"REQUIRED", "RELEVANT_BUT_OPTIONAL", "IRRELEVANT", "OUT_OF_SCOPE"})


@dataclass
class SelectedConcept:
    concept_id: str
    title: str
    kind: str
    reason: str = ""
    confidence: float = 0.0


@dataclass
class SelectionResult:
    selected: list[SelectedConcept] = field(default_factory=list)
    rejected_unknown_ids: list[str] = field(default_factory=list)
    rejected_arbitrary_titles: list[str] = field(default_factory=list)
    rejected_out_of_scope: list[str] = field(default_factory=list)
    truncated: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def required_titles(self) -> list[str]:
        return [s.title for s in self.selected if s.kind == "REQUIRED"]

    @property
    def unknown_selection_count(self) -> int:
        return len(self.rejected_unknown_ids)

    @property
    def out_of_scope_selection_count(self) -> int:
        return len(self.rejected_out_of_scope)


def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return default


def parse_and_validate_selection(
    raw_text: str,
    inventory: DomainInventory,
    *,
    max_required: int,
    max_selected: int,
) -> SelectionResult:
    """Parse LLM JSON and keep only known inventory concept IDs (REQUIRED)."""
    result = SelectionResult()
    try:
        payload = parse_llm_json_object(raw_text)
    except Exception:
        result.raw = {}
        return result
    if not isinstance(payload, dict):
        return result
    result.raw = payload
    by_id = inventory.by_id()
    rows = payload.get("selected_concepts") or payload.get("selections") or []
    if not isinstance(rows, list):
        return result

    required: list[SelectedConcept] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        # Reject arbitrary title-only invention
        if "concept_id" not in row and row.get("title"):
            result.rejected_arbitrary_titles.append(str(row.get("title")))
            continue
        cid = str(row.get("concept_id") or "").strip()
        kind = str(row.get("kind") or row.get("role") or "REQUIRED").strip().upper()
        if kind not in ALLOWED_KINDS:
            kind = "REQUIRED"
        if kind == "OUT_OF_SCOPE":
            result.rejected_out_of_scope.append(cid or str(row.get("title") or ""))
            continue
        if kind == "IRRELEVANT":
            continue
        if not cid or cid not in by_id:
            result.rejected_unknown_ids.append(cid or "<missing_id>")
            continue
        if kind != "REQUIRED":
            # Record optional for analysis but do not add to graph inventory
            continue
        concept = by_id[cid]
        required.append(
            SelectedConcept(
                concept_id=cid,
                title=concept.title,
                kind="REQUIRED",
                reason=str(row.get("reason") or "").strip(),
                confidence=_as_float(row.get("confidence"), 0.5),
            )
        )

    # Deduplicate by concept_id preserving highest confidence
    best: dict[str, SelectedConcept] = {}
    for s in required:
        prev = best.get(s.concept_id)
        if prev is None or s.confidence > prev.confidence:
            best[s.concept_id] = s
    ranked = sorted(best.values(), key=lambda s: (-s.confidence, s.concept_id))
    cap = min(max_required, max_selected)
    if len(ranked) > cap:
        result.truncated = True
        ranked = ranked[:cap]
    result.selected = ranked
    return result


def concepts_to_topic_dicts(
    selected: list[SelectedConcept],
    inventory: DomainInventory,
) -> list[dict[str, Any]]:
    by_id = inventory.by_id()
    out: list[dict[str, Any]] = []
    for s in selected:
        c = by_id[s.concept_id]
        out.append(
            {
                "title": c.title,
                "summary": c.description,
                "confidence": max(0.5, s.confidence),
                "concept_id": c.id,
            }
        )
    return out
