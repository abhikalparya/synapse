"""Constrained dependency classification (edge classifier) unit tests — no API/LLM."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from app.curriculum.edge_candidates import (
    batch_candidate_pairs,
    generate_candidate_pairs,
    parse_classification_response,
)
from app.curriculum.selection import SelectedConcept
from app.services.generation_strategy import (
    resolve_generation_strategy,
    strategy_enables_domain_curriculum_prior,
    strategy_enables_domain_prior_edge_classifier,
)
from app.services.proposal_common import build_topics_and_dependencies
from app.services.topics import would_create_cycle


def _concepts(*titles: str) -> list[SelectedConcept]:
    out = []
    for i, t in enumerate(titles):
        out.append(SelectedConcept(concept_id=f"id.{i}", title=t, kind="REQUIRED"))
    return out


def test_candidate_pair_count_n_times_n_minus_1():
    selected = _concepts("A", "B", "C", "D")
    pairs, meta = generate_candidate_pairs(selected)
    assert meta["selected_concept_count"] == 4
    assert meta["candidate_space_size"] == 4 * 3
    assert meta["candidate_pairs_evaluated"] == 12
    assert meta["candidate_pairs_omitted"] == 0
    assert len(pairs) == 12


def test_no_self_loops():
    selected = _concepts("A", "B")
    pairs, _ = generate_candidate_pairs(selected)
    assert all(p.from_id != p.to_id for p in pairs)
    ids = {(p.from_id, p.to_id) for p in pairs}
    assert ("id.0", "id.0") not in ids


def test_graph_direction_preserved_in_pairs():
    """from → to means from requires to; both directions are candidates."""
    selected = _concepts("Frontend", "Parsing")
    pairs, _ = generate_candidate_pairs(selected)
    keys = {(p.from_title, p.to_title) for p in pairs}
    assert ("Frontend", "Parsing") in keys
    assert ("Parsing", "Frontend") in keys


def test_unknown_id_rejected():
    selected = _concepts("A", "B")
    pairs, _ = generate_candidate_pairs(selected)
    id_to_title = {s.concept_id: s.title for s in selected}
    raw = json.dumps(
        {
            "decisions": [
                {"from_id": "id.0", "to_id": "invented.x", "decision": "REQUIRED"},
            ]
        }
    )
    parsed = parse_classification_response(raw, pairs, id_to_title=id_to_title)
    assert parsed.required_edges == []
    assert "invented.x" in parsed.rejected_unknown_ids


def test_non_candidate_pair_rejected():
    selected = _concepts("A", "B", "C")
    pairs, _ = generate_candidate_pairs(selected)
    # Truncate candidates so A→C is not in evaluated set
    pairs = [p for p in pairs if not (p.from_id == "id.0" and p.to_id == "id.2")]
    id_to_title = {s.concept_id: s.title for s in selected}
    raw = json.dumps(
        {
            "decisions": [
                {"from_id": "id.0", "to_id": "id.2", "decision": "REQUIRED"},
            ]
        }
    )
    parsed = parse_classification_response(raw, pairs, id_to_title=id_to_title)
    assert parsed.required_edges == []
    assert ("id.0", "id.2") in parsed.rejected_non_candidate


def test_duplicate_decisions_deduplicated():
    selected = _concepts("A", "B")
    pairs, _ = generate_candidate_pairs(selected)
    id_to_title = {s.concept_id: s.title for s in selected}
    raw = json.dumps(
        {
            "decisions": [
                {"from_id": "id.0", "to_id": "id.1", "decision": "REQUIRED"},
                {"from_id": "id.0", "to_id": "id.1", "decision": "REQUIRED"},
            ]
        }
    )
    parsed = parse_classification_response(raw, pairs, id_to_title=id_to_title)
    assert len(parsed.required_edges) == 1
    assert parsed.duplicate_decision_count == 1


def test_uncertain_does_not_create_edge():
    selected = _concepts("A", "B")
    pairs, _ = generate_candidate_pairs(selected)
    id_to_title = {s.concept_id: s.title for s in selected}
    raw = json.dumps(
        {
            "decisions": [
                {"from_id": "id.0", "to_id": "id.1", "decision": "UNCERTAIN"},
            ]
        }
    )
    parsed = parse_classification_response(raw, pairs, id_to_title=id_to_title)
    assert parsed.required_edges == []
    assert parsed.uncertain_count == 1


def test_only_required_creates_edges():
    selected = _concepts("A", "B")
    pairs, _ = generate_candidate_pairs(selected)
    id_to_title = {s.concept_id: s.title for s in selected}
    raw = json.dumps(
        {
            "decisions": [
                {"from_id": "id.0", "to_id": "id.1", "decision": "REQUIRED"},
                {"from_id": "id.1", "to_id": "id.0", "decision": "NOT_REQUIRED"},
            ]
        }
    )
    parsed = parse_classification_response(raw, pairs, id_to_title=id_to_title)
    assert parsed.required_edges == [("A", "B")]


def test_dag_validation_still_runs_on_required():
    from app.services.proposal_common import review_confidence_threshold

    topics = [
        {"title": "A", "summary": "a", "confidence": 0.9},
        {"title": "B", "summary": "b", "confidence": 0.9},
    ]
    deps = [
        {"from": "A", "to": "B", "confidence": 0.9},
        {"from": "B", "to": "A", "confidence": 0.9},
    ]
    proposed_topics, proposed_deps, skipped = build_topics_and_dependencies(
        topics, deps, confidence_threshold=review_confidence_threshold()
    )
    assert len(proposed_topics) == 2
    # One edge accepted, the reverse that cycles is skipped
    assert len(proposed_deps) == 1
    assert any("cycle" in s.reason.casefold() for s in skipped)


def test_cyclic_candidates_rejected_by_existing_validator():
    accepted = [{"from_topic_id": "t1", "to_topic_id": "t2"}]
    assert would_create_cycle("t2", "t1", accepted) is True


def test_batching_deterministic():
    selected = _concepts("A", "B", "C", "D", "E")
    pairs, _ = generate_candidate_pairs(selected)
    b1 = batch_candidate_pairs(pairs, pairs_per_batch=7)
    b2 = batch_candidate_pairs(pairs, pairs_per_batch=7)
    assert [[(p.from_id, p.to_id) for p in batch] for batch in b1] == [
        [(p.from_id, p.to_id) for p in batch] for batch in b2
    ]
    assert sum(len(b) for b in b1) == len(pairs)


def test_truncation_reported_not_silent():
    selected = _concepts("A", "B", "C", "D")
    pairs, meta = generate_candidate_pairs(selected, max_candidate_pairs=5)
    assert meta["truncated"] is True
    assert meta["candidate_space_size"] == 12
    assert meta["candidate_pairs_evaluated"] == 5
    assert meta["candidate_pairs_omitted"] == 7
    assert len(pairs) == 5


def test_baseline_and_prior_strategies_unchanged():
    assert resolve_generation_strategy(None) == "baseline"
    assert resolve_generation_strategy("baseline") == "baseline"
    assert resolve_generation_strategy("domain_curriculum_prior") == "domain_curriculum_prior"
    assert strategy_enables_domain_curriculum_prior("domain_curriculum_prior")
    assert not strategy_enables_domain_curriculum_prior("domain_prior_edge_classifier")
    assert resolve_generation_strategy("domain_prior_edge_classifier") == "domain_prior_edge_classifier"
    assert resolve_generation_strategy("edge_classifier") == "domain_prior_edge_classifier"
    assert strategy_enables_domain_prior_edge_classifier("domain_prior_edge_classifier")


def test_runtime_classifier_module_has_no_gold_imports():
    import app.services.domain_prior_edge_classifier as mod
    import app.curriculum.edge_candidates as cand

    for m in (mod, cand):
        src = Path(inspect.getfile(m)).read_text(encoding="utf-8")
        assert "learning_graph_quality" not in src
        assert "gold_dependencies" not in src
        assert "load_dataset" not in src


def test_new_concept_count_zero_when_inventory_titles_only():
    """Classifier post-process only emits inventory titles from selected concepts."""
    from app.curriculum.inventory import load_domain_inventory

    inv = load_domain_inventory("compiler_construction")
    titles = {c.title for c in inv.concepts}
    selected = [
        SelectedConcept(concept_id=c.id, title=c.title, kind="REQUIRED")
        for c in inv.concepts[:4]
    ]
    pairs, _ = generate_candidate_pairs(selected)
    id_to_title = {s.concept_id: s.title for s in selected}
    decisions = [
        {"from_id": pairs[0].from_id, "to_id": pairs[0].to_id, "decision": "REQUIRED"}
    ]
    parsed = parse_classification_response(
        json.dumps({"decisions": decisions}), pairs, id_to_title=id_to_title
    )
    for frm, to in parsed.required_edges:
        assert frm in titles and to in titles
