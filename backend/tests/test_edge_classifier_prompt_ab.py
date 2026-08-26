"""Edge-classifier prompt variant tests (no API / no LLM)."""

from __future__ import annotations

import json

import pytest

from app.curriculum.edge_candidates import (
    generate_candidate_pairs,
    parse_classification_response,
)
from app.curriculum.selection import SelectedConcept
from app.prompts.domain_prior_edge_classifier import (
    EDGE_CLASSIFIER_PROMPT_VARIANTS,
    FEWSHOT_DIRECTNESS_BLOCK,
    build_edge_classification_prompt,
    edge_classifier_metadata,
    edge_classifier_prompt_body,
    edge_classifier_prompt_hash,
    resolve_edge_classifier_prompt_variant,
)
from app.services.generation_strategy import resolve_generation_strategy
from app.services.proposal_common import build_topics_and_dependencies, review_confidence_threshold


FORBIDDEN_FEWSHOT_TERMS = [
    "Parsing",
    "Kafka",
    "Replication",
    "Kubernetes",
    "compiler",
    "Lexical Analysis",
    "Paxos",
    "Raft",
    "stream processing",
    "Message Queues",
]


def test_baseline_prompt_has_no_fewshot_block():
    body = edge_classifier_prompt_body("edge_classifier_baseline")
    assert "Example 1 — DIRECT" not in body
    assert "Functions → Variables" not in body
    assert "DIRECT prerequisite" in body or "DIRECT prerequisites" in body


def test_fewshot_contains_only_synthetic_examples():
    block = FEWSHOT_DIRECTNESS_BLOCK
    assert "Functions" in block
    assert "Variables" in block
    assert "Data Types" in block
    assert "Databases" in block
    assert "Caching" in block
    assert "Testing" in block
    assert "Deployment" in block
    for term in FORBIDDEN_FEWSHOT_TERMS:
        assert term.casefold() not in block.casefold(), f"leak: {term}"


def test_fewshot_prompt_excludes_benchmark_concepts():
    selected = [
        SelectedConcept(concept_id="a", title="Alpha", kind="REQUIRED"),
        SelectedConcept(concept_id="b", title="Beta", kind="REQUIRED"),
    ]
    pairs, _ = generate_candidate_pairs(selected)
    prompt = build_edge_classification_prompt(
        "Learn something neutral",
        selected,
        pairs,
        variant="edge_classifier_fewshot_directness",
    )
    for term in FORBIDDEN_FEWSHOT_TERMS:
        # Allowed only if it appears in the request goal/selected — our goal is neutral
        if term in ("compiler",):
            assert term.casefold() not in FEWSHOT_DIRECTNESS_BLOCK.casefold()
        else:
            assert term.casefold() not in FEWSHOT_DIRECTNESS_BLOCK.casefold()


def test_prompt_variant_selection():
    assert resolve_edge_classifier_prompt_variant(None) == "edge_classifier_baseline"
    assert resolve_edge_classifier_prompt_variant("baseline") == "edge_classifier_baseline"
    assert (
        resolve_edge_classifier_prompt_variant("fewshot_directness")
        == "edge_classifier_fewshot_directness"
    )
    assert set(EDGE_CLASSIFIER_PROMPT_VARIANTS) == {
        "edge_classifier_baseline",
        "edge_classifier_fewshot_directness",
    }
    with pytest.raises(ValueError):
        resolve_edge_classifier_prompt_variant("not_a_variant")


def test_prompt_hash_and_version_recorded():
    h1 = edge_classifier_prompt_hash("edge_classifier_baseline")
    h2 = edge_classifier_prompt_hash("edge_classifier_fewshot_directness")
    assert h1 != h2
    assert len(h1) == 16
    meta = edge_classifier_metadata("compiler_construction", "v1", variant="fewshot")
    assert meta["prompt_variant"] == "edge_classifier_fewshot_directness"
    assert meta["edge_classifier_prompt_variant"] == "edge_classifier_fewshot_directness"
    assert meta["prompt_hash"] == h2
    assert meta["prompt_version"].startswith("edge_classifier_fewshot_directness@")


def test_output_schema_unchanged_across_variants():
    selected = [
        SelectedConcept(concept_id="id.0", title="A", kind="REQUIRED"),
        SelectedConcept(concept_id="id.1", title="B", kind="REQUIRED"),
    ]
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
    for _variant in EDGE_CLASSIFIER_PROMPT_VARIANTS:
        parsed = parse_classification_response(raw, pairs, id_to_title=id_to_title)
        assert parsed.required_edges == [("A", "B")]
        assert parsed.uncertain_count == 0


def test_unknown_ids_still_rejected():
    selected = [
        SelectedConcept(concept_id="id.0", title="A", kind="REQUIRED"),
        SelectedConcept(concept_id="id.1", title="B", kind="REQUIRED"),
    ]
    pairs, _ = generate_candidate_pairs(selected)
    parsed = parse_classification_response(
        json.dumps(
            {
                "decisions": [
                    {"from_id": "id.0", "to_id": "invented", "decision": "REQUIRED"}
                ]
            }
        ),
        pairs,
        id_to_title={s.concept_id: s.title for s in selected},
    )
    assert parsed.required_edges == []
    assert "invented" in parsed.rejected_unknown_ids


def test_invalid_pairs_still_rejected():
    selected = [
        SelectedConcept(concept_id="id.0", title="A", kind="REQUIRED"),
        SelectedConcept(concept_id="id.1", title="B", kind="REQUIRED"),
        SelectedConcept(concept_id="id.2", title="C", kind="REQUIRED"),
    ]
    pairs, _ = generate_candidate_pairs(selected)
    pairs = [p for p in pairs if not (p.from_id == "id.0" and p.to_id == "id.2")]
    parsed = parse_classification_response(
        json.dumps(
            {
                "decisions": [
                    {"from_id": "id.0", "to_id": "id.2", "decision": "REQUIRED"}
                ]
            }
        ),
        pairs,
        id_to_title={s.concept_id: s.title for s in selected},
    )
    assert ("id.0", "id.2") in parsed.rejected_non_candidate


def test_uncertain_still_no_edge():
    selected = [
        SelectedConcept(concept_id="id.0", title="A", kind="REQUIRED"),
        SelectedConcept(concept_id="id.1", title="B", kind="REQUIRED"),
    ]
    pairs, _ = generate_candidate_pairs(selected)
    parsed = parse_classification_response(
        json.dumps(
            {
                "decisions": [
                    {"from_id": "id.0", "to_id": "id.1", "decision": "UNCERTAIN"}
                ]
            }
        ),
        pairs,
        id_to_title={s.concept_id: s.title for s in selected},
    )
    assert parsed.required_edges == []
    assert parsed.uncertain_count == 1


def test_dag_validation_still_executes():
    topics = [
        {"title": "A", "summary": "a", "confidence": 0.9},
        {"title": "B", "summary": "b", "confidence": 0.9},
    ]
    deps = [
        {"from": "A", "to": "B", "confidence": 0.9},
        {"from": "B", "to": "A", "confidence": 0.9},
    ]
    _topics, accepted, skipped = build_topics_and_dependencies(
        topics, deps, confidence_threshold=review_confidence_threshold()
    )
    assert len(accepted) == 1
    assert any("cycle" in s.reason.casefold() for s in skipped)


def test_production_baseline_unchanged():
    assert resolve_generation_strategy(None) == "baseline"
    assert resolve_generation_strategy("baseline") == "baseline"


def test_fewshot_prompt_adds_no_concepts():
    selected = [
        SelectedConcept(concept_id="id.0", title="A", kind="REQUIRED"),
        SelectedConcept(concept_id="id.1", title="B", kind="REQUIRED"),
    ]
    pairs, _ = generate_candidate_pairs(selected)
    prompt = build_edge_classification_prompt(
        "goal", selected, pairs, variant="edge_classifier_fewshot_directness"
    )
    assert "Do NOT invent concept IDs" in prompt
    assert "Do NOT invent new pairs" in prompt


def test_baseline_and_fewshot_bodies_differ():
    b = edge_classifier_prompt_body("edge_classifier_baseline")
    f = edge_classifier_prompt_body("edge_classifier_fewshot_directness")
    assert len(f) > len(b)
    assert FEWSHOT_DIRECTNESS_BLOCK in f
    assert FEWSHOT_DIRECTNESS_BLOCK not in b
