"""Inventory vs relationship attribution tests (no API / no LLM)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from app.evaluation.inventory_attribution import (
    attribute_invalid_edge_vs_inventory,
    attribute_missing_edge_vs_inventory,
    edge_opportunity_and_conditional_recall,
    evaluate_inventory,
    extract_stage1_inventory,
    inventory_graph_from_titles,
    run_inventory_attribution,
    stage1_data_available,
)
from app.evaluation.metrics import score_graph
from app.evaluation.schemas import EvalExample, GeneratedGraph


def _ex(**kwargs) -> EvalExample:
    defaults = dict(
        id="python_basics_001",
        category="programming",
        difficulty="beginner",
        goal="Learn the basics of Python programming",
        gold_topics=["Variables", "Control Flow", "Functions"],
        gold_dependencies=[
            ("Control Flow", "Variables"),
            ("Functions", "Control Flow"),
        ],
        required_topics=["Variables", "Control Flow", "Functions"],
        required_dependencies=[
            ("Control Flow", "Variables"),
            ("Functions", "Control Flow"),
        ],
        optional_topics=["Testing"],
        acceptable_dependencies=[],
        ambiguous_dependencies=[],
        topic_aliases={"Control Flow": ["Control Structures"]},
    )
    defaults.update(kwargs)
    return EvalExample(**defaults)  # type: ignore[arg-type]


def test_stage1_inventory_evaluated_independently_of_dependencies():
    ex = _ex()
    metrics = evaluate_inventory(["Variables", "Control Flow", "Functions", "Ghost Topic"], ex)
    assert metrics["n_topics"] == 4
    assert metrics["topic_recall"] == 1.0
    assert "topic_f1" in metrics
    graph = inventory_graph_from_titles(["Variables", "Control Flow", "Functions"])
    assert graph.dependencies == []
    s = score_graph(ex, graph)
    # Inventory-only graph has topics but no edges → required-edge recall is 0
    assert s.dependency_recall == 0.0
    assert s.topic_f1 == 1.0


def test_baseline_inventory_evaluation_still_works():
    ex = _ex()
    m = evaluate_inventory(["Variables", "Control Structures", "Functions"], ex)
    assert m["topic_recall"] == 1.0  # Control Structures aliases Control Flow
    assert m["topic_precision"] == 1.0


def test_missing_source_absent():
    ex = _ex()
    inv = inventory_graph_from_titles(["Variables", "Control Flow"])
    rec = attribute_missing_edge_vs_inventory("Functions", "Control Flow", ex, inv)
    assert rec["primary_attribution"] == "SOURCE_ABSENT_FROM_INVENTORY"


def test_missing_target_absent():
    ex = _ex()
    inv = inventory_graph_from_titles(["Functions", "Control Flow"])
    rec = attribute_missing_edge_vs_inventory("Control Flow", "Variables", ex, inv)
    assert rec["primary_attribution"] == "TARGET_ABSENT_FROM_INVENTORY"


def test_missing_both_absent():
    ex = _ex()
    inv = inventory_graph_from_titles(["Unrelated Topic"])
    rec = attribute_missing_edge_vs_inventory("Functions", "Control Flow", ex, inv)
    assert rec["primary_attribution"] == "BOTH_ENDPOINTS_ABSENT_FROM_INVENTORY"


def test_missing_both_available_edge_omitted():
    ex = _ex()
    inv = inventory_graph_from_titles(["Variables", "Control Flow", "Functions"])
    rec = attribute_missing_edge_vs_inventory("Functions", "Control Flow", ex, inv)
    assert rec["primary_attribution"] == "BOTH_ENDPOINTS_AVAILABLE_EDGE_OMITTED"
    assert rec["both_endpoints_available"] is True


def test_granularity_not_counted_as_pure_relationship_failure():
    ex = _ex(
        gold_topics=["Programming Fundamentals", "Variables"],
        required_topics=["Programming Fundamentals", "Variables"],
        gold_dependencies=[("Variables", "Programming Fundamentals")],
        required_dependencies=[("Variables", "Programming Fundamentals")],
    )
    # Inventory has finer parts but not the umbrella → decomposition/granularity, not omission
    inv = inventory_graph_from_titles(["Variables", "Data Types", "Control Flow", "Functions"])
    rec = attribute_missing_edge_vs_inventory(
        "Variables", "Programming Fundamentals", ex, inv
    )
    assert rec["primary_attribution"] != "BOTH_ENDPOINTS_AVAILABLE_EDGE_OMITTED"
    assert rec["primary_attribution"] in {
        "TARGET_ABSENT_FROM_INVENTORY",
        "ENDPOINT_GRANULARITY_MISMATCH",
        "ENDPOINT_DECOMPOSITION",
        "ENDPOINT_ABSTRACTION_MISMATCH",
        "UNKNOWN",
    }


def test_invalid_extra_out_of_scope_endpoint():
    ex = _ex()
    rec = attribute_invalid_edge_vs_inventory("Underwater Basket Weaving", "Variables", ex)
    assert rec["primary_attribution"] in {
        "SOURCE_OUT_OF_SCOPE_INVENTORY",
        "BOTH_ENDPOINTS_OUT_OF_SCOPE_INVENTORY",
    }


def test_invalid_extra_both_valid_edge_invalid():
    ex = _ex()
    # Functions → Variables skips Control Flow; both endpoints valid required concepts
    rec = attribute_invalid_edge_vs_inventory("Functions", "Variables", ex)
    assert rec["primary_attribution"] == "BOTH_ENDPOINTS_VALID_EDGE_INVALID"
    assert rec["both_endpoints_valid"] is True


def test_edge_opportunity_rate():
    ex = _ex()
    inv = inventory_graph_from_titles(["Variables", "Control Flow"])  # Functions missing
    # Only one of two required edges has both endpoints
    out = edge_opportunity_and_conditional_recall(ex, inv, deps := [("Control Flow", "Variables")])
    assert out["required_edge_count"] == 2
    assert out["opportunity_edge_count"] == 1
    assert out["EDGE_OPPORTUNITY_RATE"] == 0.5
    assert out["CONDITIONAL_EDGE_RECALL"] == 1.0


def test_conditional_edge_recall_partial():
    ex = _ex()
    inv = inventory_graph_from_titles(["Variables", "Control Flow", "Functions"])
    out = edge_opportunity_and_conditional_recall(
        ex, inv, [("Control Flow", "Variables")]  # missing Functions→Control Flow
    )
    assert out["EDGE_OPPORTUNITY_RATE"] == 1.0
    assert out["opportunity_edge_count"] == 2
    assert out["opportunity_correct"] == 1
    assert out["CONDITIONAL_EDGE_RECALL"] == 0.5


def test_zero_denominator_safe():
    ex = _ex(required_dependencies=[], gold_dependencies=[], required_topics=[], gold_topics=[])
    inv = inventory_graph_from_titles([])
    out = edge_opportunity_and_conditional_recall(ex, inv, [])
    assert out["EDGE_OPPORTUNITY_RATE"] == 0.0
    assert out["CONDITIONAL_EDGE_RECALL"] == 0.0
    m = evaluate_inventory([], ex)
    assert m["topic_precision"] == 1.0  # empty gen convention from score_graph
    assert m["missing_foundational_concept_rate"] == 0.0


def test_extract_stage1_and_availability():
    row = {
        "generated_topics": ["A", "B"],
        "generation_meta": {
            "candidate_concepts": [{"title": "Variables"}, {"title": "Functions"}],
            "normalized_inventory": ["Variables", "Functions"],
        },
    }
    assert stage1_data_available(row)
    assert extract_stage1_inventory(row) == ["Variables", "Functions"]
    assert not stage1_data_available({"generated_topics": ["A"]})


def test_analyze_stored_artifact_without_api_keys(tmp_path: Path):
    ds = Path(__file__).resolve().parents[2] / "data" / "eval" / "learning_graph_quality_v1.jsonl"
    first = json.loads(ds.read_text(encoding="utf-8").splitlines()[0])
    eid = first["id"]
    artifact = {
        "dataset": "learning_graph_quality_v1",
        "model": "gpt-4o-mini",
        "seed": 42,
        "repetitions": 1,
        "example_count": 1,
        "systems": {
            "synapse": {
                "example_results": [
                    {
                        "example_id": eid,
                        "repetition": 0,
                        "parse_ok": True,
                        "generated_topics": list(first.get("gold_topics") or ["A"])[:3],
                        "generated_dependencies": (first.get("gold_dependencies") or [])[:1],
                        "skipped_dependencies": [],
                    }
                ]
            },
            "concept_first": {
                "example_results": [
                    {
                        "example_id": eid,
                        "repetition": 0,
                        "parse_ok": True,
                        "generated_topics": list(first.get("gold_topics") or ["A"])[:2],
                        "generated_dependencies": [],
                        "skipped_dependencies": [],
                        "generation_meta": {
                            "candidate_concepts": [
                                {"title": t} for t in (first.get("gold_topics") or ["A"])[:2]
                            ],
                            "normalized_inventory": list(first.get("gold_topics") or ["A"])[:2],
                        },
                    }
                ]
            },
        },
    }
    path = tmp_path / "quality.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    out = tmp_path / "fa"
    out.mkdir()

    # Ensure no LLM is contacted
    with patch("app.services.llm.call_llm_detailed") as llm:
        md, js = run_inventory_attribution(path, dataset_path=ds, output_dir=out, max_cases=3)
        llm.assert_not_called()
    assert md.is_file() and js.is_file()
    payload = json.loads(js.read_text(encoding="utf-8"))
    assert payload["llm_calls"] == "NO_NEW_LLM_CALLS"
    assert "diagnosis" in payload
    assert payload["aggregate"]["n_cases"] >= 1


def test_existing_evaluation_results_unchanged():
    """Attribution helpers must not alter score_graph outputs."""
    ex = _ex()
    graph = GeneratedGraph(
        topics=["Variables", "Control Flow", "Functions"],
        dependencies=[("Control Flow", "Variables"), ("Functions", "Control Flow")],
    )
    before = score_graph(ex, graph)
    _ = attribute_missing_edge_vs_inventory("Functions", "Control Flow", ex, graph)
    _ = evaluate_inventory(graph.topics, ex)
    after = score_graph(ex, graph)
    assert before.topic_f1 == after.topic_f1
    assert before.required_edge_f1 == after.required_edge_f1
    assert before.dependency_f1 == after.dependency_f1
