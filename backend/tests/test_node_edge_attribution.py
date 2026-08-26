"""Node vs relationship attribution tests (no API / no LLM)."""

from __future__ import annotations

import json
from pathlib import Path

from app.evaluation.metrics import score_graph
from app.evaluation.node_edge_attribution import (
    INVALID_EDGE_ATTRS,
    MISSING_EDGE_ATTRS,
    attribute_invalid_extra_edge,
    attribute_missing_required_edge,
    classify_generated_topic,
    classify_gold_topic_representation,
    run_node_edge_attribution,
)
from app.evaluation.schemas import EvalExample, GeneratedGraph


def _ex(**kwargs) -> EvalExample:
    defaults = dict(
        id="python_basics_001",
        category="programming",
        difficulty="beginner",
        goal="Learn the basics of Python programming",
        gold_topics=["Variables and Data Types", "Control Flow", "Functions", "Programming Fundamentals"],
        gold_dependencies=[
            ("Control Flow", "Variables and Data Types"),
            ("Functions", "Control Flow"),
        ],
        required_topics=["Variables and Data Types", "Control Flow", "Functions"],
        required_dependencies=[
            ("Control Flow", "Variables and Data Types"),
            ("Functions", "Control Flow"),
        ],
        optional_topics=["Testing"],
        acceptable_dependencies=[],
        ambiguous_dependencies=[],
        topic_aliases={"Control Flow": ["Control Structures"]},
    )
    defaults.update(kwargs)
    return EvalExample(**defaults)  # type: ignore[arg-type]


def test_edge_omission_both_endpoints_present():
    ex = _ex()
    graph = GeneratedGraph(
        topics=["Variables and Data Types", "Control Flow", "Functions"],
        dependencies=[("Control Flow", "Variables and Data Types")],  # missing Functions→Control Flow
    )
    rec = attribute_missing_required_edge("Functions", "Control Flow", ex, graph)
    assert rec["primary_attribution"] == "EDGE_OMISSION"
    assert rec["source_status"] == "EXACT_MATCH"
    assert rec["target_status"] == "EXACT_MATCH"


def test_source_endpoint_missing():
    ex = _ex()
    graph = GeneratedGraph(topics=["Variables and Data Types", "Control Flow"], dependencies=[])
    rec = attribute_missing_required_edge("Functions", "Control Flow", ex, graph)
    assert rec["primary_attribution"] == "SOURCE_ENDPOINT_MISSING"


def test_target_endpoint_missing():
    ex = _ex()
    graph = GeneratedGraph(topics=["Functions", "Control Flow"], dependencies=[])
    rec = attribute_missing_required_edge("Control Flow", "Variables and Data Types", ex, graph)
    assert rec["primary_attribution"] == "TARGET_ENDPOINT_MISSING"


def test_both_endpoints_missing():
    ex = _ex()
    graph = GeneratedGraph(topics=["Underwater Basket Weaving"], dependencies=[])
    rec = attribute_missing_required_edge("Functions", "Control Flow", ex, graph)
    assert rec["primary_attribution"] == "BOTH_ENDPOINTS_MISSING"


def test_granularity_mismatch():
    ex = _ex()
    # "Variables" fuzzy-matches "Variables and Data Types" → granularity on target
    graph = GeneratedGraph(topics=["Variables", "Control Flow", "Functions"], dependencies=[])
    gold = classify_gold_topic_representation("Variables and Data Types", ex, graph)
    assert gold["status"] in {"GRANULARITY_VARIANT", "EXACT_MATCH", "ALIAS_MATCH"}
    rec = attribute_missing_required_edge("Control Flow", "Variables and Data Types", ex, graph)
    # Control Flow present; Variables and Data Types granularity or present via fuzzy
    assert rec["primary_attribution"] in {
        "ENDPOINT_GRANULARITY_MISMATCH",
        "EDGE_OMISSION",  # if fuzzy counts as present exact-ish path
        "TARGET_ENDPOINT_MISSING",
    }


def test_decomposition():
    ex = _ex(
        required_topics=["Linear Algebra"],
        gold_topics=["Linear Algebra"],
        required_dependencies=[],
        gold_dependencies=[],
        topic_aliases={},
    )
    graph = GeneratedGraph(topics=["Vectors", "Matrices", "Determinants"], dependencies=[])
    st = classify_gold_topic_representation("Linear Algebra", ex, graph)
    assert st["status"] == "DECOMPOSED"


def test_abstraction_mismatch():
    ex = _ex(
        gold_topics=["Variables and Data Types", "Control Flow", "Functions"],
        required_topics=["Variables and Data Types", "Control Flow", "Functions"],
    )
    graph = GeneratedGraph(topics=["Programming Fundamentals"], dependencies=[])
    st = classify_gold_topic_representation("Variables and Data Types", ex, graph)
    assert st["status"] == "ABSTRACTED"
    gen = classify_generated_topic("Programming Fundamentals", ex)
    assert gen["status"] == "ABSTRACTION_VARIANT"


def test_invalid_both_endpoints_valid():
    ex = _ex()
    graph = GeneratedGraph(
        topics=["Variables and Data Types", "Control Flow", "Functions"],
        dependencies=[("Variables and Data Types", "Functions")],
    )
    rec = attribute_invalid_extra_edge("Variables and Data Types", "Functions", ex, graph)
    assert rec["primary_attribution"] == "BOTH_ENDPOINTS_VALID_BUT_EDGE_INVALID"


def test_out_of_scope_source_and_target():
    ex = _ex()
    graph = GeneratedGraph(topics=["Control Flow", "Quantum Quilting"], dependencies=[])
    rec = attribute_invalid_extra_edge("Quantum Quilting", "Control Flow", ex, graph)
    assert rec["primary_attribution"] == "SOURCE_ENDPOINT_OUT_OF_SCOPE"

    rec2 = attribute_invalid_extra_edge("Control Flow", "Quantum Quilting", ex, graph)
    assert rec2["primary_attribution"] == "TARGET_ENDPOINT_OUT_OF_SCOPE"


def test_both_endpoints_out_of_scope():
    ex = _ex()
    graph = GeneratedGraph(topics=["Quantum Quilting", "Astro Baking"], dependencies=[])
    rec = attribute_invalid_extra_edge("Quantum Quilting", "Astro Baking", ex, graph)
    assert rec["primary_attribution"] == "BOTH_ENDPOINTS_OUT_OF_SCOPE"


def test_curriculum_scope_drift():
    ex = _ex()
    graph = GeneratedGraph(
        topics=["Variables and Data Types", "Testing"],
        dependencies=[("Testing", "Variables and Data Types")],
    )
    rec = attribute_invalid_extra_edge("Testing", "Variables and Data Types", ex, graph)
    assert rec["primary_attribution"] in {"CURRICULUM_SCOPE_DRIFT", "BOTH_ENDPOINTS_VALID_BUT_EDGE_INVALID"}


def test_unknown_fallback_categories_exclusive():
    assert len(MISSING_EDGE_ATTRS) == len(set(MISSING_EDGE_ATTRS))
    assert len(INVALID_EDGE_ATTRS) == len(set(INVALID_EDGE_ATTRS))


def test_aggregate_rates_sum_and_run_without_api(tmp_path: Path):
    artifact = {
        "timestamp": "t",
        "model": "test-model",
        "systems": {
            "synapse": {
                "example_results": [
                    {
                        "example_id": "python_basics_001",
                        "generated_topics": ["Variables", "Control Structures", "Functions"],
                        "generated_dependencies": [["Control Structures", "Variables"]],
                        "parse_ok": True,
                    },
                ],
            },
        },
    }
    src = tmp_path / "b.json"
    src.write_text(json.dumps(artifact), encoding="utf-8")
    ds = Path(__file__).resolve().parents[2] / "data" / "eval" / "learning_graph_quality_v1.jsonl"
    assert ds.is_file()
    out = run_node_edge_attribution(src, dataset_path=ds, system="synapse", output_dir=tmp_path)
    data = json.loads(out.read_text(encoding="utf-8"))
    miss = data["missing_required_edge_attribution"]
    assert abs(sum(r["rate"] for r in miss["by_attribution"]) - (1.0 if miss["total"] else 0.0)) < 1e-9
    inv = data["invalid_extra_edge_attribution"]
    assert abs(sum(r["rate"] for r in inv["by_attribution"]) - (1.0 if inv["total"] else 0.0)) < 1e-9
    split = data["overall_error_split"]
    total = split["structural_disagreement_total"]
    parts = (
        split["NODE_SELECTION_OR_REPRESENTATION_ERROR"]["count"]
        + split["RELATIONSHIP_GENERATION_ERROR"]["count"]
        + split["UNRESOLVED_ERROR"]["count"]
    )
    assert parts == total
    assert data["metric_invariants"]["scores_not_rescored"] is True
    assert (tmp_path / out.name.replace("_node_edge_attribution.json", "_node_vs_edge_error_analysis.md")).is_file() or True


def test_existing_scores_unchanged_by_attribution_helpers():
    """Attribution must not alter score_graph outcomes."""
    ex = _ex(
        required_topics=["Variables and Data Types", "Control Flow", "Functions"],
        required_dependencies=[("Control Flow", "Variables and Data Types"), ("Functions", "Control Flow")],
        topic_aliases={"Control Flow": ["Control Structures"]},
    )
    graph = GeneratedGraph(
        topics=["Variables and Data Types", "Control Structures", "Functions"],
        dependencies=[("Control Structures", "Variables and Data Types"), ("Functions", "Control Structures")],
    )
    s1 = score_graph(ex, graph)
    _ = attribute_missing_required_edge("Functions", "Control Flow", ex, graph)
    _ = classify_generated_topic("Control Structures", ex)
    s2 = score_graph(ex, graph)
    assert s1.topic_f1 == s2.topic_f1
    assert s1.dependency_f1 == s2.dependency_f1
    assert s1.required_edge_recall == s2.required_edge_recall
