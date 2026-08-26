"""Persistent failure attribution + required-edge scoring fix tests (no API / no LLM)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.evaluation.metrics import compare_graphs, score_graph
from app.evaluation.persistent_failure_attribution import (
    attribute_stable_missing_edge,
    has_prerequisite_path,
    run_persistent_failure_attribution,
)
from app.evaluation.schemas import EvalExample, GeneratedGraph, example_to_dict


def _ex(**kwargs) -> EvalExample:
    defaults = dict(
        id="case_a",
        category="programming",
        difficulty="beginner",
        goal="Learn Python",
        gold_topics=["Variables", "Control Flow", "Functions", "Kafka"],
        gold_dependencies=[
            ("Control Flow", "Variables"),
            ("Functions", "Control Flow"),
            ("Functions", "Kafka"),
        ],
        required_topics=["Variables", "Control Flow", "Functions", "Kafka"],
        required_dependencies=[
            ("Control Flow", "Variables"),
            ("Functions", "Control Flow"),
            ("Functions", "Kafka"),
        ],
        optional_topics=[],
        acceptable_dependencies=[],
        ambiguous_dependencies=[],
        topic_aliases={"Control Flow": ["Control Structures"]},
    )
    defaults.update(kwargs)
    return EvalExample(**defaults)  # type: ignore[arg-type]


def _write_ds(path: Path, *examples: EvalExample) -> None:
    path.write_text(
        "\n".join(json.dumps(example_to_dict(ex)) for ex in examples) + "\n",
        encoding="utf-8",
    )


def test_matched_plus_reversed_same_edge_missing_rate_non_negative():
    ex = _ex(
        gold_topics=["A", "B"],
        gold_dependencies=[("B", "A")],
        required_topics=["A", "B"],
        required_dependencies=[("B", "A")],
        topic_aliases={},
    )
    gen = GeneratedGraph(topics=["A", "B"], dependencies=[("B", "A"), ("A", "B")])
    s = score_graph(ex, gen)
    assert 0.0 <= s.missing_required_edge_rate <= 1.0
    assert s.missing_required_edge_rate == 0.0
    assert s.dependency_direction_error_rate == 1.0
    assert s.required_edge_recall == 1.0


def test_duplicate_matched_edges_cannot_produce_negative_missing_rate():
    """Scoring correctness: duplicate gen edges mapping to one required edge."""
    ex = _ex(
        gold_topics=["A", "B"],
        gold_dependencies=[("B", "A")],
        required_topics=["A", "B"],
        required_dependencies=[("B", "A")],
        topic_aliases={},
    )
    # Three identical required edges generated — previously inflated matched count.
    gen = GeneratedGraph(
        topics=["A", "B"],
        dependencies=[("B", "A"), ("B", "A"), ("B", "A")],
    )
    cmp = compare_graphs(ex, gen)
    assert len(cmp["matched_dependencies"]) == 1
    assert len(cmp["matched_dependencies"]) <= len(ex.required_dependency_list())
    s = score_graph(ex, gen)
    assert 0.0 <= s.missing_required_edge_rate <= 1.0
    assert s.missing_required_edge_rate == 0.0
    assert s.required_edge_recall == 1.0
    # Duplicates after first match become extras
    assert len(cmp["extra_dependencies"]) == 2


def test_matched_required_count_cannot_exceed_unique_required():
    ex = _ex(
        gold_topics=["A", "B", "C"],
        gold_dependencies=[("B", "A"), ("C", "B")],
        required_topics=["A", "B", "C"],
        required_dependencies=[("B", "A"), ("C", "B")],
        topic_aliases={},
    )
    gen = GeneratedGraph(
        topics=["A", "B", "C"],
        dependencies=[("B", "A"), ("B", "A"), ("C", "B"), ("C", "B")],
    )
    cmp = compare_graphs(ex, gen)
    assert len(cmp["matched_dependencies"]) <= 2
    s = score_graph(ex, gen)
    assert 0.0 <= s.missing_required_edge_rate <= 1.0
    assert s.missing_required_edge_rate == 0.0


def test_source_never_present_attribution():
    assert (
        attribute_stable_missing_edge(
            source="Functions",
            target="Variables",
            source_freq=0.0,
            target_freq=1.0,
            source_rep=None,
            target_rep=None,
        )
        == "SOURCE_NEVER_PRESENT"
    )


def test_target_never_present_attribution():
    assert (
        attribute_stable_missing_edge(
            source="Functions",
            target="Kafka",
            source_freq=1.0,
            target_freq=0.0,
            source_rep=None,
            target_rep=None,
        )
        == "TARGET_NEVER_PRESENT"
    )


def test_both_endpoints_never_present_attribution():
    assert (
        attribute_stable_missing_edge(
            source="A",
            target="B",
            source_freq=0.0,
            target_freq=0.0,
            source_rep=None,
            target_rep=None,
        )
        == "BOTH_ENDPOINTS_NEVER_PRESENT"
    )


def test_endpoint_representation_mismatch_attribution():
    rep = {"status": "GRANULARITY_VARIANT", "subtype": "GRANULARITY_MISMATCH", "candidates": ["x"]}
    assert (
        attribute_stable_missing_edge(
            source="Control Flow",
            target="Variables",
            source_freq=0.0,
            target_freq=1.0,
            source_rep=rep,
            target_rep=None,
        )
        == "ENDPOINT_REPRESENTATION_MISMATCH"
    )


def test_both_endpoints_present_edge_omission():
    assert (
        attribute_stable_missing_edge(
            source="Functions",
            target="Control Flow",
            source_freq=1.0,
            target_freq=1.0,
            source_rep=None,
            target_rep=None,
        )
        == "BOTH_ENDPOINTS_PRESENT_EDGE_OMISSION"
    )


def test_mixed_endpoint_availability():
    assert (
        attribute_stable_missing_edge(
            source="Functions",
            target="Control Flow",
            source_freq=1.0,
            target_freq=0.5,
            source_rep=None,
            target_rep=None,
        )
        == "MIXED_ENDPOINT_AVAILABILITY"
    )


def test_alternative_path_detection_and_direction():
    # from requires to: Functions → Control Flow → Variables means Functions reaches Variables
    deps = [("Functions", "Control Flow"), ("Control Flow", "Variables")]
    assert has_prerequisite_path(deps, "Functions", "Variables") is True
    assert has_prerequisite_path(deps, "Variables", "Functions") is False  # direction respected
    assert has_prerequisite_path([("Functions", "Kafka")], "Functions", "Variables") is False


def test_no_alternative_path_when_disconnected():
    assert has_prerequisite_path([("A", "B"), ("C", "D")], "A", "D") is False


def _stability_artifact(rows: list[dict], *, repetitions: int = 3) -> dict:
    return {
        "benchmark_type": "quality_stability",
        "dataset": "learning_graph_quality_v1",
        "model": "gpt-4o-mini",
        "seed_supported": True,
        "repetitions": repetitions,
        "systems": {"synapse": {"example_results": rows}},
    }


def _row(eid: str, rep: int, topics: list[str], deps: list[list[str]]) -> dict:
    return {
        "example_id": eid,
        "repetition": rep,
        "generation_index": rep,
        "parse_ok": True,
        "generated_topics": topics,
        "generated_dependencies": deps,
        "skipped_dependencies": [],
        "generation_meta": {"seed": 42 + rep, "seed_supported": True},
    }


def test_full_attribution_pipeline_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # Gen always: Variables, Control Flow, Functions — never Kafka.
    # Missing: Functions→Kafka (TARGET_NEVER_PRESENT)
    # Also omit Control Flow→Variables even though both present → pure omission
    # Include Functions→Control Flow so that edge is NOT stable-missing
    rows = [
        _row(
            "case_a",
            i,
            ["Variables", "Control Flow", "Functions"],
            [["Functions", "Control Flow"]],  # omit Control Flow→Variables and Functions→Kafka
        )
        for i in range(3)
    ]
    art = tmp_path / "stab.json"
    art.write_text(json.dumps(_stability_artifact(rows)), encoding="utf-8")
    ds = tmp_path / "ds.jsonl"
    _write_ds(ds, _ex())

    md, js, pareto = run_persistent_failure_attribution(
        art, dataset_path=ds, output_dir=tmp_path / "out"
    )
    payload = json.loads(js.read_text(encoding="utf-8"))
    assert md.is_file() and pareto.is_file()
    assert payload["scoring_fix"]["kind"] == "metric_bug_fix"

    edges = payload["stable_missing_edges"]
    assert edges, "expected stable missing edges"
    # Exactly one primary attribution each
    assert all(e["primary_attribution"] for e in edges)
    attrs = {e["edge_key"]: e["primary_attribution"] for e in edges}
    assert attrs.get("Functions→Kafka") == "TARGET_NEVER_PRESENT"
    assert attrs.get("Control Flow→Variables") == "BOTH_ENDPOINTS_PRESENT_EDGE_OMISSION"

    # Dependency impact for Kafka
    kafka = next(t for t in payload["never_present_topics"] if t["topic"] == "Kafka")
    assert kafka["target_impact"] >= 1
    assert kafka["total_impact"] == kafka["source_impact"] + kafka["target_impact"]

    # Pareto descending
    impacts = [r["stable_missing_edges_explained"] for r in payload["pareto_by_never_present_topic"]]
    assert impacts == sorted(impacts, reverse=True)
    if payload["pareto_by_never_present_topic"]:
        cum = [r["cumulative_pct"] for r in payload["pareto_by_never_present_topic"]]
        assert cum == sorted(cum)
        assert abs(cum[-1] - sum(r["pct_of_stable_missing"] for r in payload["pareto_by_never_present_topic"])) < 1e-9 or cum[-1] <= 1.0 + 1e-6

    rates = {r["primary_attribution"]: r["rate"] for r in payload["attribution_summary"]}
    pure = rates.get("BOTH_ENDPOINTS_PRESENT_EDGE_OMISSION", 0)
    node = payload["node_or_endpoint_failure_rate"]
    assert abs(payload["pure_relationship_failure_rate"] - pure) < 1e-9
    assert 0.0 <= node <= 1.0
    assert 0.0 <= payload["pure_relationship_failure_rate"] <= 1.0
    assert payload["diagnosis"]["code"] in {
        "ENDPOINT_COVERAGE_DOMINANT",
        "REPRESENTATION_DOMINANT",
        "RELATIONSHIP_REASONING_DOMINANT",
        "MIXED_ROOT_CAUSES",
        "INSUFFICIENT_EVIDENCE",
    }


def test_source_never_and_both_never_via_pipeline(tmp_path: Path):
    # Never generate Functions or Kafka; Variables+Control Flow always present with edge
    rows = [
        _row("case_a", i, ["Variables", "Control Flow"], [["Control Flow", "Variables"]])
        for i in range(3)
    ]
    art = tmp_path / "a.json"
    art.write_text(json.dumps(_stability_artifact(rows)), encoding="utf-8")
    ds = tmp_path / "ds.jsonl"
    _write_ds(ds, _ex())
    _, js, _ = run_persistent_failure_attribution(art, dataset_path=ds, output_dir=tmp_path / "out")
    payload = json.loads(js.read_text(encoding="utf-8"))
    attrs = {e["edge_key"]: e["primary_attribution"] for e in payload["stable_missing_edges"]}
    assert attrs["Functions→Control Flow"] == "SOURCE_NEVER_PRESENT"
    assert attrs["Functions→Kafka"] == "BOTH_ENDPOINTS_NEVER_PRESENT"


def test_alternative_path_in_pipeline(tmp_path: Path):
    # Both endpoints present; omit direct Control Flow→Variables but provide
    # Control Flow → Functions → Variables (wrong curriculum but path exists)
    # Wait: path from Control Flow to Variables: Control Flow → Functions → Variables
    rows = [
        _row(
            "case_a",
            i,
            ["Variables", "Control Flow", "Functions", "Kafka"],
            [
                ["Control Flow", "Functions"],
                ["Functions", "Variables"],
                ["Functions", "Control Flow"],
                ["Functions", "Kafka"],
            ],
        )
        for i in range(3)
    ]
    art = tmp_path / "a.json"
    art.write_text(json.dumps(_stability_artifact(rows)), encoding="utf-8")
    ds = tmp_path / "ds.jsonl"
    _write_ds(ds, _ex())
    _, js, _ = run_persistent_failure_attribution(art, dataset_path=ds, output_dir=tmp_path / "out")
    payload = json.loads(js.read_text(encoding="utf-8"))
    cf_var = next(e for e in payload["stable_missing_edges"] if e["edge_key"] == "Control Flow→Variables")
    assert cf_var["primary_attribution"] == "BOTH_ENDPOINTS_PRESENT_EDGE_OMISSION"
    assert cf_var["alternative_path_present"] is True
