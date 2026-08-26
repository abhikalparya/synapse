"""Pure relationship failure analysis tests (no API / no LLM)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.evaluation.persistent_failure_attribution import has_prerequisite_path
from app.evaluation.pure_relationship_analysis import (
    classify_relationship_failure,
    endpoints_acceptably_present,
    remap_dependencies_to_gold,
    run_pure_relationship_analysis,
    shortest_prerequisite_path_length,
    source_centered_stats,
    target_centered_stats,
)
from app.evaluation.metrics import normalize_topic
from app.evaluation.schemas import EvalExample, GeneratedGraph, example_to_dict


def _ex(**kwargs) -> EvalExample:
    defaults = dict(
        id="case_a",
        category="programming",
        difficulty="beginner",
        goal="Learn Python",
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
        optional_topics=[],
        acceptable_dependencies=[],
        ambiguous_dependencies=[],
        topic_aliases={},
    )
    defaults.update(kwargs)
    return EvalExample(**defaults)  # type: ignore[arg-type]


def _write_ds(path: Path, *examples: EvalExample) -> None:
    path.write_text(
        "\n".join(json.dumps(example_to_dict(ex)) for ex in examples) + "\n",
        encoding="utf-8",
    )


def _artifact(path: Path, examples: list[tuple[EvalExample, list[GeneratedGraph]]]) -> None:
    rows = []
    for ex, gens in examples:
        for i, g in enumerate(gens):
            rows.append(
                {
                    "example_id": ex.id,
                    "repetition": i,
                    "generation_index": i,
                    "generated_topics": list(g.topics),
                    "generated_dependencies": [list(d) for d in g.dependencies],
                    "parse_ok": True,
                    "generation_meta": {},
                }
            )
    payload = {
        "dataset": "toy_pure_rel",
        "systems": {"synapse": {"example_results": rows}},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_both_endpoints_present_edge_absent_is_pure(tmp_path: Path):
    ex = _ex(id="pure_001")
    gens = [
        GeneratedGraph(
            topics=["Variables", "Control Flow", "Functions"],
            dependencies=[("Functions", "Control Flow")],  # missing CF→Variables
        )
        for _ in range(3)
    ]
    ds = tmp_path / "ds.jsonl"
    art = tmp_path / "stab.json"
    _write_ds(ds, ex)
    _artifact(art, [(ex, gens)])
    md, js, pareto = run_pure_relationship_analysis(art, dataset_path=ds, output_dir=tmp_path / "out")
    data = json.loads(js.read_text())
    assert data["no_new_llm_calls"] is True
    assert data["metrics"]["PURE_RELATIONSHIP_FAILURE_COUNT"] >= 1
    keys = {(f["source_topic"], f["target_topic"]) for f in data["failures"]}
    assert ("Control Flow", "Variables") in keys
    assert md.is_file() and pareto.is_file()


def test_missing_endpoint_excluded(tmp_path: Path):
    ex = _ex(id="miss_ep")
    gens = [
        GeneratedGraph(
            topics=["Control Flow", "Functions"],  # Variables missing
            dependencies=[("Functions", "Control Flow")],
        )
        for _ in range(3)
    ]
    ds = tmp_path / "ds.jsonl"
    art = tmp_path / "stab.json"
    _write_ds(ds, ex)
    _artifact(art, [(ex, gens)])
    _, js, _ = run_pure_relationship_analysis(art, dataset_path=ds, output_dir=tmp_path / "out")
    data = json.loads(js.read_text())
    keys = {(f["source_topic"], f["target_topic"]) for f in data["failures"]}
    assert ("Control Flow", "Variables") not in keys
    assert any(e["exclude_reason"].startswith(("source", "target", "both", "representation")) for e in data["excluded_candidates"])


def test_representation_mismatch_excluded(tmp_path: Path):
    """Granularity-only stand-in should not enter the pure set."""
    ex = _ex(
        id="gran_001",
        gold_topics=["Variables and Data Types", "Control Flow"],
        required_topics=["Variables and Data Types", "Control Flow"],
        gold_dependencies=[("Control Flow", "Variables and Data Types")],
        required_dependencies=[("Control Flow", "Variables and Data Types")],
        topic_aliases={},
    )
    # "Variables" is a granularity variant of "Variables and Data Types" in many maps;
    # without alias, curated_alias match should fail → excluded.
    gens = [
        GeneratedGraph(
            topics=["Variables", "Control Flow"],
            dependencies=[],
        )
        for _ in range(3)
    ]
    ds = tmp_path / "ds.jsonl"
    art = tmp_path / "stab.json"
    _write_ds(ds, ex)
    _artifact(art, [(ex, gens)])
    _, js, _ = run_pure_relationship_analysis(art, dataset_path=ds, output_dir=tmp_path / "out")
    data = json.loads(js.read_text())
    keys = {(f["source_topic"], f["target_topic"]) for f in data["failures"]}
    assert ("Control Flow", "Variables and Data Types") not in keys


def test_reverse_direction_classification():
    assert (
        classify_relationship_failure(
            gold_edge_freq=0.0,
            reverse_freq=0.67,
            alt_path_freq=0.0,
            alt_direct_from_source_freq=0.0,
            no_relationship_freq=0.33,
            median_path_length=None,
            source_gold_target_recall=0.0,
            source_has_any_outgoing_freq=0.0,
        )
        == "REVERSED_DIRECTION"
    )


def test_alternative_path_detection():
    remapped = [("Control Flow", "Loops"), ("Loops", "Variables")]
    assert has_prerequisite_path(remapped, "Control Flow", "Variables")
    assert shortest_prerequisite_path_length(remapped, "Control Flow", "Variables") == 2
    assert (
        classify_relationship_failure(
            gold_edge_freq=0.0,
            reverse_freq=0.0,
            alt_path_freq=1.0,
            alt_direct_from_source_freq=1.0,
            no_relationship_freq=0.0,
            median_path_length=2.0,
            source_gold_target_recall=0.0,
            source_has_any_outgoing_freq=1.0,
        )
        == "ALTERNATIVE_PATH"
    )


def test_direct_omission_classification():
    assert (
        classify_relationship_failure(
            gold_edge_freq=0.0,
            reverse_freq=0.0,
            alt_path_freq=0.0,
            alt_direct_from_source_freq=0.0,
            no_relationship_freq=1.0,
            median_path_length=None,
            source_gold_target_recall=0.0,
            source_has_any_outgoing_freq=0.0,
        )
        == "MISSING_DIRECT_PREREQUISITE"
    )


def test_alternate_direct_relationship_classification():
    assert (
        classify_relationship_failure(
            gold_edge_freq=0.0,
            reverse_freq=0.0,
            alt_path_freq=0.0,
            alt_direct_from_source_freq=1.0,
            no_relationship_freq=0.0,
            median_path_length=None,
            source_gold_target_recall=0.0,
            source_has_any_outgoing_freq=1.0,
        )
        == "ALTERNATE_DIRECT_RELATIONSHIP"
    )


def test_incoming_only_to_target_is_not_alternate_direct():
    """Other nodes pointing at the gold target ≠ source selecting wrong prereqs."""
    assert (
        classify_relationship_failure(
            gold_edge_freq=0.0,
            reverse_freq=0.0,
            alt_path_freq=0.0,
            alt_direct_from_source_freq=0.0,
            no_relationship_freq=1.0,
            median_path_length=None,
            source_gold_target_recall=0.0,
            source_has_any_outgoing_freq=0.0,
        )
        == "MISSING_DIRECT_PREREQUISITE"
    )


def test_source_centered_analysis():
    remapped = [
        [("Functions", "Control Flow"), ("Functions", "Loops")],
        [("Functions", "Objects")],
        [("Functions", "Control Flow")],
    ]
    stats = source_centered_stats(
        "Functions",
        "Variables",
        remapped,
        {"Control Flow", "Variables"},
    )
    assert stats["SOURCE_TARGET_DIVERSITY"] >= 2
    assert stats["SOURCE_GOLD_TARGET_RECALL"] == 0.0
    assert 0.0 <= stats["TARGET_SELECTION_PRECISION"] <= 1.0


def test_target_centered_analysis():
    remapped = [
        [("Functions", "Data Types")],
        [("Control Flow", "Logic")],
        [],
    ]
    stats = target_centered_stats("Variables", {"Functions", "Control Flow"}, remapped)
    assert stats["TARGET_COVERAGE"] == 0.0
    assert stats["REQUIRED_EDGE_RECALL_PER_TARGET"] == 0.0


def test_frequency_across_generations(tmp_path: Path):
    ex = _ex(id="freq_001")
    gens = [
        GeneratedGraph(
            topics=["Variables", "Control Flow", "Functions"],
            dependencies=[("Variables", "Control Flow")],  # reverse of CF→Variables
        ),
        GeneratedGraph(
            topics=["Variables", "Control Flow", "Functions"],
            dependencies=[("Variables", "Control Flow")],
        ),
        GeneratedGraph(
            topics=["Variables", "Control Flow", "Functions"],
            dependencies=[],
        ),
    ]
    ds = tmp_path / "ds.jsonl"
    art = tmp_path / "stab.json"
    _write_ds(ds, ex)
    _artifact(art, [(ex, gens)])
    _, js, _ = run_pure_relationship_analysis(art, dataset_path=ds, output_dir=tmp_path / "out")
    data = json.loads(js.read_text())
    hit = next(
        f
        for f in data["failures"]
        if f["source_topic"] == "Control Flow" and f["target_topic"] == "Variables"
    )
    assert hit["generation_count"] == 3
    assert abs(hit["reverse_edge_frequency"] - 2 / 3) < 1e-9
    assert hit["failure_category"] == "REVERSED_DIRECTION"


def test_stable_failure_and_pareto(tmp_path: Path):
    ex = _ex(id="stable_001")
    gens = [
        GeneratedGraph(
            topics=["Variables", "Control Flow", "Functions"],
            dependencies=[],
        )
        for _ in range(3)
    ]
    ds = tmp_path / "ds.jsonl"
    art = tmp_path / "stab.json"
    _write_ds(ds, ex)
    _artifact(art, [(ex, gens)])
    _, js, pareto = run_pure_relationship_analysis(art, dataset_path=ds, output_dir=tmp_path / "out")
    data = json.loads(js.read_text())
    assert data["metrics"]["STABLE_RELATIONSHIP_FAILURE_RATE"] == 1.0
    assert "Rank" in pareto.read_text()
    assert data["pareto"][0]["rank"] == 1


def test_path_direction_semantics():
    # from requires to: Control Flow → Variables means CF requires Variables
    remapped = [("Control Flow", "Variables")]
    assert shortest_prerequisite_path_length(remapped, "Control Flow", "Variables") == 1
    assert shortest_prerequisite_path_length(remapped, "Variables", "Control Flow") is None


def test_endpoints_gate_requires_acceptable_rep():
    from app.evaluation.node_edge_attribution import load_node_representation_map

    ex = _ex()
    g = GeneratedGraph(topics=["Variables", "Control Flow"], dependencies=[])
    gate = endpoints_acceptably_present(
        "Control Flow",
        "Variables",
        ex,
        g,
        rep_map=load_node_representation_map(),
    )
    assert gate["ok"] is True


def test_no_api_key_required(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ex = _ex(id="offline")
    gens = [
        GeneratedGraph(
            topics=["Variables", "Control Flow", "Functions"],
            dependencies=[],
        )
        for _ in range(3)
    ]
    ds = tmp_path / "ds.jsonl"
    art = tmp_path / "stab.json"
    _write_ds(ds, ex)
    _artifact(art, [(ex, gens)])
    run_pure_relationship_analysis(art, dataset_path=ds, output_dir=tmp_path / "out")


def test_remap_preserves_gold_space():
    ex = _ex(topic_aliases={"Control Flow": ["Control Structures"]})
    g = GeneratedGraph(
        topics=["Variables", "Control Structures", "Functions"],
        dependencies=[("Control Structures", "Variables")],
    )
    remapped = remap_dependencies_to_gold(g, ex)
    norms = {(normalize_topic(a), normalize_topic(b)) for a, b in remapped}
    assert (normalize_topic("Control Flow"), normalize_topic("Variables")) in norms
