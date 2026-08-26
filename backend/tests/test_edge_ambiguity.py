"""Gold-edge ambiguity calibration tests (no API / no LLM)."""

from __future__ import annotations

import json
from pathlib import Path

from app.evaluation.edge_ambiguity import (
    adapt_example_for_edge_mode,
    approved_acceptable_records,
    load_edge_policy,
    rescore_edge_ambiguity_modes,
)
from app.evaluation.metrics import compare_graphs, score_graph
from app.evaluation.schemas import EvalExample, GeneratedGraph


def _ex(**kwargs) -> EvalExample:
    defaults = dict(
        id="python_basics_001",
        category="programming",
        difficulty="beginner",
        goal="Learn Python",
        gold_topics=["Variables", "Control Flow", "Functions"],
        gold_dependencies=[("Control Flow", "Variables"), ("Functions", "Control Flow")],
        required_topics=["Variables", "Control Flow", "Functions"],
        required_dependencies=[("Control Flow", "Variables"), ("Functions", "Control Flow")],
        acceptable_dependencies=[],
        ambiguous_dependencies=[],
        topic_aliases={},
    )
    defaults.update(kwargs)
    return EvalExample(**defaults)  # type: ignore[arg-type]


def test_required_edge_match_and_missing_remains_error():
    ex = adapt_example_for_edge_mode(_ex(), "fair")
    perfect = GeneratedGraph(
        topics=["Variables", "Control Flow", "Functions"],
        dependencies=[("Control Flow", "Variables"), ("Functions", "Control Flow")],
    )
    s = score_graph(ex, perfect)
    assert s.required_edge_recall == 1.0
    assert s.missing_required_edge_rate == 0.0

    missing = GeneratedGraph(
        topics=["Variables", "Control Flow", "Functions"],
        dependencies=[("Control Flow", "Variables")],
    )
    s2 = score_graph(ex, missing)
    assert s2.required_edge_recall == 0.5
    assert s2.missing_required_edge_rate == 0.5
    assert "MISSING_PREREQUISITE" in s2.failures


def test_acceptable_alternative_not_invalid_and_does_not_inflate_recall(tmp_path: Path):
    policy = {
        "entries": [
            {
                "case_id": "python_basics_001",
                "from": "Functions",
                "to": "Variables",
                "classification": "ACCEPTABLE_ALTERNATIVE",
                "approved": True,
                "reason": "skip-level ok",
                "why_not_required": "canonical uses Control Flow intermediate",
            },
        ],
    }
    path = tmp_path / "pol.json"
    path.write_text(json.dumps(policy), encoding="utf-8")
    reg = load_edge_policy(path)

    ex = adapt_example_for_edge_mode(_ex(), "edge_calibrated", edge_policy=reg)
    gen = GeneratedGraph(
        topics=["Variables", "Control Flow", "Functions"],
        dependencies=[
            ("Control Flow", "Variables"),
            ("Functions", "Control Flow"),
            ("Functions", "Variables"),  # acceptable alternative
        ],
    )
    s = score_graph(ex, gen)
    assert s.acceptable_alternative_edge_count == 1
    assert s.invalid_extra_edge_count == 0
    assert s.required_edge_recall == 1.0  # not inflated beyond required matches
    # required recall uses matched required only — still 2/2
    assert ("Functions", "Variables") not in [
        tuple(x) for x in compare_graphs(ex, gen)["matched_dependencies"]
    ]
    assert compare_graphs(ex, gen)["acceptable_dependencies_used"]


def test_acceptable_does_not_count_toward_required_recall_when_required_missing(tmp_path: Path):
    """Acceptable alternative must not substitute for a missing required edge."""
    policy = {
        "entries": [
            {
                "case_id": "python_basics_001",
                "from": "Functions",
                "to": "Variables",
                "classification": "ACCEPTABLE_ALTERNATIVE",
                "approved": True,
                "reason": "r",
                "why_not_required": "n",
            },
        ],
    }
    p = tmp_path / "p.json"
    p.write_text(json.dumps(policy), encoding="utf-8")
    reg = load_edge_policy(p)
    ex = adapt_example_for_edge_mode(_ex(), "edge_calibrated", edge_policy=reg)
    # Missing Functions→Control Flow required; only acceptable Functions→Variables present
    gen = GeneratedGraph(
        topics=["Variables", "Control Flow", "Functions"],
        dependencies=[("Control Flow", "Variables"), ("Functions", "Variables")],
    )
    s = score_graph(ex, gen)
    assert s.required_edge_recall == 0.5
    assert s.acceptable_alternative_edge_count == 1
    assert "MISSING_PREREQUISITE" in s.failures


def test_invalid_edges_remain_invalid():
    ex = adapt_example_for_edge_mode(_ex(), "fair")
    gen = GeneratedGraph(
        topics=["Variables", "Control Flow", "Functions"],
        dependencies=[
            ("Control Flow", "Variables"),
            ("Functions", "Control Flow"),
            ("Variables", "Functions"),  # wrong / extra
        ],
    )
    s = score_graph(ex, gen)
    assert s.invalid_extra_edge_count >= 1
    assert "INVALID_EXTRA_EDGE" in s.failures or "EXTRA_DEPENDENCY" in s.failures


def test_ambiguous_visible_not_correct(tmp_path: Path):
    policy = {
        "entries": [
            {
                "case_id": "python_basics_001",
                "from": "Functions",
                "to": "Variables",
                "classification": "AMBIGUOUS",
                "approved": True,
                "reason": "unclear",
                "why_not_required": "n",
            },
        ],
    }
    p = tmp_path / "p.json"
    p.write_text(json.dumps(policy), encoding="utf-8")
    ex = adapt_example_for_edge_mode(_ex(), "edge_calibrated", edge_policy=load_edge_policy(p))
    gen = GeneratedGraph(
        topics=["Variables", "Control Flow", "Functions"],
        dependencies=[
            ("Control Flow", "Variables"),
            ("Functions", "Control Flow"),
            ("Functions", "Variables"),
        ],
    )
    s = score_graph(ex, gen)
    assert s.ambiguous_edge_count == 1
    assert s.acceptable_alternative_edge_count == 0
    assert s.invalid_extra_edge_count == 0
    # Ambiguous must not inflate legacy precision as "correct"
    # dependency_precision = matched_required / gen (acceptable=0)
    assert s.dependency_precision == 2 / 3


def test_no_auto_promotion_from_frequency_or_similarity():
    """Repeated extras are still invalid unless in approved policy."""
    ex = adapt_example_for_edge_mode(_ex(), "edge_calibrated", edge_policy={"entries": []})
    gen = GeneratedGraph(
        topics=["Variables", "Control Flow", "Functions"],
        dependencies=[
            ("Control Flow", "Variables"),
            ("Functions", "Control Flow"),
            ("Functions", "Variables"),
        ],
    )
    s = score_graph(ex, gen)
    assert s.invalid_extra_edge_count == 1
    assert s.acceptable_alternative_edge_count == 0


def test_no_promotion_via_alias_alone():
    ex = adapt_example_for_edge_mode(
        _ex(topic_aliases={"Control Flow": ["Control Structures"]}),
        "fair",
    )
    # Alias resolves topic identity but does not invent acceptable edges
    gen = GeneratedGraph(
        topics=["Variables", "Control Structures", "Functions"],
        dependencies=[
            ("Control Structures", "Variables"),
            ("Functions", "Control Structures"),
            ("Functions", "Variables"),
        ],
    )
    s = score_graph(ex, gen)
    assert s.acceptable_alternative_edge_count == 0
    assert s.invalid_extra_edge_count == 1


def test_no_promotion_via_transitive_reachability():
    ex = adapt_example_for_edge_mode(_ex(), "fair")
    # A→C when A→B→C exists is redundant/extra, not acceptable
    gen = GeneratedGraph(
        topics=["Variables", "Control Flow", "Functions"],
        dependencies=[
            ("Control Flow", "Variables"),
            ("Functions", "Control Flow"),
            ("Functions", "Variables"),
        ],
    )
    s = score_graph(ex, gen)
    assert s.acceptable_alternative_edge_count == 0
    assert s.invalid_extra_edge_count == 1


def test_fair_backward_compatible_acceptable_from_dataset():
    ex = adapt_example_for_edge_mode(
        _ex(acceptable_dependencies=[("Functions", "Variables")]),
        "fair",
    )
    gen = GeneratedGraph(
        topics=["Variables", "Control Flow", "Functions"],
        dependencies=[
            ("Control Flow", "Variables"),
            ("Functions", "Control Flow"),
            ("Functions", "Variables"),
        ],
    )
    s = score_graph(ex, gen)
    assert s.acceptable_alternative_edge_count == 1
    assert s.dependency_precision == 1.0
    assert s.required_edge_recall == 1.0


def test_rescore_without_api(tmp_path: Path):
    artifact = {
        "timestamp": "t",
        "model": "test-model",
        "systems": {
            "synapse": {
                "example_results": [
                    {
                        "example_id": "python_basics_001",
                        "generated_topics": ["Variables", "Control Flow", "Functions"],
                        "generated_dependencies": [
                            ["Control Flow", "Variables"],
                            ["Functions", "Control Flow"],
                            ["Functions", "Variables"],
                        ],
                        "parse_ok": True,
                    },
                ],
            },
        },
    }
    src = tmp_path / "b.json"
    src.write_text(json.dumps(artifact), encoding="utf-8")
    ds = Path(__file__).resolve().parents[2] / "data" / "eval" / "learning_graph_quality_v1.jsonl"
    # Use synthetic example id that exists in dataset — python_basics_001 does
    assert ds.is_file()
    policy = {
        "entries": [
            {
                "case_id": "python_basics_001",
                "from": "Functions",
                "to": "Variables",
                "classification": "ACCEPTABLE_ALTERNATIVE",
                "approved": True,
                "reason": "r",
                "why_not_required": "n",
            },
        ],
    }
    pol = tmp_path / "pol.json"
    pol.write_text(json.dumps(policy), encoding="utf-8")
    out = rescore_edge_ambiguity_modes(
        src,
        modes=["fair", "edge_calibrated"],
        dataset_path=ds,
        system="synapse",
        output_dir=tmp_path,
        edge_policy_path=pol,
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "comparison_table" in data
    assert data["approved_acceptable_alternative_count"] == 1
    # Calibrated should have higher/equal dependency precision when alternative present
    fair = data["metrics_by_mode"]["fair"]
    cal = data["metrics_by_mode"]["edge_calibrated"]
    assert cal["acceptable_alternative_rate"] >= fair["acceptable_alternative_rate"]
    assert cal["required_edge_recall"] == fair["required_edge_recall"]


def test_policy_loader_approved_only(tmp_path: Path):
    p = tmp_path / "p.json"
    p.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "case_id": "a",
                        "from": "X",
                        "to": "Y",
                        "classification": "ACCEPTABLE_ALTERNATIVE",
                        "approved": True,
                    },
                    {
                        "case_id": "a",
                        "from": "X",
                        "to": "Z",
                        "classification": "ACCEPTABLE_ALTERNATIVE",
                        "approved": False,
                    },
                ],
            },
        ),
        encoding="utf-8",
    )
    assert len(approved_acceptable_records(load_edge_policy(p))) == 1
