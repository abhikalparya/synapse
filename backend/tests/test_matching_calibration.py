"""Matching calibration + curated aliases (no LLM / no API keys)."""

from __future__ import annotations

import json
from pathlib import Path

from app.evaluation.matching_calibration import rescore_matching_modes
from app.evaluation.matching_modes import adapt_example_for_mode, approved_alias_map, load_curated_aliases
from app.evaluation.metrics import match_topic, score_graph
from app.evaluation.schemas import EvalExample, GeneratedGraph
from app.evaluation.topic_equivalence import classify_unmatched_topic


def _ex(**kwargs) -> EvalExample:
    defaults = dict(
        id="t",
        category="programming",
        difficulty="beginner",
        goal="g",
        gold_topics=["Control Flow", "Variables and Data Types", "Linear Algebra", "Probability", "XSS"],
        gold_dependencies=[("Control Flow", "Variables and Data Types")],
        required_topics=["Control Flow", "Variables and Data Types"],
        optional_topics=[],
        topic_aliases={},
    )
    defaults.update(kwargs)
    return EvalExample(**defaults)  # type: ignore[arg-type]


def test_exact_and_case_insensitive_alias_match():
    ex = adapt_example_for_mode(
        _ex(topic_aliases={"Control Flow": ["Control Structures"]}),
        "fair",
    )
    assert match_topic("Control Structures", ex) == "Control Flow"
    assert match_topic("control structures", ex) == "Control Flow"
    assert match_topic("CONTROL STRUCTURES", ex) == "Control Flow"


def test_normalized_alias_match():
    ex = adapt_example_for_mode(
        _ex(topic_aliases={"Control Flow": ["Control-Structures!"]}),
        "fair",
    )
    assert match_topic("Control Structures", ex) == "Control Flow"


def test_dependency_matching_after_alias_resolution():
    ex = adapt_example_for_mode(
        _ex(
            required_topics=["Control Flow", "Variables and Data Types"],
            gold_topics=["Control Flow", "Variables and Data Types"],
            gold_dependencies=[("Control Flow", "Variables and Data Types")],
            required_dependencies=[("Control Flow", "Variables and Data Types")],
            topic_aliases={"Control Flow": ["Control Structures"]},
        ),
        "fair",
    )
    gen = GeneratedGraph(
        topics=["Control Structures", "Variables and Data Types"],
        dependencies=[("Control Structures", "Variables and Data Types")],
    )
    s = score_graph(ex, gen)
    assert s.dependency_recall == 1.0
    assert s.dependency_f1 == 1.0
    assert s.topic_f1 == 1.0


def test_multiple_aliases_for_one_canonical():
    ex = adapt_example_for_mode(
        _ex(topic_aliases={"XSS": ["Cross-Site Scripting", "Cross-Site Scripting (XSS)"]}),
        "fair",
    )
    assert match_topic("Cross-Site Scripting", ex) == "XSS"
    assert match_topic("Cross-Site Scripting (XSS)", ex) == "XSS"


def test_curated_registry_loads_approved_only(tmp_path: Path):
    path = tmp_path / "aliases.json"
    path.write_text(
        json.dumps(
            {
                "version": "t",
                "entries": [
                    {"canonical": "XSS", "aliases": ["Cross-Site Scripting (XSS)"], "approved": True},
                    {"canonical": "Probability", "aliases": ["Statistics"], "approved": False},
                ],
            },
        ),
        encoding="utf-8",
    )
    reg = load_curated_aliases(path)
    m = approved_alias_map(reg)
    assert m == {"XSS": ["Cross-Site Scripting (XSS)"]}


def test_concept_decomposition_not_auto_matched():
    ex = adapt_example_for_mode(
        _ex(
            required_topics=["Linear Algebra"],
            gold_topics=["Linear Algebra"],
            gold_dependencies=[],
            topic_aliases={},
        ),
        "fair",
    )
    assert match_topic("Vectors", ex) is None
    c = classify_unmatched_topic("Vectors", ex)
    assert c["proposed_classification"] == "CONCEPT_DECOMPOSITION"
    assert c["candidate_alias"] is False


def test_concept_abstraction_not_auto_matched():
    ex = adapt_example_for_mode(
        _ex(
            required_topics=["Variables and Data Types"],
            gold_topics=["Variables and Data Types"],
            gold_dependencies=[],
        ),
        "fair",
    )
    assert match_topic("Programming Fundamentals", ex) is None
    c = classify_unmatched_topic("Programming Fundamentals", ex)
    assert c["proposed_classification"] == "CONCEPT_ABSTRACTION"
    assert c["candidate_alias"] is False


def test_related_but_distinct_remain_unmatched():
    ex = adapt_example_for_mode(
        _ex(required_topics=["Probability"], gold_topics=["Probability"], gold_dependencies=[]),
        "fair",
    )
    assert match_topic("Statistics", ex) is None
    c = classify_unmatched_topic("Statistics", ex)
    assert c["proposed_classification"] in {
        "CONCEPT_DECOMPOSITION",
        "SEMANTICALLY_RELATED_BUT_DISTINCT",
        "GENUINE_HALLUCINATION",
        "UNKNOWN",
    }
    assert c["candidate_alias"] is False


def test_genuine_hallucination_unmatched():
    ex = adapt_example_for_mode(
        _ex(required_topics=["Control Flow"], gold_topics=["Control Flow"], gold_dependencies=[]),
        "fair",
    )
    assert match_topic("Underwater Basket Weaving", ex) is None
    c = classify_unmatched_topic("Underwater Basket Weaving", ex)
    assert c["proposed_classification"] == "GENUINE_HALLUCINATION"
    assert c["candidate_alias"] is False


def test_strict_strips_aliases_fair_keeps_them():
    base = _ex(topic_aliases={"Control Flow": ["Control Structures"]})
    strict = adapt_example_for_mode(base, "strict")
    fair = adapt_example_for_mode(base, "fair")
    assert match_topic("Control Structures", strict) is None or match_topic("Control Structures", strict) != "Control Flow"
    # Without alias, Jaccard may still fail for Structures vs Flow
    assert match_topic("Control Structures", fair) == "Control Flow"


def test_alias_exact_only_does_not_fuzzy_accept_related_phrase(tmp_path: Path):
    """Curated alias 'Asynchronous Processing' must not accept 'Error Handling in …'."""
    path = tmp_path / "a.json"
    path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "canonical": "Async Processing",
                        "aliases": ["Asynchronous Processing", "Introduction to Asynchronous Processing"],
                        "approved": True,
                    },
                ],
            },
        ),
        encoding="utf-8",
    )
    reg = load_curated_aliases(path)
    base = _ex(
        required_topics=["Async Processing"],
        gold_topics=["Async Processing"],
        gold_dependencies=[],
        topic_aliases={},
    )
    curated = adapt_example_for_mode(base, "curated_alias", curated_registry=reg)
    assert match_topic("Introduction to Asynchronous Processing", curated) == "Async Processing"
    assert match_topic("Asynchronous Processing", curated) == "Async Processing"
    assert match_topic("Error Handling in Asynchronous Processing", curated) is None


def test_curated_mode_adds_approved_aliases(tmp_path: Path):
    path = tmp_path / "a.json"
    path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "canonical": "XSS",
                        "aliases": ["Cross-Site Scripting (XSS)"],
                        "approved": True,
                        "reason": "acronym",
                        "classification": "EXACT_ALIAS",
                    },
                ],
            },
        ),
        encoding="utf-8",
    )
    reg = load_curated_aliases(path)
    base = _ex(required_topics=["XSS"], gold_topics=["XSS"], gold_dependencies=[], topic_aliases={})
    fair = adapt_example_for_mode(base, "fair", curated_registry=reg)
    curated = adapt_example_for_mode(base, "curated_alias", curated_registry=reg)
    assert match_topic("Cross-Site Scripting (XSS)", fair) is None
    assert match_topic("Cross-Site Scripting (XSS)", curated) == "XSS"


def test_alias_does_not_fix_wrong_dependency_direction():
    ex = adapt_example_for_mode(
        _ex(
            required_topics=["Control Flow", "Variables and Data Types"],
            gold_topics=["Control Flow", "Variables and Data Types"],
            required_dependencies=[("Control Flow", "Variables and Data Types")],
            gold_dependencies=[("Control Flow", "Variables and Data Types")],
            topic_aliases={"Control Flow": ["Control Structures"]},
        ),
        "fair",
    )
    gen = GeneratedGraph(
        topics=["Control Structures", "Variables and Data Types"],
        dependencies=[("Variables and Data Types", "Control Structures")],
    )
    s = score_graph(ex, gen)
    assert "WRONG_DEPENDENCY_DIRECTION" in s.failures
    assert s.dependency_recall == 0.0


def test_rescore_matching_modes_without_api(tmp_path: Path):
    # Minimal synthetic benchmark artifact
    artifact = {
        "timestamp": "t",
        "benchmark_type": "quality",
        "model": "test-model",
        "dataset": "learning_graph_quality_v1",
        "systems": {
            "synapse": {
                "example_results": [
                    {
                        "example_id": "python_basics_001",
                        "generated_topics": ["Control Structures", "Variables", "Functions", "Data Structures"],
                        "generated_dependencies": [
                            ["Control Structures", "Variables"],
                            ["Functions", "Control Structures"],
                        ],
                        "parse_ok": True,
                    },
                ],
                "metrics": {},
            },
        },
    }
    src = tmp_path / "bench.json"
    src.write_text(json.dumps(artifact), encoding="utf-8")
    ds = Path(__file__).resolve().parents[2] / "data" / "eval" / "learning_graph_quality_v1.jsonl"
    assert ds.is_file(), ds
    out = rescore_matching_modes(
        src,
        modes=["strict", "fair", "curated_alias"],
        dataset_path=ds,
        system="synapse",
        output_dir=tmp_path,
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "comparison_table" in data
    assert "fair" in data["metrics_by_mode"]
    assert data["notes"]
    assert out.with_suffix(".md").is_file()
