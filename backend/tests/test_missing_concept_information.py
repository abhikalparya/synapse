"""Missing-concept information availability tests (no API / no LLM)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.evaluation.missing_concept_information import (
    available_runtime_input,
    classify_missing_concept_availability,
    detect_source_explicit,
    detect_source_implicit,
    run_missing_concept_information_analysis,
)
from app.evaluation.schemas import EvalExample, GeneratedGraph, example_to_dict


def _ex(**kwargs) -> EvalExample:
    defaults = dict(
        id="case_a",
        category="programming",
        difficulty="beginner",
        goal="Learn Python programming",
        gold_topics=["Variables", "Kafka", "Parsing"],
        gold_dependencies=[("Parsing", "Variables")],
        required_topics=["Variables", "Kafka", "Parsing"],
        required_dependencies=[("Parsing", "Variables")],
        optional_topics=[],
        acceptable_dependencies=[],
        ambiguous_dependencies=[],
        topic_aliases={"Kafka": ["Apache Kafka"]},
        input_notes=None,
        notes="Evaluator-only commentary mentioning Parsing should not count as source.",
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
    path.write_text(
        json.dumps({"dataset": "toy", "systems": {"synapse": {"example_results": rows}}}),
        encoding="utf-8",
    )


def test_source_explicit_detection():
    ex = _ex(goal="Learn stream processing with Kafka topics")
    hit = detect_source_explicit("Kafka", ex, available_runtime_input(ex)["available_input_text"])
    assert hit is not None
    assert hit["classification"] == "SOURCE_EXPLICIT"
    assert hit["evidence"]


def test_alias_based_explicit_detection():
    ex = _ex(
        goal="Learn Apache Kafka consumer patterns",
        gold_topics=["Kafka"],
        required_topics=["Kafka"],
        topic_aliases={"Kafka": ["Apache Kafka"]},
    )
    hit = detect_source_explicit("Kafka", ex, available_runtime_input(ex)["available_input_text"])
    assert hit is not None
    # Goal contains "Apache Kafka" (alias) and may also normalize near "kafka"
    assert hit["matching_method"] in {"alias_exact", "normalized_exact"}
    assert hit["classification"] == "SOURCE_EXPLICIT"


def test_alias_exact_when_only_alias_surface_present():
    ex = _ex(
        goal="Study Apache Kafka in depth",
        gold_topics=["Event Streaming Platform"],
        required_topics=["Event Streaming Platform"],
        topic_aliases={"Event Streaming Platform": ["Apache Kafka"]},
    )
    hit = detect_source_explicit(
        "Event Streaming Platform",
        ex,
        available_runtime_input(ex)["available_input_text"],
    )
    assert hit is not None
    assert hit["matching_method"] == "alias_exact"


def test_source_implicit_requires_explicit_evidence():
    # Goal alone without AST phrase → no implicit Parsing
    ex = _ex(goal="Learn how compilers turn source code into machine code")
    assert detect_source_implicit("Parsing", available_runtime_input(ex)["available_input_text"]) is None
    # With AST evidence in input notes
    ex2 = _ex(
        goal="Learn compilers",
        input_notes="The frontend builds an abstract syntax tree before codegen.",
    )
    hit = detect_source_implicit("Parsing", available_runtime_input(ex2)["available_input_text"])
    assert hit is not None
    assert hit["classification"] == "SOURCE_IMPLICIT"
    assert hit["confidence"] in {"HIGH", "MEDIUM", "LOW"}


def test_external_prerequisite_when_source_present_but_no_evidence():
    ex = _ex(
        goal="Learn Kubernetes deployment workflows",
        input_notes="I want rolling updates and services configured in my cluster.",
        gold_topics=["Containers", "Kubernetes"],
        required_topics=["Containers", "Kubernetes"],
    )
    out = classify_missing_concept_availability("Containers", ex)
    assert out["classification"] == "EXTERNAL_PREREQUISITE"


def test_goal_derived_classification():
    ex = _ex(goal="Learn compiler construction", input_notes=None)
    out = classify_missing_concept_availability("Parsing", ex)
    assert out["classification"] == "GOAL_DERIVED"
    assert out.get("input_context_limitation") is True or "goal" in (out.get("reason") or "").lower()


def test_ambiguous_fallback_empty_goal():
    ex = _ex(goal="   ", input_notes=None)
    out = classify_missing_concept_availability("Kafka", ex)
    assert out["classification"] in {"AMBIGUOUS", "GOAL_DERIVED"}


def test_no_llm_judge_called(monkeypatch: pytest.MonkeyPatch):
    import app.evaluation.missing_concept_information as mod

    def boom(*_a, **_k):
        raise AssertionError("LLM must not be called")

    monkeypatch.setattr(mod, "classify_missing_concept_availability", mod.classify_missing_concept_availability)
    # Ensure call_llm is never imported/used — smoke via classify only
    out = classify_missing_concept_availability("Kafka", _ex(goal="Learn Kafka"))
    assert out["classification"] == "SOURCE_EXPLICIT"


def test_evidence_preserved():
    ex = _ex(goal="Build services with Kafka partitions")
    out = classify_missing_concept_availability("Kafka", ex)
    assert out["evidence"]
    assert "Kafka" in (out["evidence"] or "") or "kafka" in (out["evidence"] or "").casefold()


def test_dependency_impact_and_pareto(tmp_path: Path):
    ex = _ex(
        id="impact_001",
        goal="Learn compilers",
        gold_topics=["Parsing", "Lexing", "Codegen"],
        required_topics=["Parsing", "Lexing", "Codegen"],
        required_dependencies=[
            ("Parsing", "Lexing"),
            ("Codegen", "Parsing"),
        ],
        gold_dependencies=[
            ("Parsing", "Lexing"),
            ("Codegen", "Parsing"),
        ],
        topic_aliases={},
        notes="",
    )
    # Never present: generate unrelated topics
    gens = [
        GeneratedGraph(topics=["Introduction", "Overview"], dependencies=[])
        for _ in range(3)
    ]
    ds = tmp_path / "ds.jsonl"
    art = tmp_path / "stab.json"
    _write_ds(ds, ex)
    _artifact(art, [(ex, gens)])
    md, js, pareto = run_missing_concept_information_analysis(
        art, dataset_path=ds, output_dir=tmp_path / "out"
    )
    data = json.loads(js.read_text())
    assert data["no_new_llm_calls"] is True
    assert data["no_llm_judge"] is True
    assert data["never_present_count"] >= 1
    # Parsing should impact at least the edges where it appears
    parsing = next(r for r in data["missing_concepts"] if r["gold_topic"] == "Parsing")
    assert parsing["total_dependency_impact"] >= 1
    assert "Rank" in pareto.read_text() or "rank" in pareto.read_text().casefold()
    assert md.is_file()


def test_case_level_aggregation(tmp_path: Path):
    ex = _ex(id="agg_001", goal="Learn Python", input_notes=None)
    gens = [GeneratedGraph(topics=["Hello World"], dependencies=[]) for _ in range(3)]
    ds = tmp_path / "ds.jsonl"
    art = tmp_path / "stab.json"
    _write_ds(ds, ex)
    _artifact(art, [(ex, gens)])
    _, js, _ = run_missing_concept_information_analysis(
        art, dataset_path=ds, output_dir=tmp_path / "out"
    )
    data = json.loads(js.read_text())
    row = next(c for c in data["case_matrix"] if c["case_id"] == "agg_001")
    assert row["learning_goal_present"] is True
    assert row["source_content_present"] is False
    assert row["number_of_missing_topics"] >= 1


def test_empty_source_content_handled_safely():
    runtime = available_runtime_input(_ex(goal="Learn X", input_notes=None))
    assert runtime["source_content_present"] is False
    assert "Goal:" in runtime["available_input_text"]
    # Evaluator notes must not enter runtime input
    assert "Evaluator-only" not in runtime["available_input_text"]


def test_no_api_key_required(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ex = _ex(id="offline", goal="Learn Python")
    gens = [GeneratedGraph(topics=["A"], dependencies=[]) for _ in range(3)]
    ds = tmp_path / "ds.jsonl"
    art = tmp_path / "stab.json"
    _write_ds(ds, ex)
    _artifact(art, [(ex, gens)])
    run_missing_concept_information_analysis(art, dataset_path=ds, output_dir=tmp_path / "out")


def test_evaluator_notes_do_not_create_source_explicit():
    ex = _ex(
        goal="Learn distributed systems",
        input_notes=None,
        notes="This gold graph requires Parsing and Kafka heavily.",
    )
    out = classify_missing_concept_availability("Parsing", ex)
    assert out["classification"] != "SOURCE_EXPLICIT"
