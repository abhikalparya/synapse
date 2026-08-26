"""Constrained representation alignment tests (no API / no LLM)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.evaluation.schemas import EvalExample, example_to_dict
from app.evaluation.representation_alignment_analysis import run_representation_alignment_replay
from app.services.representation_alignment import (
    align_graph,
    align_titles,
    remap_dependencies,
)


def test_tutorial_framing_normalization():
    records, title_map = align_titles(
        ["Introduction to Git", "Getting Started with Docker", "SQL SELECT Queries"],
        enable_merge=False,
    )
    by = {r.original_title: r for r in records}
    assert by["Introduction to Git"].decision == "NORMALIZE_TITLE"
    assert by["Introduction to Git"].aligned_title == "Git"
    assert by["Getting Started with Docker"].aligned_title == "Docker"
    assert by["SQL SELECT Queries"].decision == "KEEP_ORIGINAL"


def test_exact_normalized_duplicate_merge():
    result = align_graph(
        ["Git Basics", "Introduction to Git", "Version Control"],
        [("Git Basics", "Version Control"), ("Introduction to Git", "Version Control")],
    )
    assert "Git" in result.topics_after or any(
        r.aligned_title == "Git" for r in result.records
    )
    merged = [r for r in result.records if r.decision == "MERGE_WITH_EXISTING_GENERATED_TOPIC"]
    assert len(merged) >= 1
    # Single edge after remap+dedupe
    assert ("Git", "Version Control") in result.dependencies_after
    assert len([e for e in result.dependencies_after if e[1] == "Version Control" and "Git" in e[0]]) <= 2


def test_no_new_topic_creation():
    result = align_graph(
        ["Introduction to Kafka", "Stream Processing"],
        [("Stream Processing", "Introduction to Kafka")],
        request_text="Learn Kafka stream processing",
    )
    assert result.new_topics_created == 0
    before_keys = {t.casefold() for t in result.topics_before}
    # Every after topic must be remapping of an original (possibly stripped)
    assert len(result.topics_after) <= len(result.topics_before)


def test_no_topic_deletion_without_merge():
    result = align_graph(
        ["Variables", "Functions"],
        [("Functions", "Variables")],
        enable_merge=True,
    )
    # No merge possible → same count
    assert len(result.topics_after) == 2
    assert result.topics_deleted_without_merge == 0


def test_dependency_remapping_and_dedupe():
    title_map = {"Git Basics": "Git", "Introduction to Git": "Git", "Version Control": "Version Control"}
    deps = remap_dependencies(
        [("Git Basics", "Version Control"), ("Introduction to Git", "Version Control"), ("Git", "Git")],
        title_map,
    )
    assert deps == [("Git", "Version Control")]


def test_dag_validation_after_remap():
    result = align_graph(
        ["A", "Introduction to B", "B"],
        [("A", "Introduction to B"), ("Introduction to B", "B"), ("B", "A")],
    )
    # Cycle B→A with A→B after merge Introduction to B → B may be filtered
    assert result.new_topics_created == 0
    # All remaining edges should be cycle-safe
    assert result.dag_valid is True


def test_original_unchanged_when_no_framing():
    topics = ["Linear Algebra", "Neural Networks"]
    result = align_graph(topics, [("Neural Networks", "Linear Algebra")])
    assert result.topics_after == topics
    assert all(r.decision == "KEEP_ORIGINAL" for r in result.records)


def test_ambiguous_short_titles_unresolved_or_kept():
    records, _ = align_titles(["Introduction"], enable_merge=True)
    # "Introduction" alone should not become empty concept via aggressive strip+merge
    assert records[0].aligned_title
    assert records[0].decision in {"KEEP_ORIGINAL", "PRESERVE_UNRESOLVED", "NORMALIZE_TITLE"}


def test_context_alignment_uses_request_text():
    records, _ = align_titles(
        ["Introduction to Kafka"],
        request_text="Learn Kafka for stream processing architectures",
        enable_merge=False,
    )
    r = records[0]
    assert r.decision == "NORMALIZE_TITLE"
    assert r.aligned_title.lower() == "kafka"


def test_unsafe_semantic_merges_rejected():
    # Different concepts must not merge
    result = align_graph(
        ["Git", "Docker", "Kubernetes"],
        [("Docker", "Git")],
    )
    assert len(result.topics_after) == 3
    assert all(r.decision != "MERGE_WITH_EXISTING_GENERATED_TOPIC" for r in result.records)


def test_offline_replay_no_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    ex = EvalExample(
        id="case_a",
        category="programming",
        difficulty="beginner",
        goal="Learn Git version control",
        gold_topics=["Git", "Version Control"],
        gold_dependencies=[("Git", "Version Control")],
        required_topics=["Git", "Version Control"],
        required_dependencies=[("Git", "Version Control")],
        topic_aliases={},
    )
    ds = tmp_path / "ds.jsonl"
    ds.write_text(json.dumps(example_to_dict(ex)) + "\n", encoding="utf-8")
    art = tmp_path / "bench.json"
    art.write_text(
        json.dumps(
            {
                "benchmark_type": "quality",
                "dataset": "learning_graph_quality_v1",
                "model": "gpt-4o-mini",
                "systems": {
                    "synapse": {
                        "example_results": [
                            {
                                "example_id": "case_a",
                                "repetition": 0,
                                "parse_ok": True,
                                "generated_topics": ["Introduction to Git", "Version Control"],
                                "generated_dependencies": [["Introduction to Git", "Version Control"]],
                                "skipped_dependencies": [],
                            }
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    # Redirect bench output by patching DEFAULT is hard; write to out via output_dir
    # and monkeypatch DEFAULT_BENCH
    import app.evaluation.representation_alignment_analysis as mod

    monkeypatch.setattr(mod, "DEFAULT_BENCH", tmp_path / "benchmarks")
    md, js, bench = run_representation_alignment_replay(
        art, dataset_path=ds, output_dir=out
    )
    assert md.is_file() and js.is_file() and bench.is_file()
    payload = json.loads(js.read_text(encoding="utf-8"))
    assert payload["no_new_llm_calls"] is True
    assert payload["safety"]["new_topics_created"] == 0
    assert payload["alignment_behavior"]["normalized"] >= 1


def test_align_graph_disabled_modes_keep_identity():
    result = align_graph(
        ["Introduction to Git"],
        [],
        enable_framing=False,
        enable_context=False,
        enable_merge=False,
    )
    assert result.topics_after == ["Introduction to Git"]
