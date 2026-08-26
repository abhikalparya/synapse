"""Prompt A/B + redundant-transitive metrics (no live API calls)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

from app.evaluation.baselines import generate_direct_llm_raw
from app.evaluation.benchmark import evaluate_example, run_benchmark
from app.evaluation.inspect import classify_comparison
from app.evaluation.metrics import compare_graphs, find_redundant_transitive_edges, score_graph
from app.evaluation.prompt_compare import build_variant_comparison_table
from app.evaluation.schemas import EvalExample, GeneratedGraph
from app.evaluation.reporting import render_markdown_report, write_benchmark_result
from app.prompts.ingest import (
    INGEST_CONCEPT_DIRECT_PREREQUISITE,
    INGEST_JSON_SCHEMA,
    build_ingest_prompt,
    prompt_metadata,
    prompt_version_hash,
    resolve_prompt_variant,
)
from app.services.llm import LLMCallRecord


def _ex(**kwargs) -> EvalExample:
    defaults = dict(
        id="t",
        category="programming",
        difficulty="beginner",
        goal="g",
        gold_topics=["A", "B", "C"],
        gold_dependencies=[("B", "A"), ("C", "B")],
    )
    defaults.update(kwargs)
    return EvalExample(**defaults)  # type: ignore[arg-type]


def test_resolve_prompt_variant_aliases():
    assert resolve_prompt_variant("baseline") == "baseline"
    assert resolve_prompt_variant("concept_direct_prerequisite") == "concept_direct_prerequisite"
    assert resolve_prompt_variant("concept") == "concept_direct_prerequisite"
    assert resolve_prompt_variant("concept-direct") == "concept_direct_prerequisite"


def test_prompt_bodies_differ_and_hash_stable():
    assert "curriculum-design assistant" in INGEST_JSON_SCHEMA
    assert "reusable concepts" in INGEST_CONCEPT_DIRECT_PREREQUISITE
    assert "DIRECT prerequisites" in INGEST_CONCEPT_DIRECT_PREREQUISITE or "direct" in INGEST_CONCEPT_DIRECT_PREREQUISITE.lower()
    b = prompt_metadata("baseline")
    c = prompt_metadata("concept_direct_prerequisite")
    assert b["prompt_variant"] == "baseline"
    assert c["prompt_variant"] == "concept_direct_prerequisite"
    assert b["prompt_hash"] != c["prompt_hash"]
    assert b["prompt_hash"] == prompt_version_hash(INGEST_JSON_SCHEMA)
    assert "Introduction to…" in INGEST_CONCEPT_DIRECT_PREREQUISITE or "Introduction" in INGEST_CONCEPT_DIRECT_PREREQUISITE


def test_build_ingest_prompt_selects_variant():
    base = build_ingest_prompt("Learn Python", variant="baseline")
    concept = build_ingest_prompt("Learn Python", variant="concept_direct_prerequisite")
    assert "curriculum-design assistant" in base
    assert "learning-dependency graph designer" in concept
    assert "Goal / content:" in base and "Goal / content:" in concept


def test_find_redundant_transitive_respects_from_requires_to():
    # B requires A, C requires B, C requires A → C→A is transitive shortcut
    edges = [("B", "A"), ("C", "B"), ("C", "A")]
    red = find_redundant_transitive_edges(edges)
    assert red == [("C", "A")]

    # Necessary chain only — no redundant
    assert find_redundant_transitive_edges([("B", "A"), ("C", "B")]) == []

    # Longer path A requires B requires C requires D; A→D redundant; A→C redundant
    long = [("A", "B"), ("B", "C"), ("C", "D"), ("A", "C"), ("A", "D")]
    red2 = set(find_redundant_transitive_edges(long))
    assert ("A", "C") in red2
    assert ("A", "D") in red2
    assert ("A", "B") not in red2
    assert ("B", "C") not in red2


def test_direct_edge_not_redundant_without_alternate_path():
    # Diamond: C requires A and C requires B; A and B both require F — neither C→A nor C→B is transitive
    edges = [("A", "F"), ("B", "F"), ("C", "A"), ("C", "B")]
    assert find_redundant_transitive_edges(edges) == []


def test_score_graph_redundant_metrics_and_failures():
    ex = _ex()
    gen = GeneratedGraph(
        topics=["A", "B", "C"],
        dependencies=[("B", "A"), ("C", "B"), ("C", "A")],
    )
    s = score_graph(ex, gen)
    assert s.redundant_transitive_edge_count == 1
    assert s.redundant_transitive_edge_rate == 1 / 3
    assert "REDUNDANT_TRANSITIVE_EDGE" in s.failures
    assert "EXTRA_DEPENDENCY" not in s.failures
    assert s.extra_dependency_rate == 1 / 3


def test_inspect_classifies_extra_redundant_and_reversed():
    ex = _ex()
    # Extra non-redundant: A→C (wrong direction / unrelated); reverse B←A already covered elsewhere
    gen = GeneratedGraph(
        topics=["A", "B", "C"],
        dependencies=[("B", "A"), ("C", "B"), ("C", "A"), ("A", "B")],
    )
    comp = compare_graphs(ex, gen)
    fails = classify_comparison(ex, comp)
    cats = {f["category"] for f in fails}
    assert "REDUNDANT_TRANSITIVE_EDGE" in cats
    assert "WRONG_DEPENDENCY_DIRECTION" in cats
    # C→A should be classified as redundant (not only EXTRA)
    redundant_gens = [f["generated"] for f in fails if f["category"] == "REDUNDANT_TRANSITIVE_EDGE"]
    assert any("C" in g and "A" in g for g in redundant_gens)


def test_benchmark_artifact_records_prompt_metadata(tmp_path: Path):
    example = _ex(
        id="transformers_001",
        gold_topics=["A", "B"],
        gold_dependencies=[("B", "A")],
    )
    graph_json = json.dumps(
        {
            "topics": [
                {"title": "A", "summary": "s", "confidence": 0.9},
                {"title": "B", "summary": "s", "confidence": 0.8},
            ],
            "dependencies": [{"from": "B", "to": "A"}],
        },
    )

    async def fake_detailed(prompt: str, *, temperature=None, seed=None):
        assert "learning-dependency graph designer" in prompt
        return LLMCallRecord(
            text=graph_json,
            latency_ms=1.0,
            provider="openai",
            model="gpt-4o-mini",
            input_tokens=5,
            output_tokens=5,
            tokens_estimated=False,
            estimated_cost_usd=0.0,
            success=True,
            operation="test",
        )

    async def run():
        with patch("app.evaluation.baselines.call_llm_detailed", new=AsyncMock(side_effect=fake_detailed)):
            return await run_benchmark(
                [example],
                systems=["direct_llm_graph", "synapse"],
                repetitions=1,
                temperature=0.0,
                seed=42,
                include_ops_latency=False,
                dataset_name="test_ds",
                model="gpt-4o-mini",
                prompt_variant="concept_direct_prerequisite",
            )

    result = asyncio.run(run())
    assert result["prompt_variant"] == "concept_direct_prerequisite"
    assert result["prompt_hash"]
    assert result["prompt_version"].startswith("concept_direct_prerequisite@")
    path = write_benchmark_result(result, tmp_path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["prompt_variant"] == "concept_direct_prerequisite"
    md = render_markdown_report(loaded)
    assert "concept_direct_prerequisite" in md


def test_evaluate_example_passes_variant_into_prompt():
    example = _ex(gold_topics=["A", "B"], gold_dependencies=[("B", "A")])
    captured: list[str] = []

    async def fake_detailed(prompt: str, *, temperature=None, seed=None):
        captured.append(prompt)
        return LLMCallRecord(
            text='{"topics":[{"title":"A","summary":"s","confidence":0.9},{"title":"B","summary":"s","confidence":0.8}],'
            '"dependencies":[{"from":"B","to":"A"}]}',
            latency_ms=1.0,
            provider="openai",
            model="m",
            input_tokens=1,
            output_tokens=1,
            tokens_estimated=True,
            estimated_cost_usd=None,
            success=True,
            operation="t",
        )

    async def run():
        with patch("app.evaluation.baselines.call_llm_detailed", new=AsyncMock(side_effect=fake_detailed)):
            await evaluate_example(
                example,
                systems=["synapse"],
                repetition=0,
                temperature=0.0,
                seed=1,
                prompt_variant="baseline",
            )
            await evaluate_example(
                example,
                systems=["synapse"],
                repetition=0,
                temperature=0.0,
                seed=1,
                prompt_variant="concept_direct_prerequisite",
            )

    asyncio.run(run())
    assert len(captured) == 2
    assert "curriculum-design assistant" in captured[0]
    assert "learning-dependency graph designer" in captured[1]


def test_variant_comparison_table_delta():
    baseline = {
        "systems": {
            "synapse": {
                "metrics": {"topic_f1": 0.5, "dependency_f1": 0.1, "extra_dependency_rate": 0.8,
                            "redundant_transitive_edge_rate": 0.2, "missing_prerequisite_rate": 0.5,
                            "dependency_direction_error_rate": 0.05, "hallucinated_topic_rate": 0.1,
                            "topic_precision": 0.5, "topic_recall": 0.5,
                            "dependency_precision": 0.1, "dependency_recall": 0.1},
                "latency": {"p50_ms": 100},
                "cost": {"average_cost_usd": 0.001},
            },
        },
    }
    concept = {
        "systems": {
            "synapse": {
                "metrics": {"topic_f1": 0.6, "dependency_f1": 0.2, "extra_dependency_rate": 0.5,
                            "redundant_transitive_edge_rate": 0.05, "missing_prerequisite_rate": 0.4,
                            "dependency_direction_error_rate": 0.04, "hallucinated_topic_rate": 0.05,
                            "topic_precision": 0.6, "topic_recall": 0.6,
                            "dependency_precision": 0.2, "dependency_recall": 0.2},
                "latency": {"p50_ms": 120},
                "cost": {"average_cost_usd": 0.0012},
            },
        },
    }
    rows = {r["metric"]: r for r in build_variant_comparison_table(baseline, concept)}
    assert abs(rows["topic_f1"]["delta"] - 0.1) < 1e-9
    assert abs(rows["dependency_f1"]["delta"] - 0.1) < 1e-9
    assert rows["extra_dependency_rate"]["delta"] < 0


def test_default_production_ingest_still_baseline():
    """Production service path omits variant → baseline body."""
    prompt = build_ingest_prompt("goal text")
    assert "curriculum-design assistant" in prompt


def test_generate_direct_llm_raw_records_prompt_meta():
    async def fake_detailed(prompt: str, *, temperature=None, seed=None):
        return LLMCallRecord(
            text="{}",
            latency_ms=1.0,
            provider="p",
            model="m",
            input_tokens=1,
            output_tokens=1,
            tokens_estimated=True,
            estimated_cost_usd=None,
            success=True,
            operation="t",
        )

    async def run():
        with patch("app.evaluation.baselines.call_llm_detailed", new=AsyncMock(side_effect=fake_detailed)):
            _text, meta = await generate_direct_llm_raw(_ex(), prompt_variant="concept")
        return meta

    meta = asyncio.run(run())
    assert meta["prompt_variant"] == "concept_direct_prerequisite"
    assert meta["prompt_hash"]
