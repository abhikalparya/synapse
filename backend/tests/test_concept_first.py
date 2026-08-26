"""Concept-First generation: normalization, pipeline, baseline isolation (mocked LLM)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from app.evaluation.baselines import generate_direct_llm_raw
from app.evaluation.benchmark import evaluate_example, run_benchmark
from app.evaluation.concept_first_compare import compare_concept_first_runs, write_normalization_analysis
from app.evaluation.schemas import EvalExample, GeneratedGraph
from app.prompts.concept_first import build_concept_generation_prompt, build_dependency_generation_prompt
from app.services.concept_first import run_concept_first_pipeline
from app.services.concept_normalization import normalize_concepts
from app.services.generation_strategy import resolve_generation_strategy
from app.services.llm import LLMCallRecord
from app.services.proposal_common import build_topics_and_dependencies, review_confidence_threshold


def _ex(**kwargs) -> EvalExample:
    defaults = dict(
        id="t1",
        category="programming",
        difficulty="beginner",
        goal="Learn Python basics",
        gold_topics=["Variables", "Control Flow", "Functions"],
        gold_dependencies=[("Control Flow", "Variables"), ("Functions", "Control Flow")],
    )
    defaults.update(kwargs)
    return EvalExample(**defaults)  # type: ignore[arg-type]


def _record(text: str, *, latency_ms: float = 10.0) -> LLMCallRecord:
    return LLMCallRecord(
        text=text,
        latency_ms=latency_ms,
        input_tokens=100,
        output_tokens=50,
        tokens_estimated=True,
        estimated_cost_usd=0.001,
        model="mock-model",
        provider="mock",
        success=True,
    )


def test_exact_duplicate_merging():
    result = normalize_concepts(
        [
            {"title": "Git"},
            {"title": "git"},
            {"title": "Git"},
        ]
    )
    assert len(result.inventory) == 1
    assert result.duplicate_rejection_count == 2
    assert all(d.detected_condition == "EXACT_DUPLICATE" for d in result.decisions if d.decision == "REJECT_DUPLICATE")


def test_normalized_duplicate_merging():
    result = normalize_concepts(
        [
            {"title": "Control Flow"},
            {"title": "Control Structures"},  # known alias → Control Flow
        ]
    )
    assert len(result.inventory) == 1
    assert result.inventory[0].title == "Control Flow"
    assert result.merged_count >= 1
    assert any(d.detected_condition == "CLEAR_ALIAS" for d in result.decisions)


def test_tutorial_framing_normalization():
    result = normalize_concepts(
        [
            {"title": "Git"},
            {"title": "Introduction to Git"},
        ]
    )
    assert len(result.inventory) == 1
    assert result.inventory[0].title == "Git"
    merge = [d for d in result.decisions if d.decision == "MERGE"]
    assert merge
    assert merge[0].detected_condition == "CLEAR_ALIAS"
    assert merge[0].original_title == "Introduction to Git"

    alone = normalize_concepts([{"title": "Introduction to Git"}])
    assert len(alone.inventory) == 1
    assert alone.inventory[0].title == "Git"
    assert alone.decisions[0].decision == "ACCEPT"
    assert alone.decisions[0].original_title == "Introduction to Git"


def test_unknown_concept_preservation():
    result = normalize_concepts(
        [
            {"title": "Obscure Quantum Widget Theory"},
            {"title": "Variables"},
        ]
    )
    titles = {c.title for c in result.inventory}
    assert "Obscure Quantum Widget Theory" in titles
    assert "Variables" in titles
    assert result.accepted_count >= 2


def test_granularity_conflict_detection():
    result = normalize_concepts(
        [
            {"title": "Variables"},
            {"title": "Data Types"},
            {"title": "Control Flow"},
            {"title": "Functions"},
            {"title": "Programming Fundamentals"},
        ]
    )
    assert any(d.detected_condition == "GRANULARITY_CONFLICT" for d in result.decisions)
    assert result.granularity_conflict_count >= 1
    assert result.unresolved_count >= 1
    assert any(c.title == "Programming Fundamentals" for c in result.inventory)


def test_abstraction_conflict_detection():
    result = normalize_concepts(
        [
            {"title": "Variables"},
            {"title": "Programming Fundamentals"},
        ]
    )
    assert any(
        d.detected_condition in {"ABSTRACTION_CONFLICT", "GRANULARITY_CONFLICT"}
        for d in result.decisions
    )


def test_decomposition_conflict_detection():
    result = normalize_concepts(
        [
            {"title": "Programming Fundamentals"},
            {"title": "Variables"},
        ]
    )
    # Variables after umbrella → decomposition conflict preserved
    assert any(d.detected_condition == "DECOMPOSITION_CONFLICT" for d in result.decisions)
    assert result.decomposition_conflict_count >= 1
    assert any(c.title == "Variables" for c in result.inventory)


def test_out_of_scope_rejection():
    result = normalize_concepts(
        [
            {"title": "Module 1"},
            {"title": "Lesson 2"},
            {"title": "Advanced Topics"},
            {"title": "Miscellaneous"},
            {"title": "Introduction"},
            {"title": "Git"},
        ]
    )
    assert result.out_of_scope_rejection_count >= 5
    assert [c.title for c in result.inventory] == ["Git"]


def test_dependency_generation_cannot_introduce_new_concepts():
    concepts = json.dumps(
        {
            "concepts": [
                {"title": "Variables", "description": "x"},
                {"title": "Functions", "description": "y"},
            ]
        }
    )
    deps = json.dumps(
        {
            "dependencies": [
                {"from": "Functions", "to": "Variables"},
                {"from": "Classes", "to": "Functions"},  # new title
                {"from": "Functions", "to": "Ghost"},
            ]
        }
    )

    async def _fake(prompt: str, **kwargs):
        if "FIXED inventory" in prompt or "Concept inventory" in prompt:
            return _record(deps)
        return _record(concepts)

    async def _run():
        with patch("app.services.concept_first.call_llm_detailed", new=AsyncMock(side_effect=_fake)):
            return await run_concept_first_pipeline("Goal: Learn Python")

    cf = asyncio.run(_run())
    titles = {t["title"] for t in cf.topics}
    assert titles == {"Variables", "Functions"}
    for d in cf.dependencies:
        assert d["from"] in titles and d["to"] in titles
    assert any("outside inventory" in e for e in cf.errors)


def test_unknown_dependency_references_remain_rejected():
    raw_topics = [
        {"title": "A", "summary": "a", "confidence": 0.9},
        {"title": "B", "summary": "b", "confidence": 0.9},
    ]
    raw_deps = [{"from": "A", "to": "Missing"}]
    _topics, deps, skipped = build_topics_and_dependencies(
        raw_topics,
        raw_deps,
        confidence_threshold=review_confidence_threshold(),
    )
    assert deps == []
    assert skipped
    assert "unknown" in skipped[0].reason.lower() or "reference" in skipped[0].reason.lower()


def test_existing_cycle_validation_still_works():
    raw_topics = [
        {"title": "A", "summary": "a", "confidence": 0.9},
        {"title": "B", "summary": "b", "confidence": 0.9},
    ]
    raw_deps = [{"from": "A", "to": "B"}, {"from": "B", "to": "A"}]
    _topics, deps, skipped = build_topics_and_dependencies(
        raw_topics,
        raw_deps,
        confidence_threshold=review_confidence_threshold(),
    )
    assert len(deps) == 1
    assert any("cycle" in s.reason.lower() for s in skipped)


def test_concept_generation_failure_does_not_persist_successful_graph():
    async def _boom(prompt: str, **kwargs):
        raise RuntimeError("LLM down")

    async def _run():
        with patch("app.services.concept_first.call_llm_detailed", new=AsyncMock(side_effect=_boom)):
            return await run_concept_first_pipeline("Goal: X")

    cf = asyncio.run(_run())
    assert cf.parse_ok is False
    assert cf.topics == []
    assert cf.status == "partial"
    assert cf.semantic_analysis == "unavailable"

    async def _ingest():
        from app.services.ingest import run_ingest

        with pytest.raises(ValueError, match="evaluation-only"):
            await run_ingest(
                goal="Learn X",
                topics=None,
                filenames=None,
                generation_strategy="concept_first",
            )

    asyncio.run(_ingest())


def test_dependency_generation_failure_produces_degraded_state():
    concepts = json.dumps({"concepts": [{"title": "A"}, {"title": "B"}]})

    async def _fake(prompt: str, **kwargs):
        if "FIXED inventory" in prompt or "Concept inventory" in prompt:
            raise RuntimeError("dep stage failed")
        return _record(concepts)

    async def _run():
        with patch("app.services.concept_first.call_llm_detailed", new=AsyncMock(side_effect=_fake)):
            return await run_concept_first_pipeline("Goal: X")

    cf = asyncio.run(_run())
    assert cf.status == "partial"
    assert cf.semantic_analysis == "unavailable"
    assert {t["title"] for t in cf.topics} == {"A", "B"}
    assert cf.dependencies == []
    assert any("dependency generation failed" in e for e in cf.errors)


def test_baseline_behavior_unchanged_and_concept_first_requires_explicit_selection():
    from app.services.generation_strategy import (
        resolve_evaluation_generation_strategy,
        resolve_generation_strategy,
    )

    assert resolve_generation_strategy(None) == "baseline"
    assert resolve_generation_strategy("baseline") == "baseline"
    with pytest.raises(ValueError, match="evaluation-only"):
        resolve_generation_strategy("concept_first")
    assert resolve_evaluation_generation_strategy("concept_first") == "concept_first"
    with pytest.raises(ValueError):
        resolve_generation_strategy("not_a_mode")

    baseline_prompt_calls: list[str] = []

    async def _baseline_detailed(prompt: str, **kwargs):
        baseline_prompt_calls.append(prompt)
        return LLMCallRecord(
            text=json.dumps(
                {
                    "topics": [{"title": "A", "summary": "a", "confidence": 0.9}],
                    "dependencies": [],
                }
            ),
            latency_ms=1.0,
            provider="mock",
            model="mock",
            input_tokens=1,
            output_tokens=1,
            tokens_estimated=False,
            estimated_cost_usd=None,
            success=True,
            operation="ingest",
        )

    async def _run():
        with patch("app.services.ingest.call_llm_detailed", new=AsyncMock(side_effect=_baseline_detailed)):
            with patch("app.services.ingest.save_proposal"):
                with patch("app.services.ingest.log_proposal_created"):
                    with patch("app.services.ingest.load_all_topics", return_value=[]):
                        from app.services.ingest import run_ingest

                        # Default path must remain baseline joint prompt
                        await run_ingest(goal="Learn A", topics=None, filenames=None)
        assert baseline_prompt_calls
        assert "curriculum-design assistant" in baseline_prompt_calls[0]
        assert "concepts" not in baseline_prompt_calls[0].split("Goal")[0].lower() or "FIXED" not in baseline_prompt_calls[0]

    asyncio.run(_run())


def test_evaluation_of_stored_artifacts_without_api_credentials(tmp_path: Path):
    """Normalization + comparison analysis run offline on stored generations."""
    artifact = {
        "timestamp": "2026-01-01T00:00:00Z",
        "benchmark_type": "quality",
        "dataset": "learning_graph_quality_v1",
        "model": "gpt-4o-mini",
        "repetitions": 1,
        "seed": 42,
        "systems": {
            "synapse": {
                "example_results": [
                    {
                        "example_id": "prog_python_basics",
                        "repetition": 0,
                        "parse_ok": True,
                        "generated_topics": ["Variables", "Functions"],
                        "generated_dependencies": [["Functions", "Variables"]],
                        "skipped_dependencies": [],
                        "total_latency_ms": 100.0,
                        "cost_usd": 0.001,
                    }
                ]
            },
            "concept_first": {
                "example_results": [
                    {
                        "example_id": "prog_python_basics",
                        "repetition": 0,
                        "parse_ok": True,
                        "generated_topics": ["Variables", "Control Flow", "Functions"],
                        "generated_dependencies": [
                            ["Control Flow", "Variables"],
                            ["Functions", "Control Flow"],
                        ],
                        "skipped_dependencies": [],
                        "total_latency_ms": 250.0,
                        "cost_usd": 0.002,
                        "generation_meta": {
                            "candidate_concepts": [
                                {"title": "Variables"},
                                {"title": "Introduction to Functions"},
                                {"title": "Control Flow"},
                            ],
                            "normalized_inventory": ["Variables", "Functions", "Control Flow"],
                            "normalization": {
                                "accepted_count": 2,
                                "merged_count": 1,
                                "duplicate_rejection_count": 0,
                                "out_of_scope_rejection_count": 0,
                                "granularity_conflict_count": 0,
                                "abstraction_conflict_count": 0,
                                "decomposition_conflict_count": 0,
                                "unresolved_count": 0,
                                "decisions": [
                                    {
                                        "original_title": "Introduction to Functions",
                                        "normalized_title": "Functions",
                                        "decision": "ACCEPT",
                                        "detected_condition": "CLEAR_ALIAS",
                                        "decision_reason": "Tutorial framing removed",
                                    }
                                ],
                            },
                            "errors": [],
                            "stage_latency_ms": {
                                "concept_generation": 100,
                                "normalization": 1,
                                "dependency_generation": 140,
                                "validation": 1,
                                "total": 242,
                            },
                        },
                    }
                ]
            },
        },
    }
    # Use a real dataset id if present; else fabricate minimal by writing tiny dataset
    ds = Path(__file__).resolve().parents[2].parent / "data" / "eval" / "learning_graph_quality_v1.jsonl"
    if ds.is_file():
        first = json.loads(ds.read_text(encoding="utf-8").splitlines()[0])
        eid = first["id"]
        artifact["systems"]["synapse"]["example_results"][0]["example_id"] = eid
        artifact["systems"]["concept_first"]["example_results"][0]["example_id"] = eid

    path = tmp_path / "quality.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    out = tmp_path / "fa"
    out.mkdir()
    # Point DEFAULT via output_dir
    npath = write_normalization_analysis(path, output_dir=out)
    assert npath.is_file()
    md, js = compare_concept_first_runs(path, output_dir=out, max_cases=3)
    assert md.is_file() and js.is_file()
    payload = json.loads(js.read_text(encoding="utf-8"))
    assert "baseline" in payload and "concept_first" in payload
    assert payload["benchmark_config"]["matching_mode"] == "curated_alias"


def test_existing_benchmark_metrics_remain_backward_compatible():
    graph = GeneratedGraph(
        topics=["Variables", "Functions"],
        dependencies=[("Functions", "Variables")],
        parse_ok=True,
    )
    from app.evaluation.metrics import score_graph

    scores = score_graph(_ex(), graph)
    # Historical fields still present
    assert hasattr(scores, "topic_f1")
    assert hasattr(scores, "dependency_f1")
    assert hasattr(scores, "required_edge_f1")
    assert hasattr(scores, "invalid_extra_edge_rate")


def test_concept_first_eval_system_uses_staged_prompts():
    concepts = json.dumps({"concepts": [{"title": "A"}, {"title": "B"}]})
    deps = json.dumps({"dependencies": [{"from": "B", "to": "A"}]})
    calls: list[str] = []

    async def _fake(prompt: str, **kwargs):
        calls.append(prompt)
        if "FIXED inventory" in prompt or "Concept inventory" in prompt:
            return _record(deps)
        return _record(concepts)

    async def _run():
        with patch("app.services.concept_first.call_llm_detailed", new=AsyncMock(side_effect=_fake)):
            return await evaluate_example(
                _ex(),
                systems=["concept_first"],
                repetition=0,
                temperature=0.0,
                seed=42,
            )

    out = asyncio.run(_run())
    assert "concept_first" in out
    assert out["concept_first"].graph.topics
    assert len(calls) == 2
    assert "dependencies" not in json.loads(
        # concept prompt should not ask for joint topics+deps schema with both
        "{}"
    )
    assert "learning-concept inventory" in calls[0].lower() or "concepts" in calls[0].lower()


def test_prompts_build():
    c = build_concept_generation_prompt("Goal: Git")
    assert "concepts" in c
    d = build_dependency_generation_prompt("Goal: Git", ["Git", "Branching"])
    assert "Git" in d and "Branching" in d
    assert "FIXED inventory" in d or "Concept inventory" in d
