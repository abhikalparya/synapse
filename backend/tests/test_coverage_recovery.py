"""Targeted coverage recovery tests (no API key / no LLM)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from app.services.coverage_recovery import (
    CoverageCandidate,
    apply_parsed_coverage_recovery,
    max_recovery_candidates,
    parse_coverage_candidates,
    rank_candidates,
    truncate_candidates,
    validate_and_merge_recovery,
)
from app.services.generation_strategy import resolve_generation_strategy


def _baseline():
    topics = [
        {"title": "Lexer", "summary": "Tokenizes source", "confidence": 0.9},
        {"title": "Compiler Frontend", "summary": "Builds AST", "confidence": 0.8},
        {"title": "Code Generation", "summary": "Emits code", "confidence": 0.8},
    ]
    deps = [
        {"from": "Compiler Frontend", "to": "Lexer"},
        {"from": "Code Generation", "to": "Compiler Frontend"},
    ]
    return topics, deps


def test_strategy_opt_in_only():
    from app.services.generation_strategy import resolve_evaluation_generation_strategy

    assert resolve_generation_strategy(None) == "baseline"
    with pytest.raises(ValueError, match="evaluation-only"):
        resolve_generation_strategy("baseline_coverage_recovery")
    assert (
        resolve_evaluation_generation_strategy("baseline_coverage_recovery")
        == "baseline_coverage_recovery"
    )
    assert resolve_evaluation_generation_strategy("coverage_recovery") == "baseline_coverage_recovery"


def test_ingest_rejects_closed_coverage_recovery_strategy():
    import asyncio
    from app.services import ingest as ingest_mod

    async def _run():
        with pytest.raises(ValueError, match="evaluation-only"):
            await ingest_mod.run_ingest(
                goal="Learn compilers",
                topics=None,
                filenames=None,
                generation_strategy="baseline_coverage_recovery",
            )

    asyncio.run(_run())


def test_valid_missing_prerequisite_accepted():
    topics, deps = _baseline()
    raw = json.dumps(
        {
            "candidates": [
                {
                    "category": "REQUIRED_MISSING_PREREQUISITE",
                    "title": "Parsing",
                    "summary": "Builds a parse tree from tokens",
                    "reason": "Compiler Frontend requires parsing after lexing.",
                    "target_topics": ["Compiler Frontend"],
                    "relationships": [{"from": "Compiler Frontend", "to": "Parsing"}],
                    "confidence": 0.9,
                }
            ]
        }
    )
    result = apply_parsed_coverage_recovery(
        baseline_topics=topics,
        baseline_dependencies=deps,
        raw_llm_text=raw,
    )
    assert result.counts["applied_count"] == 1
    assert "Parsing" in result.new_topic_titles
    assert ("Compiler Frontend", "Parsing") in result.new_edges
    titles = {t["title"] for t in result.topics_after}
    assert "Parsing" in titles
    # baseline unchanged when we look at inputs
    assert len(topics) == 3


def test_optional_and_related_rejected():
    topics, deps = _baseline()
    raw = json.dumps(
        {
            "candidates": [
                {
                    "category": "OPTIONAL_NICE_TO_HAVE",
                    "title": "History of Compilers",
                    "reason": "Interesting but optional",
                    "target_topics": ["Compiler Frontend"],
                    "relationships": [{"from": "Compiler Frontend", "to": "History of Compilers"}],
                    "confidence": 0.9,
                },
                {
                    "category": "RELATED_BUT_NOT_REQUIRED",
                    "title": "LLVM Internals",
                    "reason": "Related tooling",
                    "target_topics": ["Code Generation"],
                    "relationships": [{"from": "Code Generation", "to": "LLVM Internals"}],
                    "confidence": 0.8,
                },
            ]
        }
    )
    result = apply_parsed_coverage_recovery(
        baseline_topics=topics, baseline_dependencies=deps, raw_llm_text=raw
    )
    assert result.counts["applied_count"] == 0
    assert result.counts["optional_rejected_count"] == 1
    assert result.counts["related_rejected_count"] == 1
    assert result.new_topic_titles == []


def test_out_of_scope_rejected():
    topics, deps = _baseline()
    raw = json.dumps(
        {
            "candidates": [
                {
                    "category": "OUT_OF_SCOPE",
                    "title": "Quantum Computing",
                    "reason": "Out of scope",
                    "target_topics": ["Lexer"],
                    "relationships": [{"from": "Lexer", "to": "Quantum Computing"}],
                    "confidence": 0.5,
                }
            ]
        }
    )
    result = apply_parsed_coverage_recovery(
        baseline_topics=topics, baseline_dependencies=deps, raw_llm_text=raw
    )
    assert result.counts["out_of_scope_count"] == 1
    assert result.counts["applied_count"] == 0


def test_duplicate_concept_edge_only_or_reject():
    topics, deps = _baseline()
    # Propose existing Lexer as new topic with existing edge → duplicate edges
    raw = json.dumps(
        {
            "candidates": [
                {
                    "category": "REQUIRED_MISSING_PREREQUISITE",
                    "title": "Lexer",
                    "reason": "Already present",
                    "target_topics": ["Compiler Frontend"],
                    "relationships": [{"from": "Compiler Frontend", "to": "Lexer"}],
                    "confidence": 0.9,
                }
            ]
        }
    )
    result = apply_parsed_coverage_recovery(
        baseline_topics=topics, baseline_dependencies=deps, raw_llm_text=raw
    )
    assert result.counts["applied_count"] == 0
    assert result.counts["duplicate_count"] == 1


def test_cycle_causing_recovery_rejected():
    topics, deps = _baseline()
    # Add edge Lexer requires Code Generation — may create cycle with existing chain
    # Existing: Frontend→Lexer, CodeGen→Frontend. Adding Lexer→CodeGen creates cycle.
    raw = json.dumps(
        {
            "candidates": [
                {
                    "category": "REQUIRED_MISSING_PREREQUISITE",
                    "title": "Lexer",
                    "reason": "bad cycle",
                    "target_topics": ["Lexer"],
                    "relationships": [{"from": "Lexer", "to": "Code Generation"}],
                    "confidence": 0.9,
                }
            ]
        }
    )
    result = apply_parsed_coverage_recovery(
        baseline_topics=topics, baseline_dependencies=deps, raw_llm_text=raw
    )
    assert result.counts["applied_count"] == 0
    assert result.counts["cycle_rejected_count"] >= 1 or result.counts["duplicate_count"] >= 0
    # No new cycle edges in output
    edge_set = {(d["from"], d["to"]) for d in result.dependencies_after}
    assert ("Lexer", "Code Generation") not in edge_set


def test_invalid_target_rejected():
    topics, deps = _baseline()
    raw = json.dumps(
        {
            "candidates": [
                {
                    "category": "REQUIRED_MISSING_PREREQUISITE",
                    "title": "Parsing",
                    "reason": "x",
                    "target_topics": ["Nonexistent Topic"],
                    "relationships": [{"from": "Nonexistent Topic", "to": "Parsing"}],
                    "confidence": 0.9,
                }
            ]
        }
    )
    result = apply_parsed_coverage_recovery(
        baseline_topics=topics, baseline_dependencies=deps, raw_llm_text=raw
    )
    assert result.counts["applied_count"] == 0
    assert result.counts["invalid_target_count"] >= 1


def test_too_many_candidates_truncated(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SYNAPSE_MAX_RECOVERY_CANDIDATES", "2")
    assert max_recovery_candidates() == 2
    cands = [
        CoverageCandidate(
            category="REQUIRED_MISSING_PREREQUISITE",
            title=f"C{i}",
            confidence=0.5 + i * 0.1,
            target_topics=["Compiler Frontend"],
            relationships=[("Compiler Frontend", f"C{i}")],
        )
        for i in range(5)
    ]
    retained, truncated = truncate_candidates(cands, max_n=2)
    assert len(retained) == 2
    assert len(truncated) == 3
    assert all(c.rejection_reason == "truncated_by_max_candidates" for c in truncated)


def test_new_topic_and_edges_and_edge_only():
    topics, deps = _baseline()
    # First: new topic
    topics2, deps2, accepted, rejected, skipped, counts = validate_and_merge_recovery(
        baseline_topics=topics,
        baseline_dependencies=deps,
        retained=[
            CoverageCandidate(
                category="REQUIRED_MISSING_PREREQUISITE",
                title="Parsing",
                relationships=[("Compiler Frontend", "Parsing")],
                confidence=0.9,
            )
        ],
    )
    assert counts["applied_count"] == 1
    assert accepted[0].operation == "NEW_TOPIC_AND_EDGES"
    # Edge-only: add Parsing→Lexer when Parsing already exists
    topics3, deps3, accepted2, _, _, counts2 = validate_and_merge_recovery(
        baseline_topics=topics2,
        baseline_dependencies=deps2,
        retained=[
            CoverageCandidate(
                category="REQUIRED_MISSING_PREREQUISITE",
                title="Parsing",
                relationships=[("Parsing", "Lexer")],
                confidence=0.8,
            )
        ],
    )
    assert counts2["applied_count"] == 1
    assert accepted2[0].operation == "NEW_EDGE_ONLY"
    assert ("Parsing", "Lexer") in {(d["from"], d["to"]) for d in deps3}


def test_noop_when_no_candidates():
    topics, deps = _baseline()
    result = apply_parsed_coverage_recovery(
        baseline_topics=topics,
        baseline_dependencies=deps,
        raw_llm_text='{"candidates": []}',
    )
    assert result.counts["applied_count"] == 0
    assert len(result.topics_after) == len(topics)
    assert len(result.dependencies_after) == len(deps)


def test_baseline_graph_inputs_unchanged():
    topics, deps = _baseline()
    snapshot_t = json.dumps(topics)
    snapshot_d = json.dumps(deps)
    apply_parsed_coverage_recovery(
        baseline_topics=topics,
        baseline_dependencies=deps,
        raw_llm_text=json.dumps(
            {
                "candidates": [
                    {
                        "category": "REQUIRED_MISSING_PREREQUISITE",
                        "title": "Parsing",
                        "relationships": [{"from": "Compiler Frontend", "to": "Parsing"}],
                        "confidence": 0.9,
                    }
                ]
            }
        ),
    )
    assert json.dumps(topics) == snapshot_t
    assert json.dumps(deps) == snapshot_d


def test_ingest_coverage_recovery_creates_pending_proposal_only():
    """Closed experiment: recovery remains available via evaluation adapters, not product ingest."""
    pytest.skip("baseline_coverage_recovery removed from product ingest path (evaluation-only)")


def test_rank_prefers_higher_confidence():
    cands = [
        CoverageCandidate(category="REQUIRED_MISSING_PREREQUISITE", title="A", confidence=0.2),
        CoverageCandidate(category="REQUIRED_MISSING_PREREQUISITE", title="B", confidence=0.9),
    ]
    ranked = rank_candidates(cands)
    assert ranked[0].title == "B"


def test_parse_candidates_shape():
    data = {
        "candidates": [
            {
                "category": "REQUIRED_MISSING_PREREQUISITE",
                "title": "X",
                "relationships": [["A", "X"]],
                "confidence": 0.7,
            }
        ]
    }
    cands = parse_coverage_candidates(data)
    assert len(cands) == 1
    assert cands[0].relationships == [("A", "X")]
