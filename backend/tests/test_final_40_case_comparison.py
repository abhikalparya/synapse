"""Final 40-case comparison tests — no API / no LLM."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.evaluation.dataset import load_dataset
from app.evaluation.final_40_case_comparison import (
    REQUIRED_SYSTEMS,
    FinalBenchmarkError,
    build_final_comparison,
    case_winner,
    regression_label,
    relative_delta,
    validate_final_artifact,
)
from app.evaluation.metrics import score_graph
from app.evaluation.reliability import run_reliability_benchmark
from app.evaluation.schemas import EvalExample, GeneratedGraph
from app.services.generation_strategy import resolve_generation_strategy


DS = Path(__file__).resolve().parents[2] / "data" / "eval" / "learning_graph_quality_v1.jsonl"


def test_production_baseline_unchanged():
    assert resolve_generation_strategy(None) == "baseline"
    assert resolve_generation_strategy("baseline") == "baseline"


def test_relative_and_regression_helpers():
    assert relative_delta(0.18, 0.10) == pytest.approx(0.8)
    assert relative_delta(0.1, 0.0) is None
    assert regression_label(0.20, 0.10) == "IMPROVED"
    assert regression_label(0.05, 0.10) == "REGRESSED"
    assert regression_label(0.11, 0.10) == "UNCHANGED"
    assert case_winner({"baseline": 0.1, "domain_prior": 0.2, "edge_classifier": 0.15}) == "domain_prior"
    assert case_winner({"baseline": 0.2, "domain_prior": 0.2}) == "TIE"


def test_missing_edge_rate_in_unit_interval_and_no_duplicate_inflation():
    ex = EvalExample(
        id="t",
        category="x",
        difficulty="beginner",
        goal="g",
        gold_topics=["A", "B"],
        gold_dependencies=[("A", "B")],
        required_topics=["A", "B"],
        required_dependencies=[("A", "B")],
    )
    g = GeneratedGraph(
        topics=["A", "B"],
        dependencies=[("A", "B"), ("A", "B")],
        parse_ok=True,
    )
    sc = score_graph(ex, g)
    assert 0.0 <= sc.missing_required_edge_rate <= 1.0
    assert sc.required_edge_recall == pytest.approx(1.0)
    assert sc.matched_dependencies == 1


def test_validate_requires_three_systems_and_40_cases():
    examples = load_dataset(DS)
    assert len(examples) == 40
    payload = {"systems": {}, "generations": 3}
    with pytest.raises(FinalBenchmarkError):
        validate_final_artifact(payload, expected_cases=40, expected_generations=3)

    def rows(sys: str):
        out = []
        for ex in examples:
            for g in range(3):
                out.append(
                    {
                        "example_id": ex.id,
                        "generation_index": g,
                        "repetition": g,
                        "seed": 42 + g,
                        "seed_supported": True,
                        "parse_ok": sys == "synapse",
                        "generated_topics": ["Variables"] if sys == "synapse" else [],
                        "generated_dependencies": [],
                        "total_latency_ms": 10.0,
                        "llm_latency_ms": 8.0,
                        "deterministic_latency_ms": 2.0,
                        "cost_usd": 0.001,
                        "input_tokens": 100,
                        "output_tokens": 50,
                        "generation_meta": {
                            "seed": 42 + g,
                            "inventory_version": "v1" if sys != "synapse" else None,
                            "prompt_version": "edge_classifier_baseline@abc"
                            if sys.endswith("classifier")
                            else "baseline",
                            "prompt_variant": "edge_classifier_baseline"
                            if sys.endswith("classifier")
                            else "baseline",
                        },
                    }
                )
        return out

    good = {
        "systems": {s: {"example_results": rows(s)} for s in REQUIRED_SYSTEMS},
        "generations": 3,
        "model": "gpt-4o-mini",
    }
    validate_final_artifact(good, expected_cases=40, expected_generations=3)

    bad = json.loads(json.dumps(good))
    bad["systems"]["synapse"]["example_results"] = [
        r for r in bad["systems"]["synapse"]["example_results"] if r["generation_index"] != 2
    ]
    with pytest.raises(FinalBenchmarkError):
        validate_final_artifact(bad, expected_cases=40, expected_generations=3)


def test_final_comparison_aggregates_and_case_winners(tmp_path: Path):
    examples = load_dataset(DS)

    def rows(sys: str):
        out = []
        for ex in examples:
            if sys == "synapse":
                topics = list(ex.required_topic_list())[:1]
                deps: list = []
            elif sys == "domain_curriculum_prior":
                topics = list(ex.required_topic_list())
                deps = [list(d) for d in ex.required_dependency_list()]
            else:
                topics = []
                deps = []
            out.append(
                {
                    "example_id": ex.id,
                    "generation_index": 0,
                    "repetition": 0,
                    "seed": 42,
                    "seed_supported": True,
                    "parse_ok": bool(topics),
                    "generated_topics": topics,
                    "generated_dependencies": deps,
                    "total_latency_ms": 100.0 if sys == "synapse" else 200.0,
                    "llm_latency_ms": 90.0,
                    "deterministic_latency_ms": 10.0,
                    "cost_usd": 0.001 if sys == "synapse" else 0.002,
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "generation_meta": {
                        "seed": 42,
                        "inventory_version": "v1",
                        "prompt_variant": "edge_classifier_baseline",
                        "prompt_version": "edge_classifier_baseline@test",
                    },
                }
            )
        return out

    payload = {
        "systems": {s: {"example_results": rows(s)} for s in REQUIRED_SYSTEMS},
        "generations": 1,
        "model": "gpt-4o-mini",
    }
    art = tmp_path / "quality.json"
    art.write_text(json.dumps(payload), encoding="utf-8")
    json_path, md_path = build_final_comparison(art, output_dir=tmp_path)
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert set(report["overall"]) == set(REQUIRED_SYSTEMS)
    assert len(report["case_results"]) == 40
    assert report["win_counts"].get("domain_prior", 0) > report["win_counts"].get("baseline", 0)
    assert report["frozen_configuration"]["selection"] == "INDEPENDENT"
    assert report["frozen_configuration"]["cases"] == 40
    assert "compiler_construction" in report["domain_summary"] or any(
        "compiler" in k for k in report["domain_summary"]
    )
    assert report["latency_cost"]["synapse"]["estimated_cost_per_case_usd"] > 0
    assert report["latency_cost"]["synapse"]["total"]["p50_ms"] > 0
    for _sys, agg in report["overall"].items():
        assert 0.0 <= agg["missing_required_edge_rate"] <= 1.0
    assert md_path.is_file()
    text = md_path.read_text(encoding="utf-8")
    assert "Executive summary" in text
    assert "Final architecture decision" in text


def test_reliability_suite_unchanged_and_no_api():
    result = run_reliability_benchmark()
    m = result["metrics"]
    assert m["validation_catch_rate"] == 1.0
    assert m["cycle_prevention_rate"] == 1.0
    assert m["transaction_integrity_rate"] == 1.0
    assert m["rollback_correctness_rate"] == 1.0


def test_domain_aggregation_groups_unmapped_by_category(tmp_path: Path):
    examples = load_dataset(DS)
    payload = {
        "systems": {
            s: {
                "example_results": [
                    {
                        "example_id": ex.id,
                        "generation_index": 0,
                        "seed": 1,
                        "parse_ok": False,
                        "generated_topics": [],
                        "generated_dependencies": [],
                        "total_latency_ms": 1.0,
                        "llm_latency_ms": 0.0,
                        "deterministic_latency_ms": 1.0,
                        "cost_usd": 0.0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "generation_meta": {
                            "seed": 1,
                            "inventory_version": "v1",
                            "prompt_variant": "baseline",
                        },
                    }
                    for ex in examples
                ]
            }
            for s in REQUIRED_SYSTEMS
        },
        "generations": 1,
        "model": "gpt-4o-mini",
    }
    art = tmp_path / "q.json"
    art.write_text(json.dumps(payload), encoding="utf-8")
    json_path, _ = build_final_comparison(art, output_dir=tmp_path)
    report = json.loads(json_path.read_text())
    keys = set(report["domain_summary"])
    assert "compiler_construction" in keys
    assert any(k.startswith("unmapped:") for k in keys)
