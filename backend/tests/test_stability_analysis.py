"""Baseline generation stability analysis tests (no API key / no LLM)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from app.evaluation.schemas import EvalExample, GeneratedGraph, SystemExampleResult, example_to_dict
from app.evaluation.stability_analysis import (
    _distribution,
    _jaccard,
    _mean_pairwise_jaccard,
    classify_case_stability,
    run_baseline_stability_analysis,
)


def _ex(**kwargs) -> EvalExample:
    defaults = dict(
        id="case_a",
        category="programming",
        difficulty="beginner",
        goal="Learn the basics of Python programming",
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
        topic_aliases={"Control Flow": ["Control Structures"]},
    )
    defaults.update(kwargs)
    return EvalExample(**defaults)  # type: ignore[arg-type]


def _write_ds(path: Path, *examples: EvalExample) -> None:
    path.write_text(
        "\n".join(json.dumps(example_to_dict(ex)) for ex in examples) + "\n",
        encoding="utf-8",
    )


def _row(
    *,
    example_id: str,
    repetition: int,
    topics: list[str],
    deps: list[list[str]],
    seed: int | None = None,
    seed_supported: bool | None = True,
    latency: float = 100.0,
    cost: float = 0.001,
    output_tokens: int = 200,
) -> dict:
    meta = {
        "seed": seed if seed is not None else 42 + repetition,
        "seed_supported": seed_supported,
        "generation_index": repetition,
        "prompt_variant": "baseline",
    }
    return {
        "example_id": example_id,
        "repetition": repetition,
        "generation_index": repetition,
        "seed": meta["seed"],
        "seed_supported": seed_supported,
        "parse_ok": True,
        "error": None,
        "failures": [],
        "scores": None,
        "total_latency_ms": latency,
        "llm_latency_ms": latency,
        "deterministic_latency_ms": 0.0,
        "cost_usd": cost,
        "cost_estimated": True,
        "input_tokens": 100,
        "output_tokens": output_tokens,
        "generated_topics": topics,
        "generated_dependencies": deps,
        "skipped_dependencies": [],
        "generation_meta": meta,
    }


def _artifact(rows: list[dict], *, repetitions: int = 3) -> dict:
    return {
        "benchmark_type": "quality_stability",
        "dataset": "learning_graph_quality_v1",
        "model": "gpt-4o-mini",
        "provider": "openai",
        "prompt_variant": "baseline",
        "temperature": 0.0,
        "seed": 42,
        "seed_supported": True,
        "repetitions": repetitions,
        "generations": repetitions,
        "systems": {"synapse": {"example_results": rows, "metrics": {}}},
    }


def test_distribution_mean_median_std():
    d = _distribution([0.0, 0.5, 1.0])
    assert d["mean"] == pytest.approx(0.5)
    assert d["median"] == pytest.approx(0.5)
    assert d["min"] == 0.0
    assert d["max"] == 1.0
    assert d["std_dev"] == pytest.approx((0.5**2 * 2 / 3) ** 0.5)
    empty = _distribution([])
    assert empty["mean"] is None
    assert empty["n"] == 0


def test_jaccard_and_pairwise():
    assert _jaccard(set(), set()) == 1.0
    assert _jaccard({"a"}, {"a"}) == 1.0
    assert _jaccard({"a"}, {"b"}) == 0.0
    assert _jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)
    assert _mean_pairwise_jaccard([{ "a"}, {"a"}, {"a"}]) == 1.0
    assert _mean_pairwise_jaccard([]) == 0.0


def test_classify_consistently_good():
    gens = [
        {"scores": {"topic_f1": 0.8, "required_edge_f1": 0.5}},
        {"scores": {"topic_f1": 0.82, "required_edge_f1": 0.48}},
        {"scores": {"topic_f1": 0.79, "required_edge_f1": 0.52}},
    ]
    assert classify_case_stability(gens) == "CONSISTENTLY_GOOD"


def test_classify_consistently_bad():
    gens = [
        {"scores": {"topic_f1": 0.2, "required_edge_f1": 0.05}},
        {"scores": {"topic_f1": 0.22, "required_edge_f1": 0.08}},
        {"scores": {"topic_f1": 0.18, "required_edge_f1": 0.1}},
    ]
    assert classify_case_stability(gens) == "CONSISTENTLY_BAD"


def test_classify_high_variance():
    gens = [
        {"scores": {"topic_f1": 0.2, "required_edge_f1": 0.1}},
        {"scores": {"topic_f1": 0.9, "required_edge_f1": 0.7}},
        {"scores": {"topic_f1": 0.5, "required_edge_f1": 0.4}},
    ]
    assert classify_case_stability(gens) == "HIGH_VARIANCE"


def test_multi_generation_rows_independent(tmp_path: Path):
    """Multiple generations for one case are stored and analyzed independently."""
    rows = [
        _row(
            example_id="case_a",
            repetition=0,
            topics=["Variables", "Control Flow", "Functions"],
            deps=[["Control Flow", "Variables"], ["Functions", "Control Flow"]],
            seed=42,
        ),
        _row(
            example_id="case_a",
            repetition=1,
            topics=["Variables", "Control Structures"],
            deps=[["Control Structures", "Variables"]],
            seed=43,
        ),
        _row(
            example_id="case_a",
            repetition=2,
            topics=["Variables", "Ghost Topic"],
            deps=[["Ghost Topic", "Variables"]],
            seed=44,
        ),
    ]
    art = tmp_path / "stability.json"
    art.write_text(json.dumps(_artifact(rows)), encoding="utf-8")
    ds = tmp_path / "ds.jsonl"
    _write_ds(ds, _ex())
    md, js = run_baseline_stability_analysis(art, dataset_path=ds, output_dir=tmp_path / "out")
    payload = json.loads(js.read_text(encoding="utf-8"))
    assert payload["benchmark_config"]["generations_per_case_observed"] == 3
    case = payload["cases"][0]
    assert len(case["generations"]) == 3
    assert [g["generation_index"] for g in case["generations"]] == [0, 1, 2]
    assert [g["seed"] for g in case["generations"]] == [42, 43, 44]
    assert payload["benchmark_config"]["seed_supported"] is True
    assert case["gold_topic_frequencies"]["Control Flow"] >= 2 / 3
    assert case["gold_topic_frequencies"]["Functions"] == pytest.approx(1 / 3)
    assert "Functions→Control Flow" in case["required_edge_frequencies"]
    assert payload["hallucination_persistence"]["one_off"] >= 1
    assert md.is_file()


def test_unsupported_seed_reported(tmp_path: Path):
    rows = [
        _row(
            example_id="case_a",
            repetition=i,
            topics=["Variables", "Control Flow", "Functions"],
            deps=[["Control Flow", "Variables"], ["Functions", "Control Flow"]],
            seed=None,
            seed_supported=False,
        )
        for i in range(3)
    ]
    for r in rows:
        r["seed"] = None
        r["generation_meta"]["seed"] = None
        r["generation_meta"]["seed_supported"] = False
    art = tmp_path / "a.json"
    art.write_text(json.dumps(_artifact(rows)), encoding="utf-8")
    ds = tmp_path / "ds.jsonl"
    _write_ds(ds, _ex())
    _, js = run_baseline_stability_analysis(art, dataset_path=ds, output_dir=tmp_path / "out")
    payload = json.loads(js.read_text(encoding="utf-8"))
    assert payload["benchmark_config"]["seed_supported"] is False


def test_stable_missing_and_repeated_invalid(tmp_path: Path):
    # Edge always missing; invalid edge repeats
    rows = [
        _row(
            example_id="case_a",
            repetition=i,
            topics=["Variables", "Control Flow", "Functions"],
            deps=[["Functions", "Variables"]],  # invalid shortcut; missing both required
            seed=42 + i,
        )
        for i in range(3)
    ]
    art = tmp_path / "a.json"
    art.write_text(json.dumps(_artifact(rows)), encoding="utf-8")
    ds = tmp_path / "ds.jsonl"
    _write_ds(ds, _ex())
    _, js = run_baseline_stability_analysis(art, dataset_path=ds, output_dir=tmp_path / "out")
    payload = json.loads(js.read_text(encoding="utf-8"))
    case = payload["cases"][0]
    assert set(case["stable_missing_edges"]) == {
        "Control Flow→Variables",
        "Functions→Control Flow",
    }
    assert payload["stable_missing_edge_count"] == 2
    assert payload["invalid_edge_persistence"]["repeated"] >= 1 or payload["invalid_edge_persistence"]["all_generations"] >= 1
    fp = payload["failure_persistence"]["missing_edges"]
    assert fp["all_generations"] >= 1


def test_failure_jaccard_identical_signatures(tmp_path: Path):
    rows = [
        _row(
            example_id="case_a",
            repetition=i,
            topics=["Variables"],
            deps=[],
            seed=42 + i,
        )
        for i in range(3)
    ]
    art = tmp_path / "a.json"
    art.write_text(json.dumps(_artifact(rows)), encoding="utf-8")
    ds = tmp_path / "ds.jsonl"
    _write_ds(ds, _ex())
    _, js = run_baseline_stability_analysis(art, dataset_path=ds, output_dir=tmp_path / "out")
    payload = json.loads(js.read_text(encoding="utf-8"))
    assert payload["error_signature_similarity"]["mean_topic_failure_jaccard"] == pytest.approx(1.0)
    assert payload["error_signature_similarity"]["mean_edge_failure_jaccard"] == pytest.approx(1.0)


def test_zero_denominator_safe():
    assert _jaccard(set(), {"a"}) == 0.0
    assert _distribution([])["cv"] is None


def test_single_generation_benchmark_unchanged_default():
    async def fake_eval(*args, **kwargs):
        g = GeneratedGraph(topics=["Variables"], dependencies=[], parse_ok=True)
        g.generation_meta = {"seed": 42, "seed_supported": True, "generation_index": 0}
        return {
            "synapse": SystemExampleResult(
                example_id="case_a",
                system="synapse",
                repetition=0,
                graph=g,
                scores=None,
                total_latency_ms=1.0,
                llm_latency_ms=1.0,
                deterministic_latency_ms=0.0,
                cost_usd=None,
                cost_estimated=True,
                input_tokens=None,
                output_tokens=None,
            )
        }

    from app.evaluation.benchmark import run_benchmark

    async def _run():
        with patch("app.evaluation.benchmark.evaluate_example", new=fake_eval):
            with patch("app.evaluation.benchmark.collect_proposal_metrics", return_value={}):
                return await run_benchmark(
                    [_ex()],
                    systems=["synapse"],
                    repetitions=1,
                    include_ops_latency=False,
                    ops_latency_samples=0,
                )

    result = asyncio.run(_run())
    assert result["benchmark_type"] == "quality"
    assert result["repetitions"] == 1
    row = result["systems"]["synapse"]["example_results"][0]
    assert row["generation_index"] == 0
    assert row["seed"] == 42


def test_stability_analysis_requires_no_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    rows = [
        _row(
            example_id="case_a",
            repetition=i,
            topics=["Variables", "Control Flow", "Functions"],
            deps=[["Control Flow", "Variables"], ["Functions", "Control Flow"]],
            seed=42 + i,
        )
        for i in range(2)
    ]
    art = tmp_path / "a.json"
    art.write_text(json.dumps(_artifact(rows, repetitions=2)), encoding="utf-8")
    ds = tmp_path / "ds.jsonl"
    _write_ds(ds, _ex())
    md, js = run_baseline_stability_analysis(art, dataset_path=ds, output_dir=tmp_path / "out")
    assert md.is_file() and js.is_file()
    assert "diagnosis" in json.loads(js.read_text(encoding="utf-8"))
