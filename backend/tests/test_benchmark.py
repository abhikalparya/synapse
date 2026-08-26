import asyncio
from unittest.mock import AsyncMock, patch

from app.evaluation.baselines import run_direct_from_raw
from app.evaluation.benchmark import evaluate_example
from app.evaluation.schemas import EvalExample
from app.evaluation.synapse_system import run_synapse_from_raw
from app.services.llm import LLMCallRecord


def _example() -> EvalExample:
    return EvalExample(
        id="transformers_001",
        category="machine_learning",
        difficulty="intermediate",
        goal="Learn transformer-based language models",
        gold_topics=["Linear Algebra", "Neural Networks", "Transformers"],
        gold_dependencies=[("Neural Networks", "Linear Algebra"), ("Transformers", "Neural Networks")],
    )


GRAPH_JSON = """
{
  "topics": [
    {"title": "Linear Algebra", "summary": "s", "confidence": 0.9},
    {"title": "Neural Networks", "summary": "s", "confidence": 0.8},
    {"title": "Transformers", "summary": "s", "confidence": 0.7}
  ],
  "dependencies": [
    {"from": "Neural Networks", "to": "Linear Algebra"},
    {"from": "Transformers", "to": "Neural Networks"},
    {"from": "Linear Algebra", "to": "Transformers"}
  ]
}
"""

LINEAR_TEXT = """1. Linear Algebra
2. Neural Networks
3. Transformers
"""


def _record(text: str) -> LLMCallRecord:
    return LLMCallRecord(
        text=text,
        latency_ms=12.0,
        provider="openai",
        model="gpt-4o-mini",
        input_tokens=10,
        output_tokens=20,
        tokens_estimated=False,
        estimated_cost_usd=0.0001,
        success=True,
        operation="test",
    )


def test_benchmark_with_mocked_llm_does_not_need_api_keys():
    example = _example()

    async def fake_detailed(prompt: str, *, temperature=None, seed=None):
        if "numbered list" in prompt.lower() or "ordered learning roadmap" in prompt.lower():
            return _record(LINEAR_TEXT)
        return _record(GRAPH_JSON)

    async def run():
        with patch("app.evaluation.baselines.call_llm_detailed", new=AsyncMock(side_effect=fake_detailed)):
            results = await evaluate_example(
                example,
                systems=["linear_baseline", "direct_llm_graph", "synapse"],
                repetition=0,
                temperature=0.0,
                seed=42,
            )
        return results

    results = asyncio.run(run())
    assert results["linear_baseline"].scores is not None
    assert results["linear_baseline"].scores.graph_valid is True
    # Direct keeps the cycle Linear Algebra -> Transformers closing A->B->C->A
    assert results["direct_llm_graph"].scores is not None
    assert results["direct_llm_graph"].scores.cycle_attempt is True
    assert results["direct_llm_graph"].scores.graph_valid is False
    # Synapse validation drops the cycle-creating edge
    assert results["synapse"].scores is not None
    assert results["synapse"].scores.graph_valid is True
    assert results["synapse"].graph.skipped_dependencies


def test_direct_and_synapse_split_on_same_raw():
    direct = run_direct_from_raw(GRAPH_JSON)
    synapse = run_synapse_from_raw(GRAPH_JSON)
    assert len(direct.dependencies) > len(synapse.dependencies)
