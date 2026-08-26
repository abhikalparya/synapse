from app.evaluation.baselines import linear_topics_to_dependencies, parse_direct_dependency_graph, parse_linear_roadmap
from app.evaluation.synapse_system import run_synapse_from_raw


def test_parse_linear_numbered_and_bullets():
    text = """1. Linear Algebra
2) Probability
- Neural Networks
* Attention
"""
    topics = parse_linear_roadmap(text)
    assert topics == ["Linear Algebra", "Probability", "Neural Networks", "Attention"]
    deps = linear_topics_to_dependencies(topics)
    assert deps[0] == ("Probability", "Linear Algebra")
    assert deps[-1] == ("Attention", "Neural Networks")


def test_parse_linear_empty_raises():
    try:
        parse_linear_roadmap("sorry I cannot help")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_direct_keeps_cycle_and_self_loop():
    raw = """
    {
      "topics": [
        {"title": "A", "summary": "", "confidence": 0.9},
        {"title": "B", "summary": "", "confidence": 0.9}
      ],
      "dependencies": [
        {"from": "A", "to": "B"},
        {"from": "B", "to": "A"},
        {"from": "A", "to": "A"},
        {"from": "A", "to": "Missing"}
      ]
    }
    """
    graph = parse_direct_dependency_graph(raw)
    assert ("B", "A") in graph.dependencies
    assert ("A", "A") in graph.dependencies
    assert ("A", "Missing") in graph.dependencies


def test_synapse_strips_cycles_self_loops_and_unknown_refs():
    raw = """
    {
      "topics": [
        {"title": "A", "summary": "a", "confidence": 0.9},
        {"title": "B", "summary": "b", "confidence": 0.9}
      ],
      "dependencies": [
        {"from": "A", "to": "B"},
        {"from": "B", "to": "A"},
        {"from": "A", "to": "A"},
        {"from": "A", "to": "Ghost"}
      ]
    }
    """
    graph = run_synapse_from_raw(raw)
    assert graph.parse_ok
    assert ("A", "B") in graph.dependencies
    assert ("B", "A") not in graph.dependencies
    assert ("A", "A") not in graph.dependencies
    reasons = " ".join(s["reason"] for s in graph.skipped_dependencies)
    assert "cycle" in reasons or "itself" in reasons.lower() or "unknown" in reasons
    assert any("unknown" in s["reason"] for s in graph.skipped_dependencies)
