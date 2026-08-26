import json
from pathlib import Path

import pytest

from app.evaluation.dataset import DatasetError, load_dataset
from app.evaluation.golden_v1 import golden_v1
from app.evaluation.metrics import assess_graph_validity


def test_load_committed_dataset():
    examples = load_dataset()
    assert len(examples) >= 30
    ids = [e.id for e in examples]
    assert len(ids) == len(set(ids))
    categories = {e.category for e in examples}
    for needed in (
        "programming",
        "machine_learning",
        "mathematics",
        "databases",
        "distributed_systems",
        "cloud_computing",
        "frontend_engineering",
        "backend_engineering",
        "data_engineering",
        "security",
    ):
        assert needed in categories


def test_gold_graphs_are_valid_dags():
    for ex in golden_v1():
        v = assess_graph_validity(ex.gold_topics, ex.gold_dependencies)
        assert v.is_valid, f"{ex.id}: {v.details}"


def test_malformed_json(tmp_path: Path):
    p = tmp_path / "bad.jsonl"
    p.write_text("{not json\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="malformed JSON"):
        load_dataset(p)


def test_missing_required_fields(tmp_path: Path):
    p = tmp_path / "missing.jsonl"
    p.write_text(json.dumps({"id": "x", "goal": "g"}) + "\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="missing required field"):
        load_dataset(p)


def test_duplicate_ids(tmp_path: Path):
    row = {
        "id": "dup",
        "category": "programming",
        "difficulty": "beginner",
        "goal": "g",
        "gold_topics": ["A"],
        "gold_dependencies": [],
    }
    p = tmp_path / "dup.jsonl"
    p.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="duplicate example id"):
        load_dataset(p)


def test_dependency_must_reference_gold_topics(tmp_path: Path):
    row = {
        "id": "bad_edge",
        "category": "programming",
        "difficulty": "beginner",
        "goal": "g",
        "gold_topics": ["A"],
        "gold_dependencies": [["A", "Missing"]],
    }
    p = tmp_path / "edge.jsonl"
    p.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="references a topic not in gold_topics"):
        load_dataset(p)


def test_valid_fixture_dataset(tmp_path: Path):
    row = {
        "id": "ok_001",
        "category": "programming",
        "difficulty": "beginner",
        "goal": "Learn X",
        "gold_topics": ["A", "B"],
        "gold_dependencies": [["B", "A"]],
        "input_notes": None,
        "notes": "ok",
    }
    p = tmp_path / "ok.jsonl"
    p.write_text(json.dumps(row) + "\n", encoding="utf-8")
    examples = load_dataset(p)
    assert examples[0].id == "ok_001"
    assert examples[0].gold_dependencies == [("B", "A")]
    assert examples[0].required_topic_list() == ["A", "B"]


def test_aliases_optional_and_acceptable_fields(tmp_path: Path):
    row = {
        "id": "ext_001",
        "category": "programming",
        "difficulty": "beginner",
        "goal": "Learn X",
        "gold_topics": ["Control Flow", "Functions", "Python Basics"],
        "gold_dependencies": [["Functions", "Control Flow"], ["Python Basics", "Functions"]],
        "required_topics": ["Control Flow", "Functions"],
        "optional_topics": ["File I/O"],
        "aliases": {"Control Flow": ["Control Structures"]},
        "acceptable_dependencies": [["Functions", "Variables"]] if False else [["Python Basics", "Control Flow"]],
        "notes": "extended",
    }
    p = tmp_path / "ext.jsonl"
    p.write_text(json.dumps(row) + "\n", encoding="utf-8")
    ex = load_dataset(p)[0]
    assert ex.required_topic_list() == ["Control Flow", "Functions"]
    assert "File I/O" in ex.optional_topic_list()
    assert "Python Basics" in ex.optional_topic_list()
    assert "Control Structures" in ex.topic_aliases["Control Flow"]
    assert ("Python Basics", "Control Flow") in ex.acceptable_dependencies


def test_golden_v1_has_quality_annotations():
    examples = {e.id: e for e in golden_v1()}
    py = examples["python_basics_001"]
    assert py.required_topics is not None
    assert "Control Structures" in py.topic_aliases["Control Flow"]
    assert any("universally correct learning graph" in e.notes for e in examples.values())
