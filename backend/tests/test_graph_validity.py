from app.evaluation.metrics import assess_graph_validity


def test_valid_dag():
    v = assess_graph_validity(["A", "B", "C"], [("B", "A"), ("C", "B")])
    assert v.is_valid is True
    assert v.cycles == 0
    assert v.self_loops == 0
    assert v.invalid_references == 0


def test_self_loop():
    v = assess_graph_validity(["A"], [("A", "A")])
    assert v.is_valid is False
    assert v.self_loops == 1


def test_simple_cycle():
    v = assess_graph_validity(["A", "B"], [("A", "B"), ("B", "A")])
    assert v.is_valid is False
    assert v.cycles == 1


def test_multi_node_cycle():
    v = assess_graph_validity(
        ["A", "B", "C"],
        [("A", "B"), ("B", "C"), ("C", "A")],
    )
    assert v.is_valid is False
    assert v.cycles == 1


def test_invalid_references():
    v = assess_graph_validity(["A"], [("A", "Z")])
    assert v.is_valid is False
    assert v.invalid_references == 1
