from app.evaluation.metrics import assess_graph_validity, score_graph, topic_similarity
from app.evaluation.schemas import EvalExample, GeneratedGraph


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


def test_perfect_graph_match():
    ex = _ex()
    gen = GeneratedGraph(topics=["A", "B", "C"], dependencies=[("B", "A"), ("C", "B")])
    s = score_graph(ex, gen)
    assert s.topic_precision == 1.0
    assert s.topic_recall == 1.0
    assert s.topic_f1 == 1.0
    assert s.dependency_precision == 1.0
    assert s.dependency_recall == 1.0
    assert s.dependency_f1 == 1.0
    assert s.graph_valid is True
    assert s.missing_prerequisite_rate == 0.0
    assert s.hallucinated_topic_rate == 0.0


def test_partial_topic_match():
    ex = _ex()
    gen = GeneratedGraph(topics=["A", "B"], dependencies=[("B", "A")])
    s = score_graph(ex, gen)
    assert s.topic_precision == 1.0
    assert s.topic_recall == 2 / 3
    assert s.matched_topics == 2


def test_partial_dependency_match():
    ex = _ex()
    gen = GeneratedGraph(topics=["A", "B", "C"], dependencies=[("B", "A")])
    s = score_graph(ex, gen)
    assert s.dependency_precision == 1.0
    assert s.dependency_recall == 0.5
    assert "MISSING_PREREQUISITE" in s.failures


def test_duplicate_topic_handling():
    ex = _ex()
    gen = GeneratedGraph(topics=["A", "a", "B", "C"], dependencies=[("B", "A"), ("C", "B")])
    s = score_graph(ex, gen)
    assert s.duplicate_topics == 1
    assert s.generated_topics == 3
    assert "DUPLICATE_TOPIC" in s.failures


def test_empty_generated_graph():
    ex = _ex()
    gen = GeneratedGraph(topics=[], dependencies=[])
    s = score_graph(ex, gen)
    assert s.topic_precision == 1.0  # vacuously no false positives
    assert s.topic_recall == 0.0
    assert s.dependency_precision == 1.0
    assert s.dependency_recall == 0.0
    assert s.hallucinated_topic_rate == 0.0


def test_empty_gold_graph():
    ex = _ex(gold_topics=[], gold_dependencies=[])
    gen = GeneratedGraph(topics=["X"], dependencies=[])
    s = score_graph(ex, gen)
    assert s.topic_recall == 1.0
    assert s.topic_precision == 0.0
    assert s.hallucinated_topic_rate == 1.0
    assert "HALLUCINATED_TOPIC" in s.failures


def test_empty_gold_and_empty_generated():
    ex = _ex(gold_topics=[], gold_dependencies=[])
    gen = GeneratedGraph(topics=[], dependencies=[])
    s = score_graph(ex, gen)
    assert s.topic_precision == 1.0
    assert s.topic_recall == 1.0
    assert s.dependency_f1 == 1.0


def test_normalized_and_alias_match():
    ex = _ex(
        gold_topics=["Neural Networks", "Large Language Models"],
        gold_dependencies=[("Large Language Models", "Neural Networks")],
        topic_aliases={"Large Language Models": ["LLMs"]},
    )
    gen = GeneratedGraph(
        topics=["neural network", "LLMs"],
        dependencies=[("LLMs", "neural network")],
    )
    s = score_graph(ex, gen)
    assert s.topic_f1 == 1.0
    assert s.dependency_f1 == 1.0


def test_allowed_extra_is_not_hallucination():
    ex = _ex(allowed_extra_topics=["Tokenization"])
    gen = GeneratedGraph(topics=["A", "B", "C", "Tokenization"], dependencies=[("B", "A"), ("C", "B")])
    s = score_graph(ex, gen)
    assert s.hallucinated_topic_rate == 0.0
    assert s.topic_precision == 1.0


def test_token_containment_match():
    ex = _ex(
        gold_topics=["Semantic Analysis", "Code Generation", "Functions"],
        gold_dependencies=[("Code Generation", "Semantic Analysis")],
    )
    gen = GeneratedGraph(
        topics=["Introduction to Semantic Analysis", "Code Generation Techniques", "Functions and Modules"],
        dependencies=[("Code Generation Techniques", "Introduction to Semantic Analysis")],
    )
    s = score_graph(ex, gen)
    assert s.topic_recall == 1.0
    assert s.matched_topics == 3
    assert topic_similarity("Parsing", "Exploring Syntax Analysis and Parsing") >= 0.5


def test_short_acronym_does_not_containment_match():
    """'SQL' must not match 'SQL Injection' via 3-letter containment."""
    ex = _ex(
        gold_topics=["SQL", "SQL Injection"],
        gold_dependencies=[("SQL Injection", "SQL")],
    )
    gen = GeneratedGraph(topics=["SQL Injection"], dependencies=[])
    s = score_graph(ex, gen)
    assert s.matched_topics == 1
    assert s.topic_recall == 0.5


def test_incorrect_dependency_between_gold_topics():
    ex = _ex()
    gen = GeneratedGraph(topics=["A", "B", "C"], dependencies=[("A", "C"), ("B", "A"), ("C", "B")])
    s = score_graph(ex, gen)
    assert "INCORRECT_DEPENDENCY" in s.failures
    assert s.dependency_precision < 1.0


def test_reversed_dependency_is_direction_error_not_match():
    ex = _ex()
    gen = GeneratedGraph(topics=["A", "B", "C"], dependencies=[("A", "B"), ("C", "B")])
    s = score_graph(ex, gen)
    assert "WRONG_DEPENDENCY_DIRECTION" in s.failures
    assert s.dependency_direction_error_rate == 0.5
    assert s.reversed_dependencies == 1
    assert s.dependency_recall == 0.5
    # reverse is not also an extra edge
    assert s.extra_dependency_rate == 0.0


def test_optional_topic_counts_for_precision_not_recall():
    ex = _ex(optional_topics=["D"])
    gen = GeneratedGraph(topics=["A", "B", "C", "D"], dependencies=[("B", "A"), ("C", "B")])
    s = score_graph(ex, gen)
    assert s.topic_recall == 1.0
    assert s.topic_precision == 1.0
    assert s.hallucinated_topic_rate == 0.0


def test_acceptable_dependency_counts_for_precision_not_recall():
    ex = _ex(acceptable_dependencies=[("C", "A")])
    gen = GeneratedGraph(topics=["A", "B", "C"], dependencies=[("B", "A"), ("C", "A")])
    s = score_graph(ex, gen)
    assert s.dependency_recall == 0.5
    assert s.dependency_precision == 1.0
    assert "MISSING_PREREQUISITE" in s.failures


def test_curated_alias_match():
    ex = _ex(
        gold_topics=["Control Flow", "Functions"],
        gold_dependencies=[("Functions", "Control Flow")],
        topic_aliases={"Control Flow": ["Control Structures"]},
    )
    gen = GeneratedGraph(
        topics=["Control Structures", "Functions"],
        dependencies=[("Functions", "Control Structures")],
    )
    s = score_graph(ex, gen)
    assert s.topic_f1 == 1.0
    assert s.dependency_f1 == 1.0


def test_required_topics_subset_treats_other_gold_as_optional():
    ex = _ex(required_topics=["A", "B"])
    gen = GeneratedGraph(topics=["A", "B"], dependencies=[("B", "A")])
    s = score_graph(ex, gen)
    assert s.topic_recall == 1.0
    assert s.gold_topics == 2

