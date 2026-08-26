from app.evaluation.inspect import classify_comparison
from app.evaluation.metrics import compare_graphs
from app.evaluation.schemas import EvalExample, GeneratedGraph


def _ex(**kwargs) -> EvalExample:
    defaults = dict(
        id="t",
        category="programming",
        difficulty="beginner",
        goal="g",
        gold_topics=["Linear Algebra", "Mathematics"],
        gold_dependencies=[("Linear Algebra", "Mathematics")],
    )
    defaults.update(kwargs)
    return EvalExample(**defaults)  # type: ignore[arg-type]


def test_classify_reversed_dependency():
    ex = _ex()
    gen = GeneratedGraph(
        topics=["Linear Algebra", "Mathematics"],
        dependencies=[("Mathematics", "Linear Algebra")],
    )
    cats = {f["category"] for f in classify_comparison(ex, compare_graphs(ex, gen))}
    assert "WRONG_DEPENDENCY_DIRECTION" in cats
    assert "MISSING_PREREQUISITE" not in cats
    assert "EXTRA_DEPENDENCY" not in cats


def test_classify_missing_and_extra_dependency():
    ex = _ex()
    gen = GeneratedGraph(
        topics=["Linear Algebra", "Mathematics"],
        dependencies=[("Linear Algebra", "Mathematics"), ("Mathematics", "Mathematics")],
    )
    # self-loop is extra (same endpoints but not the required reverse)
    comparison = compare_graphs(ex, gen)
    cats = {f["category"] for f in classify_comparison(ex, comparison)}
    assert "EXTRA_DEPENDENCY" in cats


def test_classify_alias_mismatch():
    ex = _ex(
        gold_topics=["Symmetric Crypto", "Hashing"],
        gold_dependencies=[],
    )
    gen = GeneratedGraph(topics=["Symmetric Encryption"], dependencies=[])
    comparison = compare_graphs(ex, gen)
    failures = classify_comparison(ex, comparison)
    cats = {f["category"] for f in failures}
    assert "ALIAS_MISMATCH" in cats or "TITLE_PARAPHRASE" in cats or "MISSING_TOPIC" in cats


def test_classify_granularity_or_paraphrase_for_intro_titles():
    ex = EvalExample(
        id="sql",
        category="databases",
        difficulty="beginner",
        goal="Learn to query relational data with SQL",
        gold_topics=["Joins"],
        gold_dependencies=[],
    )
    gen = GeneratedGraph(topics=["Introduction to SQL"], dependencies=[])
    failures = classify_comparison(ex, compare_graphs(ex, gen))
    extra = [f for f in failures if f["generated"] == "Introduction to SQL"]
    assert extra
    assert extra[0]["category"] in {"EXTRA_TOPIC", "HALLUCINATED_TOPIC", "TITLE_PARAPHRASE", "ALIAS_MISMATCH"}
