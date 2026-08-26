from app.evaluation.failure_analysis import classify_llm_exception, format_failure_table, summarize_failures
from app.evaluation.metrics import score_graph
from app.evaluation.schemas import EvalExample, GeneratedGraph
from app.services.proposal_events import classify_skip_reason


def test_classify_skip_reasons():
    assert classify_skip_reason("would create a cycle with other proposed dependencies") == "CYCLE_ATTEMPT"
    assert classify_skip_reason("A topic cannot depend on itself") == "CYCLE_ATTEMPT"
    assert classify_skip_reason("unknown topic reference") == "INVALID_TOPIC_REFERENCE"
    assert classify_skip_reason("unknown or out-of-scope topic reference") == "OUT_OF_SCOPE_REFERENCE"


def test_classify_llm_exceptions():
    assert classify_llm_exception(TimeoutError("request timed out")) == "LLM_TIMEOUT"
    assert classify_llm_exception(ValueError("LLM output is not a JSON object")) == "LLM_PARSE_FAILURE"
    assert classify_llm_exception(RuntimeError("502 bad gateway")) == "LLM_PROVIDER_FAILURE"


def test_score_graph_assigns_known_failure_categories():
    ex = EvalExample(
        id="x",
        category="programming",
        difficulty="beginner",
        goal="g",
        gold_topics=["A", "B"],
        gold_dependencies=[("B", "A")],
    )
    gen = GeneratedGraph(
        topics=["A", "Z", "Z"],
        dependencies=[("A", "A"), ("Z", "A")],
    )
    s = score_graph(ex, gen)
    for needed in ("HALLUCINATED_TOPIC", "DUPLICATE_TOPIC", "SELF_LOOP", "MISSING_PREREQUISITE"):
        assert needed in s.failures


def test_failure_table_format():
    table = format_failure_table(summarize_failures([["MISSING_PREREQUISITE", "CYCLE_ATTEMPT"], ["MISSING_PREREQUISITE"]]))
    assert "Missing prerequisite" in table
    assert "12" not in table  # don't hardcode fake counts
    assert "  2" in table
    assert "Cycle attempt" in table
