"""Phase 10: in-session Q&A assistant, scoped to a single topic's own context. Distinct
from the Phase 5 closure/inline quizzes (which test recall) and the Phase 8 ingest/expand
/audit/reshape modes (which propose graph mutations) -- this is read/explain-only and
never writes to the topic/dependency graph. Each exchange is persisted as a ``qa_log``
Artifact (Phase 9) so it survives a page reload as part of that topic's study log.
"""

import logging

from app.models.artifact import ArtifactCreate
from app.prompts.ask import build_ask_prompt
from app.services.artifacts import create_artifact, list_artifacts_for_topic
from app.services.llm import call_llm
from app.services.topics import get_topic_by_id

logger = logging.getLogger(__name__)

_QA_LOG_TITLE_MAX = 120


async def answer_topic_question(topic_id: str, question: str) -> dict | None:
    """Returns {"answer": str, "artifact_id": str}, or None if no topic with that id exists."""
    topic = get_topic_by_id(topic_id)
    if topic is None:
        return None

    # Prior qa_log entries are excluded from context -- each question is answered fresh
    # from the topic's own material, not a growing conversation transcript.
    artifacts = [a for a in list_artifacts_for_topic(topic_id) if a["type"] != "qa_log"]

    prompt = build_ask_prompt(
        topic_title=topic["title"],
        topic_summary=topic["summary"],
        resources=topic["resources"],
        artifacts=artifacts,
        question=question,
    )
    answer = (await call_llm(prompt)).strip()

    log_title = question.strip()
    if len(log_title) > _QA_LOG_TITLE_MAX:
        log_title = log_title[: _QA_LOG_TITLE_MAX - 1].rstrip() + "…"

    artifact = create_artifact(
        topic_id,
        ArtifactCreate(type="qa_log", title=log_title, content=f"Q: {question.strip()}\n\nA: {answer}"),
    )
    logger.info("Answered question for topic %s (artifact=%s)", topic_id, artifact["id"] if artifact else None)

    return {"answer": answer, "artifact_id": artifact["id"] if artifact else ""}
