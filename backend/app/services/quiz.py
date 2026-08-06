"""Closure-quiz generation and grading for a Topic, gating ``status: complete``."""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.models.quiz import Quiz, QuizPublic, QuizQuestion, QuizQuestionPublic, QuizResult, QuizResultQuestion, QuizSubmission
from app.models.topic import Topic
from app.prompts.quiz import build_quiz_prompt
from app.services.file_handler import read_raw_note, resolve_raw_note_file
from app.services.llm import call_llm
from app.services.topics import TOPICS_DIR, get_topic_by_id, update_topic

logger = logging.getLogger(__name__)

QUIZZES_DIR = TOPICS_DIR / "_quizzes"


def _ensure_quizzes_dir() -> None:
    QUIZZES_DIR.mkdir(parents=True, exist_ok=True)


def _quiz_path(topic_id: str) -> Path:
    return QUIZZES_DIR / f"{Path(topic_id).name}.json"


def quiz_pass_threshold() -> float:
    raw = os.environ.get("QUIZ_PASS_THRESHOLD", "0.7").strip()
    try:
        t = float(raw)
    except ValueError:
        return 0.7
    return max(0.0, min(1.0, t))


def quiz_gate_completion_enabled() -> bool:
    raw = os.environ.get("QUIZ_GATE_COMPLETION", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def save_quiz(quiz: Quiz) -> None:
    _ensure_quizzes_dir()
    _quiz_path(quiz.topic_id).write_text(
        json.dumps(quiz.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_quiz(topic_id: str) -> Quiz | None:
    path = _quiz_path(topic_id)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return Quiz.model_validate(data)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        logger.warning("Failed to load quiz for topic %s: %s", topic_id, exc)
        return None


def _strip_json_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _parse_quiz_json(raw: str) -> dict[str, Any]:
    cleaned = _strip_json_fences(raw)
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("LLM output is not a JSON object")
    return data


def _collect_resource_texts(topic: Topic) -> list[str]:
    texts: list[str] = []
    for r in topic.resources:
        if r.type not in ("document", "note"):
            continue
        path = resolve_raw_note_file(r.source_ref)
        if path is None:
            continue
        try:
            texts.append(read_raw_note(path))
        except OSError:
            continue
    return texts


async def generate_quiz_for_topic(topic_id: str) -> QuizPublic:
    """
    Build a closure quiz from the topic's summary + any document/note resource text.
    Regenerating a quiz resets ``quiz_passed`` -- a prior pass doesn't carry over to new questions.
    """
    row = get_topic_by_id(topic_id)
    if row is None:
        raise LookupError(f"No topic with id {topic_id!r}")
    topic = Topic.model_validate({k: v for k, v in row.items() if k != "path"})

    resource_texts = _collect_resource_texts(topic)
    if not topic.summary.strip() and not resource_texts:
        raise ValueError("Topic has no summary or readable resources to quiz from")

    prompt = build_quiz_prompt(topic.title, topic.summary, resource_texts)
    raw = await call_llm(prompt)
    data = _parse_quiz_json(raw)

    raw_questions = data.get("questions")
    if not isinstance(raw_questions, list) or not raw_questions:
        raise ValueError("LLM response did not include a non-empty 'questions' list")

    questions: list[QuizQuestion] = []
    for q in raw_questions:
        if not isinstance(q, dict):
            continue
        question_text = str(q.get("question", "")).strip()
        choices_raw = q.get("choices")
        choices = [str(c).strip() for c in choices_raw] if isinstance(choices_raw, list) else []
        try:
            correct_index = int(q.get("correct_index", -1))
        except (TypeError, ValueError):
            correct_index = -1
        if not question_text or len(choices) < 2 or not (0 <= correct_index < len(choices)):
            continue
        questions.append(QuizQuestion(question=question_text, choices=choices, correct_index=correct_index))

    if not questions:
        raise ValueError("LLM did not return any well-formed questions")

    quiz = Quiz(topic_id=topic_id, questions=questions, created_at=datetime.now(timezone.utc))
    save_quiz(quiz)
    update_topic(topic_id, quiz_passed=False)

    logger.info("Generated quiz for topic %s: %s question(s)", topic_id, len(questions))
    return QuizPublic(
        topic_id=topic_id,
        questions=[QuizQuestionPublic(id=q.id, question=q.question, choices=q.choices) for q in questions],
    )


def submit_quiz(topic_id: str, submission: QuizSubmission) -> QuizResult:
    quiz = load_quiz(topic_id)
    if quiz is None:
        raise LookupError(f"No quiz has been generated for topic {topic_id!r} yet")

    results: list[QuizResultQuestion] = []
    correct_count = 0
    for q in quiz.questions:
        selected = submission.answers.get(q.id)
        is_correct = selected is not None and selected == q.correct_index
        if is_correct:
            correct_count += 1
        results.append(
            QuizResultQuestion(
                question_id=q.id,
                correct=is_correct,
                correct_index=q.correct_index,
                selected_index=selected,
            ),
        )

    total = len(quiz.questions)
    score = correct_count / total if total else 0.0
    passed = score >= quiz_pass_threshold()

    if passed:
        update_topic(topic_id, quiz_passed=True)

    logger.info("Quiz submitted for topic %s: score=%.2f passed=%s", topic_id, score, passed)
    return QuizResult(
        topic_id=topic_id,
        score=score,
        passed=passed,
        correct_count=correct_count,
        total=total,
        results=results,
    )
