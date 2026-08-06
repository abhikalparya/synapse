import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class QuizQuestion(BaseModel):
    """Server-side record -- includes the correct answer, never sent to the client directly."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    question: str
    choices: list[str]
    correct_index: int = Field(ge=0)


class Quiz(BaseModel):
    topic_id: str
    questions: list[QuizQuestion]
    created_at: datetime | None = None


class QuizQuestionPublic(BaseModel):
    id: str
    question: str
    choices: list[str]


class QuizPublic(BaseModel):
    """What the client gets after generation -- no correct answers."""

    topic_id: str
    questions: list[QuizQuestionPublic]


class QuizSubmission(BaseModel):
    answers: dict[str, int] = Field(default_factory=dict, description="question id -> selected choice index")


class QuizResultQuestion(BaseModel):
    question_id: str
    correct: bool
    correct_index: int
    selected_index: int | None


class QuizResult(BaseModel):
    topic_id: str
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    correct_count: int
    total: int
    results: list[QuizResultQuestion]
