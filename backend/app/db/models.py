"""SQLAlchemy ORM tables backing Topics, Dependencies, Resources, Proposals, and Quizzes.

Proposal contents (topics/dependencies/skipped_dependencies/errors) are stored as JSON
columns rather than normalized tables: a proposal is only ever read or written as a whole
object (never queried by sub-field), and its topics/dependencies are keyed by ephemeral
temp ids that only mean something within that one proposal -- normalizing them would add
tables with no query benefit.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, ForeignKey, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import JSON, TypeDecorator


class Base(DeclarativeBase):
    pass


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


class UTCDateTime(TypeDecorator):
    """SQLite has no native timezone-aware datetime type -- plain ``DateTime(timezone=True)``
    silently drops tzinfo on read. Every datetime this app writes is already UTC, so this
    strips tzinfo on the way in (SQLite-friendly) and re-attaches UTC on the way out."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, _dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc).replace(tzinfo=None)
        return value

    def process_result_value(self, value: datetime | None, _dialect) -> datetime | None:
        if value is None:
            return None
        return value.replace(tzinfo=timezone.utc)


class TopicRow(Base):
    __tablename__ = "topics"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    title: Mapped[str] = mapped_column(String(500), default="")
    summary: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(20), default="not_started")
    quiz_passed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=_now)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=_now)

    resources: Mapped[list["ResourceRow"]] = relationship(
        "ResourceRow",
        cascade="all, delete-orphan",
        lazy="selectin",
        order_by="ResourceRow.id",
    )


class ResourceRow(Base):
    __tablename__ = "resources"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    topic_id: Mapped[str] = mapped_column(String(32), ForeignKey("topics.id"), index=True)
    type: Mapped[str] = mapped_column(String(20))
    source_ref: Mapped[str] = mapped_column(Text, default="")
    title: Mapped[str] = mapped_column(Text, default="")


class DependencyRow(Base):
    """Directed prerequisite edge: ``from_topic`` requires ``to_topic``."""

    __tablename__ = "dependencies"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    from_topic_id: Mapped[str] = mapped_column(String(32), ForeignKey("topics.id"), index=True)
    to_topic_id: Mapped[str] = mapped_column(String(32), ForeignKey("topics.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=_now)


class ProposalRow(Base):
    __tablename__ = "proposals"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    source: Mapped[str] = mapped_column(Text, default="")
    topics: Mapped[list] = mapped_column(JSON, default=list)
    dependencies: Mapped[list] = mapped_column(JSON, default=list)
    skipped_dependencies: Mapped[list] = mapped_column(JSON, default=list)
    errors: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=_now)
    applied_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    snapshot_id: Mapped[str | None] = mapped_column(String(64), nullable=True)


class QuizRow(Base):
    """One row per topic -- regenerating a quiz overwrites the prior one."""

    __tablename__ = "quizzes"

    topic_id: Mapped[str] = mapped_column(String(32), ForeignKey("topics.id"), primary_key=True)
    questions: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=_now)
