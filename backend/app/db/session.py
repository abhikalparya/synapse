"""SQLite engine/session setup. Importing this module ensures the schema exists --
every entrypoint (FastAPI app, mcp_server.py, the migration script, tests) goes through
services/topics.py etc., which import this module, so there's no separate "run init"
step to remember.
"""

import os
import logging
from pathlib import Path

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker

from app.db.models import Base
from app.services.topic_identity import canonical_topic_title

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = _PROJECT_ROOT / "data"


def _resolve_db_path() -> Path:
    override = (os.environ.get("SYNAPSE_DB_PATH") or "").strip()
    return Path(override) if override else DATA_DIR / "synapse.db"


DB_PATH = _resolve_db_path()

DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})


@event.listens_for(engine, "connect")
def _enable_foreign_keys(dbapi_connection, _connection_record) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

Base.metadata.create_all(engine)


def ensure_topic_identity_schema(target_engine=engine) -> None:
    """Add and backfill topic identity without failing on legacy collisions.

    A unique index is created only when all existing canonical values are
    unique. Existing collisions are reported and left untouched for the later
    reviewable cleanup phase.
    """
    with target_engine.begin() as connection:
        columns = {row[1] for row in connection.execute(text("PRAGMA table_info(topics)"))}
        if not columns:
            return
        if "canonical_title" not in columns:
            connection.execute(
                text("ALTER TABLE topics ADD COLUMN canonical_title VARCHAR(500)")
            )

        rows = connection.execute(text("SELECT id, title FROM topics")).all()
        for topic_id, title in rows:
            connection.execute(
                text("UPDATE topics SET canonical_title = :canonical_title WHERE id = :topic_id"),
                {
                    "topic_id": topic_id,
                    "canonical_title": canonical_topic_title(str(title or "")),
                },
            )

        collisions = connection.execute(
            text(
                """
                SELECT canonical_title, COUNT(*) AS count
                FROM topics
                WHERE canonical_title IS NOT NULL AND canonical_title <> ''
                GROUP BY canonical_title
                HAVING COUNT(*) > 1
                """
            )
        ).all()
        blank_titles = connection.execute(
            text(
                """
                SELECT COUNT(*)
                FROM topics
                WHERE canonical_title IS NULL OR canonical_title = ''
                """
            )
        ).scalar_one()

        if collisions or blank_titles:
            logger.warning(
                "Skipping unique topic identity index: %s collision group(s), %s blank identity row(s)",
                len(collisions),
                blank_titles,
            )
            return

        connection.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_topics_canonical_title "
                "ON topics (canonical_title)"
            )
        )


ensure_topic_identity_schema()


def ensure_proposal_generation_meta_column() -> None:
    """Best-effort ADD COLUMN for existing SQLite DBs created before generation_meta."""
    from sqlalchemy import text

    with engine.begin() as conn:
        cols = {row[1] for row in conn.execute(text("PRAGMA table_info(proposals)"))}
        if "generation_meta" not in cols:
            conn.execute(
                text("ALTER TABLE proposals ADD COLUMN generation_meta JSON DEFAULT '{}'")
            )
