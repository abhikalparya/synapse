"""SQLite engine/session setup. Importing this module ensures the schema exists --
every entrypoint (FastAPI app, mcp_server.py, the migration script, tests) goes through
services/topics.py etc., which import this module, so there's no separate "run init"
step to remember.
"""

import os
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app.db.models import Base

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


def ensure_proposal_generation_meta_column() -> None:
    """Best-effort ADD COLUMN for existing SQLite DBs created before generation_meta."""
    from sqlalchemy import text

    with engine.begin() as conn:
        cols = {row[1] for row in conn.execute(text("PRAGMA table_info(proposals)"))}
        if "generation_meta" not in cols:
            conn.execute(
                text("ALTER TABLE proposals ADD COLUMN generation_meta JSON DEFAULT '{}'")
            )
