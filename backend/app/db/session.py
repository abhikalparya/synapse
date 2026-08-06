"""SQLite engine/session setup. Importing this module ensures the schema exists --
every entrypoint (FastAPI app, mcp_server.py, the migration script, tests) goes through
services/topics.py etc., which import this module, so there's no separate "run init"
step to remember.
"""

from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app.db.models import Base

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = _PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "synapse.db"

DATA_DIR.mkdir(parents=True, exist_ok=True)

engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})


@event.listens_for(engine, "connect")
def _enable_foreign_keys(dbapi_connection, _connection_record) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

Base.metadata.create_all(engine)
