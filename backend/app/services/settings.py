"""Singleton local-workspace settings (Phase 13): persona, memory, thinking level. Read
by services/llm.py on every LLM call, and by services/ask.py to decide whether prior
Q&A turns are included as context."""

from typing import Any

from app.db.models import SettingsRow
from app.db.session import SessionLocal

_SETTINGS_ID = "default"


def _settings_row_to_dict(row: SettingsRow) -> dict[str, Any]:
    return {
        "persona": row.persona,
        "memory_enabled": row.memory_enabled,
        "thinking_level": row.thinking_level,
    }


def load_settings() -> dict:
    """Returns the current settings, creating the default row on first access."""
    with SessionLocal() as session, session.begin():
        row = session.get(SettingsRow, _SETTINGS_ID)
        if row is None:
            row = SettingsRow(id=_SETTINGS_ID)
            session.add(row)
            session.flush()
        return _settings_row_to_dict(row)


def update_settings(**fields: Any) -> dict:
    """Patch only the given fields (e.g. update_settings(persona='...'))."""
    with SessionLocal() as session, session.begin():
        row = session.get(SettingsRow, _SETTINGS_ID)
        if row is None:
            row = SettingsRow(id=_SETTINGS_ID)
            session.add(row)
        for key, value in fields.items():
            setattr(row, key, value)
        session.flush()
        return _settings_row_to_dict(row)
