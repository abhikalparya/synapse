"""Isolate evaluation from the developer's live Synapse database and event log.

Must run before ``app.db.session`` is imported.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


def isolate_eval_runtime(*, force: bool = False, prefix: str = "synapse-eval-") -> Path:
    """Point DB + proposal events at a temp directory.

    ``force=True`` always uses a fresh temp DB (reliability/audit CLI) so a developer
    ``SYNAPSE_DB_PATH`` cannot be mutated. If ``app.db.session`` is already imported
    (pytest), the engine is already bound; conftest is responsible for isolation.
    """
    tmp = Path(tempfile.mkdtemp(prefix=prefix))
    if force or not (os.environ.get("SYNAPSE_DB_PATH") or "").strip():
        os.environ["SYNAPSE_DB_PATH"] = str(tmp / "eval.db")
    if force or not (os.environ.get("SYNAPSE_PROPOSAL_EVENTS_PATH") or "").strip():
        os.environ["SYNAPSE_PROPOSAL_EVENTS_PATH"] = str(tmp / "proposal_events.jsonl")
    os.environ.setdefault("SYNAPSE_LOG_LLM_USAGE", "0")
    return Path(os.environ["SYNAPSE_DB_PATH"])


def session_already_imported() -> bool:
    return "app.db.session" in sys.modules
