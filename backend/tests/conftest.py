"""Pytest bootstrap: isolate SQLite from the developer's synapse.db."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="synapse-eval-test-"))
os.environ["SYNAPSE_DB_PATH"] = str(_TMP / "synapse.db")
os.environ["SYNAPSE_PROPOSAL_EVENTS_PATH"] = str(_TMP / "proposal_events.jsonl")
os.environ.setdefault("SYNAPSE_LOG_LLM_USAGE", "0")
