#!/usr/bin/env python3
"""Repo-root helper: ``python scripts/run_eval.py --help``."""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND))

from app.evaluation.runner import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
