"""Conservative, deterministic identity for topic titles.

This is intentionally narrower than the evaluation-only concept normalization
pipeline. It does not infer aliases, singular/plural equivalence, or semantic
similarity.
"""

from __future__ import annotations

import unicodedata


def canonical_topic_title(title: str) -> str:
    """Return the stable identity key for a human-readable topic title."""
    normalized = unicodedata.normalize("NFKC", title or "").casefold().strip()
    separators = [" " if not character.isalnum() else character for character in normalized]
    return " ".join("".join(separators).split())
