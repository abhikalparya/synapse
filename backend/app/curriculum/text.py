"""Lightweight text normalization for curriculum inventories (no eval dataset imports).

Mirrors ``app.evaluation.metrics.normalize_topic`` so inventory title/alias keys stay
compatible with evaluation coverage checks, without importing evaluation datasets.
"""

from __future__ import annotations

import re
import unicodedata

_PUNCT_RE = re.compile(r"[^\w\s]+", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")
_ARTICLE_RE = re.compile(r"^(the|a|an)\s+")


def _light_stem(token: str) -> str:
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith(("sses", "xes", "zes")):
        return token[:-2]
    if token.endswith("es") and len(token) > 4 and not token.endswith(("ses", "nes")):
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")) and len(token) > 4:
        return token[:-1]
    return token


def normalize_topic(title: str) -> str:
    s = unicodedata.normalize("NFKD", title or "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.casefold().strip()
    s = _PUNCT_RE.sub(" ", s)
    s = _SPACE_RE.sub(" ", s).strip()
    s = _ARTICLE_RE.sub("", s)
    tokens = [_light_stem(tok) for tok in s.split() if tok]
    return " ".join(tokens)
