"""Versioned provider pricing for estimated cost. Update ``pricing_v1.json`` when rates change."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PRICING_PATH = _REPO_ROOT / "data" / "eval" / "pricing_v1.json"


@lru_cache(maxsize=4)
def load_pricing(path: str | None = None) -> dict[str, Any]:
    target = Path(path) if path else DEFAULT_PRICING_PATH
    if not target.is_file():
        return {
            "version": "missing",
            "as_of": None,
            "source": None,
            "currency": "USD",
            "per_million_tokens": {},
        }
    data = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Pricing file must be a JSON object: {target}")
    return data


def _normalize_model_key(model: str) -> str:
    m = (model or "").strip().casefold()
    if "/" in m:
        m = m.rsplit("/", 1)[-1]
    return m


def lookup_rates(model: str, pricing: dict[str, Any] | None = None) -> tuple[float, float] | None:
    """Return (input_usd_per_million, output_usd_per_million) or None if unknown."""
    table = (pricing or load_pricing()).get("per_million_tokens") or {}
    if not isinstance(table, dict):
        return None
    key = _normalize_model_key(model)
    entry = table.get(key)
    if entry is None:
        # try without date suffix: gpt-4o-mini-2024-07-18 -> gpt-4o-mini
        for known in table:
            if key.startswith(known) or known.startswith(key):
                entry = table[known]
                break
    if not isinstance(entry, dict):
        return None
    try:
        return float(entry["input"]), float(entry["output"])
    except (KeyError, TypeError, ValueError):
        return None


def estimate_cost_usd(
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
    *,
    pricing_path: str | None = None,
) -> float | None:
    """Estimate USD cost from token counts. Returns None when the model has no priced entry
    or token counts are missing. Never invents a price for an unknown model."""
    if input_tokens is None or output_tokens is None:
        return None
    rates = lookup_rates(model, load_pricing(pricing_path) if pricing_path else load_pricing())
    if rates is None:
        return None
    in_rate, out_rate = rates
    return (input_tokens / 1_000_000.0) * in_rate + (output_tokens / 1_000_000.0) * out_rate


def pricing_metadata(pricing_path: str | None = None) -> dict[str, Any]:
    data = load_pricing(pricing_path)
    return {
        "version": data.get("version"),
        "as_of": data.get("as_of"),
        "source": data.get("source"),
        "currency": data.get("currency", "USD"),
        "note": "Costs are estimates from the versioned pricing table; provider invoices may differ.",
    }
