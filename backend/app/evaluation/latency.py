"""Latency summary helpers (p50 / p95 / mean)."""

from __future__ import annotations

import math
from typing import Sequence


def percentile(sorted_values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile on an already-sorted non-empty sequence."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = max(1, min(len(sorted_values), math.ceil((p / 100.0) * len(sorted_values))))
    return float(sorted_values[rank - 1])


def summarize_latencies_ms(samples: Sequence[float]) -> dict[str, float | int]:
    values = sorted(float(x) for x in samples)
    if not values:
        return {"samples": 0, "p50_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0}
    mean = sum(values) / len(values)
    return {
        "samples": len(values),
        "p50_ms": round(percentile(values, 50), 3),
        "p95_ms": round(percentile(values, 95), 3),
        "mean_ms": round(mean, 3),
    }
