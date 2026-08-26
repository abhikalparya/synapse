"""Matching-mode adaptation and curated alias registry (evaluation-only).

Modes
-----
strict
    Pre-fairness schema: all ``gold_topics`` / ``gold_dependencies`` required;
    no optional topics, acceptable edges, or dataset aliases. Fuzzy Jaccard
    matching still applies (same as early eval). Reproduces stricter recall.

fair
    Current quality schema as loaded (required/optional, dataset aliases,
    acceptable edges, Jaccard).

curated_alias
    Fair matching plus explicitly approved entries from
    ``data/eval/curated_aliases_v1.json``. Never auto-accepts fuzzy neighbors.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

from app.evaluation.schemas import EvalExample

MatchingMode = Literal["strict", "fair", "curated_alias"]
MATCHING_MODES: tuple[MatchingMode, ...] = ("strict", "fair", "curated_alias")

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CURATED_ALIASES_PATH = _REPO_ROOT / "data" / "eval" / "curated_aliases_v1.json"

MATCHING_VERSIONS = {
    "strict": "strict_v1",
    "fair": "fair_quality_v1",
    "curated_alias": "fair_quality_v1+curated_aliases_v1",
}


def resolve_matching_mode(raw: str | None) -> MatchingMode:
    key = (raw or "fair").strip().casefold().replace("-", "_")
    aliases = {
        "strict": "strict",
        "fair": "fair",
        "curated_alias": "curated_alias",
        "curated": "curated_alias",
        "curated_aliases": "curated_alias",
    }
    if key not in aliases:
        raise ValueError(f"Unknown matching mode {raw!r}; choose one of {list(MATCHING_MODES)}")
    return aliases[key]  # type: ignore[return-value]


def load_curated_aliases(path: str | Path | None = None) -> dict[str, Any]:
    target = Path(path) if path else DEFAULT_CURATED_ALIASES_PATH
    if not target.is_file():
        return {
            "version": "curated_aliases_v1",
            "matching_version": MATCHING_VERSIONS["curated_alias"],
            "entries": [],
        }
    data = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Curated aliases file must be a JSON object: {target}")
    entries = data.get("entries") or []
    if not isinstance(entries, list):
        raise ValueError(f"{target}: entries must be a list")
    return data


def approved_alias_map(registry: dict[str, Any] | None = None) -> dict[str, list[str]]:
    """canonical -> approved alias strings (evaluation merge only)."""
    reg = registry if registry is not None else load_curated_aliases()
    out: dict[str, list[str]] = {}
    for entry in reg.get("entries") or []:
        if not entry.get("approved"):
            continue
        canonical = str(entry.get("canonical") or "").strip()
        if not canonical:
            continue
        aliases = [str(a).strip() for a in (entry.get("aliases") or []) if str(a).strip()]
        if not aliases:
            continue
        out.setdefault(canonical, [])
        for a in aliases:
            if a not in out[canonical]:
                out[canonical].append(a)
    return out


def approved_alias_records(registry: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    reg = registry if registry is not None else load_curated_aliases()
    return [e for e in (reg.get("entries") or []) if e.get("approved")]


def merge_aliases(base: dict[str, list[str]], extra: dict[str, list[str]]) -> dict[str, list[str]]:
    merged = {k: list(v) for k, v in base.items()}
    for canon, aliases in extra.items():
        merged.setdefault(canon, [])
        for a in aliases:
            if a not in merged[canon]:
                merged[canon].append(a)
    return merged


def adapt_example_for_mode(
    example: EvalExample,
    mode: MatchingMode | str,
    *,
    curated_registry: dict[str, Any] | None = None,
) -> EvalExample:
    """Return a copy of ``example`` configured for the given matching mode."""
    resolved = resolve_matching_mode(mode if isinstance(mode, str) else mode)
    version = MATCHING_VERSIONS[resolved]

    if resolved == "strict":
        return replace(
            example,
            required_topics=None,
            optional_topics=[],
            allowed_extra_topics=[],
            required_dependencies=None,
            acceptable_dependencies=[],
            topic_aliases={},
            dataset_version=f"{example.dataset_version}+{version}",
        )

    if resolved == "fair":
        return replace(
            example,
            topic_aliases=deepcopy(example.topic_aliases),
            dataset_version=f"{example.dataset_version}+{version}",
        )

    # curated_alias = fair schema + approved curated aliases
    extra = approved_alias_map(curated_registry)
    return replace(
        example,
        topic_aliases=merge_aliases(deepcopy(example.topic_aliases), extra),
        dataset_version=f"{example.dataset_version}+{version}",
    )
