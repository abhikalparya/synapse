"""Logical-operation correlation for Synapse AI workflows.

An ``operation_id`` identifies one user-facing AI action (e.g. one ingest request).
It is created at the service entry boundary — never inside ``call_llm_detailed``.

What it correlates:
- LLM usage log lines (``llm_usage.jsonl``)
- ``Proposal.generation_meta`` (including ``llm_calls`` summaries)
- Proposal lifecycle events (create / apply / discard / rollback when known)

What it does NOT represent:
- A distributed trace span, HTTP request id, or graph-row provenance
- Authentication, actor identity, or MCP tool invocation
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

_operation_id: ContextVar[str | None] = ContextVar("synapse_operation_id", default=None)
_operation_llm_summaries: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "synapse_operation_llm_summaries",
    default=None,
)


def new_operation_id() -> str:
    return uuid.uuid4().hex


def get_operation_id() -> str | None:
    return _operation_id.get()


def append_llm_summary(summary: dict[str, Any]) -> None:
    summaries = _operation_llm_summaries.get()
    if summaries is not None:
        summaries.append(summary)


@contextmanager
def synapse_operation() -> Iterator[str]:
    """Begin a logical AI operation; yields the new ``operation_id``."""
    op_id = new_operation_id()
    summaries: list[dict[str, Any]] = []
    token_id = _operation_id.set(op_id)
    token_summaries = _operation_llm_summaries.set(summaries)
    try:
        yield op_id
    finally:
        _operation_id.reset(token_id)
        _operation_llm_summaries.reset(token_summaries)


def finalize_generation_meta(
    partial: dict[str, Any] | None = None,
    *,
    generation_strategy: str | None = None,
) -> dict[str, Any]:
    """Merge strategy-specific metadata with operation correlation fields."""
    meta = dict(partial or {})
    if generation_strategy and not meta.get("generation_strategy"):
        meta["generation_strategy"] = generation_strategy
    op_id = get_operation_id()
    if op_id:
        meta["operation_id"] = op_id
    summaries = _operation_llm_summaries.get()
    if summaries is not None:
        meta["llm_calls"] = list(summaries)
    elif "llm_calls" not in meta:
        meta["llm_calls"] = []
    return meta


def operation_id_from_meta(meta: dict[str, Any] | None) -> str | None:
    if not meta:
        return None
    op_id = meta.get("operation_id")
    return str(op_id) if op_id else None
