"""Adversarial reliability cases. Deterministic — no LLM required.

Each case exercises production validators (`build_topics_and_dependencies`,
`filter_reshape_new_dependencies`, `apply_proposal`, `restore_snapshot`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT = _REPO_ROOT / "data" / "eval" / "learning_graph_reliability_v1.jsonl"


@dataclass
class ReliabilityCase:
    id: str
    kind: str
    description: str
    raw: str | None = None
    reshape: dict[str, Any] | None = None
    notes: str = ""
    dataset_version: str = "learning_graph_reliability_v1"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "description": self.description,
            "raw": self.raw,
            "reshape": self.reshape,
            "notes": self.notes,
            "dataset_version": self.dataset_version,
            **self.extra,
        }


def _ingest_json(topics: list[str], deps: list[tuple[str, str]], *, extra: list[dict] | None = None) -> str:
    payload: dict[str, Any] = {
        "topics": [{"title": t, "summary": f"Summary of {t} covering the core idea.", "confidence": 0.9} for t in topics],
        "dependencies": [{"from": a, "to": b} for a, b in deps],
    }
    if extra:
        payload["dependencies"].extend(extra)
    return json.dumps(payload)


def reliability_v1() -> list[ReliabilityCase]:
    return [
        ReliabilityCase(
            id="cycle_abc_001",
            kind="cycle",
            description="A requires B, B requires C, generated C requires A (closes a cycle).",
            raw=_ingest_json(
                ["A", "B", "C"],
                [("A", "B"), ("B", "C"), ("C", "A")],
            ),
            notes="Direct retains the cycle edge; Synapse skips it via would_create_cycle.",
        ),
        ReliabilityCase(
            id="cycle_ab_001",
            kind="cycle",
            description="Mutual pair A requires B and B requires A.",
            raw=_ingest_json(["A", "B"], [("A", "B"), ("B", "A")]),
        ),
        ReliabilityCase(
            id="self_loop_001",
            kind="self_loop",
            description="Generated A requires A.",
            raw=_ingest_json(["A", "B"], [("B", "A"), ("A", "A")]),
        ),
        ReliabilityCase(
            id="unknown_ref_001",
            kind="unknown_reference",
            description="Dependency names a topic that does not exist.",
            raw=_ingest_json(["A", "B"], [("B", "A")], extra=[{"from": "B", "to": "Ghost Topic"}]),
        ),
        ReliabilityCase(
            id="unknown_ref_both_001",
            kind="unknown_reference",
            description="Both endpoints are unknown titles.",
            raw=_ingest_json(["A"], [], extra=[{"from": "MissingFrom", "to": "MissingTo"}]),
        ),
        ReliabilityCase(
            id="reshape_oos_001",
            kind="out_of_scope_reshape",
            description="Reshape of {A,B} tries to reference outside topic Z.",
            reshape={
                "selected_titles": ["A", "B"],
                "existing_internal_edges": [["B", "A"]],
                "outside_titles": ["Z"],
                "new_topics": [{"title": "A-split", "summary": "A narrower piece of A.", "confidence": 0.9}],
                "new_dependencies": [
                    {"from": "B", "to": "A"},
                    {"from": "B", "to": "Z"},
                ],
            },
            notes="Production reshape title map never includes outside titles.",
        ),
        ReliabilityCase(
            id="malformed_json_001",
            kind="malformed_json",
            description="LLM output is not valid JSON.",
            raw="{not json",
        ),
        ReliabilityCase(
            id="malformed_missing_topics_001",
            kind="missing_fields",
            description="JSON object with no topics list.",
            raw=json.dumps({"dependencies": [{"from": "A", "to": "B"}]}),
        ),
        ReliabilityCase(
            id="malformed_empty_topics_001",
            kind="missing_fields",
            description="Empty topics list.",
            raw=json.dumps({"topics": [], "dependencies": []}),
        ),
        ReliabilityCase(
            id="malformed_dep_types_001",
            kind="malformed_types",
            description="Dependency endpoints are non-strings; topics are well-formed.",
            raw=json.dumps(
                {
                    "topics": [
                        {"title": "A", "summary": "Topic A summary text here.", "confidence": 0.9},
                        {"title": "B", "summary": "Topic B summary text here.", "confidence": 0.9},
                    ],
                    "dependencies": [{"from": 1, "to": 2}, "not-an-object"],
                },
            ),
        ),
        ReliabilityCase(
            id="transaction_merge_unknown_001",
            kind="transaction_failure",
            description="Apply creates a topic then merges an unknown source; the transaction must abort.",
            extra={"seed_topics": ["Keep"], "new_topic_title": "ShouldNotPersist"},
        ),
        ReliabilityCase(
            id="rollback_exact_001",
            kind="rollback",
            description="Apply a valid ingest proposal then restore the pre-apply snapshot.",
            extra={"seed_topics": ["Root"], "new_topic_title": "Child", "new_edge": ["Child", "Root"]},
        ),
    ]


def write_reliability_dataset(path: str | Path | None = None) -> Path:
    target = Path(path) if path else OUTPUT
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(c.to_json(), ensure_ascii=False) for c in reliability_v1()]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def main() -> None:
    path = write_reliability_dataset()
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
