"""Audit evaluation cases: known structural/semantic graph issues.

JSONL is the inspectable artifact; the runner imports ``audit_v1()``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT = _REPO_ROOT / "data" / "eval" / "graph_audit_eval_v1.jsonl"


def _t(tid: str, title: str, summary: str) -> dict[str, str]:
    return {"id": tid, "title": title, "summary": summary}


def audit_v1() -> list[dict[str, Any]]:
    long = "A reasonably long summary that is not thin and describes the topic clearly."
    return [
        {
            "id": "orphan_advanced_001",
            "mode": "structural",
            "graph": {
                "topics": [
                    _t("alg", "Algebra", long),
                    _t("calc", "Calculus", long),
                ],
                "dependencies": [],
            },
            "known_issues": [
                {"type": "orphaned_topic", "topic": "Algebra", "expected_finding": "no dependency edges"},
                {"type": "orphaned_topic", "topic": "Calculus", "expected_finding": "no dependency edges"},
            ],
            "repair_edges": [["Calculus", "Algebra"]],
            "reference_topics": ["Algebra", "Calculus"],
            "reference_dependencies": [["Calculus", "Algebra"]],
        },
        {
            "id": "duplicate_title_001",
            "mode": "structural",
            "graph": {
                "topics": [
                    _t("n1", "Neural Networks", long),
                    _t("n2", "Neural Networks", long),
                    _t("lin", "Linear Algebra", long),
                ],
                "dependencies": [
                    {"from_topic_id": "n1", "to_topic_id": "lin"},
                ],
            },
            "known_issues": [
                {"type": "duplicate_title", "topic": "Neural Networks", "expected_finding": "two topics share the title"},
                {"type": "orphaned_topic", "topic": "Neural Networks", "expected_finding": "n2 has no edges"},
            ],
        },
        {
            "id": "thin_summary_001",
            "mode": "structural",
            "graph": {
                "topics": [
                    _t("a", "Root", long),
                    _t("b", "Thin Child", "short"),
                ],
                "dependencies": [{"from_topic_id": "b", "to_topic_id": "a"}],
            },
            "known_issues": [
                {"type": "thin_topic", "topic": "Thin Child", "expected_finding": "very short summary"},
            ],
        },
        {
            "id": "missing_prereq_connected_001",
            "mode": "semantic",
            "graph": {
                "topics": [
                    _t("math", "Mathematics", long),
                    _t("la", "Linear Algebra", "Vectors and matrices used throughout ML, assuming algebra fluency."),
                    _t("nn", "Neural Networks", "Layered models that multiply weights; needs linear algebra."),
                ],
                "dependencies": [{"from_topic_id": "nn", "to_topic_id": "la"}],
            },
            "known_issues": [
                {
                    "type": "missing_prerequisite",
                    "topic": "Linear Algebra",
                    "expected_finding": "Linear Algebra should require Mathematics",
                },
            ],
            "repair_edges": [["Linear Algebra", "Mathematics"]],
            "reference_topics": ["Mathematics", "Linear Algebra", "Neural Networks"],
            "reference_dependencies": [["Linear Algebra", "Mathematics"], ["Neural Networks", "Linear Algebra"]],
        },
        {
            "id": "wrong_direction_001",
            "mode": "semantic",
            "graph": {
                "topics": [
                    _t("html", "HTML", long),
                    _t("css", "CSS", "Styling documents whose structure is defined in HTML."),
                ],
                "dependencies": [{"from_topic_id": "html", "to_topic_id": "css"}],
            },
            "known_issues": [
                {
                    "type": "cycle_risk",
                    "topic": "HTML",
                    "expected_finding": "HTML requiring CSS is a suspicious/reversed direction",
                },
            ],
            "reference_topics": ["HTML", "CSS"],
            "reference_dependencies": [["CSS", "HTML"]],
        },
        {
            "id": "isolated_advanced_001",
            "mode": "structural",
            "graph": {
                "topics": [
                    _t("var", "Variables", long),
                    _t("fn", "Functions", long),
                    _t("own", "Rust Ownership", long),
                ],
                "dependencies": [{"from_topic_id": "fn", "to_topic_id": "var"}],
            },
            "known_issues": [
                {"type": "orphaned_topic", "topic": "Rust Ownership", "expected_finding": "isolated advanced topic"},
            ],
            "repair_edges": [["Rust Ownership", "Functions"]],
            "reference_topics": ["Variables", "Functions", "Rust Ownership"],
            "reference_dependencies": [["Functions", "Variables"], ["Rust Ownership", "Functions"]],
        },
        {
            "id": "clean_graph_001",
            "mode": "structural",
            "graph": {
                "topics": [
                    _t("a", "Functions", long),
                    _t("b", "Limits", long),
                ],
                "dependencies": [{"from_topic_id": "b", "to_topic_id": "a"}],
            },
            "known_issues": [],
        },
    ]


def write_audit_dataset(path: str | Path | None = None) -> Path:
    target = Path(path) if path else OUTPUT
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(c, ensure_ascii=False) for c in audit_v1()]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def main() -> None:
    path = write_audit_dataset()
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
