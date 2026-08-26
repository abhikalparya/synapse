"""Shared Pydantic/dataclass shapes for the evaluation framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Difficulty = Literal["beginner", "intermediate", "advanced"]
SystemName = Literal[
    "linear_baseline",
    "direct_llm_graph",
    "synapse",
    "concept_first",
    "baseline_coverage_recovery",
    "domain_curriculum_prior",
    "domain_prior_edge_classifier",
]


@dataclass
class EvalExample:
    """One golden learning-goal case.

    ``gold_dependencies`` use Synapse edge direction: ``[from, to]`` means
    ``from`` requires ``to`` (``to`` is the prerequisite).
    """

    id: str
    category: str
    difficulty: Difficulty
    goal: str
    gold_topics: list[str]
    gold_dependencies: list[tuple[str, str]]
    input_notes: str | None = None
    notes: str = ""
    topic_aliases: dict[str, list[str]] = field(default_factory=dict)
    allowed_extra_topics: list[str] = field(default_factory=list)
    gold_topic_summaries: dict[str, str] = field(default_factory=dict)
    required_topics: list[str] | None = None
    optional_topics: list[str] = field(default_factory=list)
    required_dependencies: list[tuple[str, str]] | None = None
    acceptable_dependencies: list[tuple[str, str]] = field(default_factory=list)
    # Evaluation-only: reviewed but not counted as correct (visible disagreement).
    ambiguous_dependencies: list[tuple[str, str]] = field(default_factory=list)
    dataset_version: str = "learning_graph_eval_v1"

    def required_topic_list(self) -> list[str]:
        return list(self.required_topics) if self.required_topics is not None else list(self.gold_topics)

    def optional_topic_list(self) -> list[str]:
        extras = list(self.optional_topics) + list(self.allowed_extra_topics)
        if self.required_topics is not None:
            required_n = {t.casefold().strip() for t in self.required_topics}
            for t in self.gold_topics:
                if t.casefold().strip() not in required_n:
                    extras.append(t)
        seen: set[str] = set()
        out: list[str] = []
        for t in extras:
            if t and t not in seen:
                seen.add(t)
                out.append(t)
        return out

    def required_dependency_list(self) -> list[tuple[str, str]]:
        if self.required_dependencies is not None:
            return list(self.required_dependencies)
        return list(self.gold_dependencies)



@dataclass
class GeneratedGraph:
    """A system-produced prerequisite graph (title-keyed)."""

    topics: list[str]
    dependencies: list[tuple[str, str]]
    skipped_dependencies: list[dict[str, str]] = field(default_factory=list)
    topic_confidences: list[float] = field(default_factory=list)
    raw_response: str = ""
    parse_ok: bool = True
    error: str | None = None
    error_category: str | None = None
    generation_meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphValidity:
    is_valid: bool
    self_loops: int = 0
    cycles: int = 0
    invalid_references: int = 0
    details: list[str] = field(default_factory=list)


@dataclass
class GraphQualityScores:
    topic_precision: float
    topic_recall: float
    topic_f1: float
    dependency_precision: float
    dependency_recall: float
    dependency_f1: float
    graph_valid: bool
    cycle_attempt: bool
    missing_prerequisite_rate: float
    hallucinated_topic_rate: float
    duplicate_topics: int
    matched_topics: int
    generated_topics: int
    gold_topics: int
    matched_dependencies: int
    generated_dependencies: int
    gold_dependencies: int
    extra_dependency_rate: float = 0.0
    dependency_direction_error_rate: float = 0.0
    reversed_dependencies: int = 0
    redundant_transitive_edge_count: int = 0
    redundant_transitive_edge_rate: float = 0.0
    # Edge-ambiguity calibration metrics (evaluation-only; see score_graph docs).
    required_edge_precision: float = 0.0
    required_edge_recall: float = 0.0
    required_edge_f1: float = 0.0
    missing_required_edge_rate: float = 0.0
    acceptable_alternative_edge_count: int = 0
    acceptable_alternative_rate: float = 0.0
    invalid_extra_edge_count: int = 0
    invalid_extra_edge_rate: float = 0.0
    ambiguous_edge_count: int = 0
    ambiguous_edge_rate: float = 0.0
    failures: list[str] = field(default_factory=list)


@dataclass
class SystemExampleResult:
    example_id: str
    system: SystemName
    repetition: int
    scores: GraphQualityScores | None
    graph: GeneratedGraph
    total_latency_ms: float
    llm_latency_ms: float
    deterministic_latency_ms: float
    cost_usd: float | None
    cost_estimated: bool
    input_tokens: int | None
    output_tokens: int | None
    failures: list[str] = field(default_factory=list)


def example_to_dict(ex: EvalExample) -> dict[str, Any]:
    return {
        "id": ex.id,
        "category": ex.category,
        "difficulty": ex.difficulty,
        "goal": ex.goal,
        "input_notes": ex.input_notes,
        "gold_topics": list(ex.gold_topics),
        "gold_dependencies": [list(d) for d in ex.gold_dependencies],
        "required_topics": list(ex.required_topic_list()),
        "optional_topics": list(ex.optional_topic_list()),
        "required_dependencies": [list(d) for d in ex.required_dependency_list()],
        "acceptable_dependencies": [list(d) for d in ex.acceptable_dependencies],
        "ambiguous_dependencies": [list(d) for d in ex.ambiguous_dependencies],
        "notes": ex.notes,
        "topic_aliases": dict(ex.topic_aliases),
        "aliases": dict(ex.topic_aliases),
        "allowed_extra_topics": list(ex.allowed_extra_topics),
        "gold_topic_summaries": dict(ex.gold_topic_summaries),
        "dataset_version": ex.dataset_version,
    }
