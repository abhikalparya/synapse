"""Resolve graph generation strategy.

Production / product API default: ``baseline``.

Supported product opt-ins (explicit only):
- ``domain_curriculum_prior`` (experimental)
- ``domain_prior_edge_classifier`` (experimental only; not recommended)

Closed experiments (evaluation / historical reproducibility only):
- ``concept_first`` / ``concept_first_pruned``
- ``baseline_coverage_recovery``

Those legacy strategies are NOT accepted by the product ingest path unless
``allow_legacy=True`` is passed (evaluation helpers only).
"""

from __future__ import annotations

import os
from typing import Literal

# Product / API surface
RuntimeGenerationStrategy = Literal[
    "baseline",
    "domain_curriculum_prior",
    "domain_prior_edge_classifier",
]

# Closed experiments — evaluation reproducibility only
LegacyEvaluationStrategy = Literal[
    "concept_first",
    "concept_first_pruned",
    "baseline_coverage_recovery",
]

GenerationStrategy = RuntimeGenerationStrategy | LegacyEvaluationStrategy

RUNTIME_STRATEGIES: tuple[RuntimeGenerationStrategy, ...] = (
    "baseline",
    "domain_curriculum_prior",
    "domain_prior_edge_classifier",
)

LEGACY_EVALUATION_STRATEGIES: tuple[LegacyEvaluationStrategy, ...] = (
    "concept_first",
    "concept_first_pruned",
    "baseline_coverage_recovery",
)

GENERATION_STRATEGIES: tuple[GenerationStrategy, ...] = (
    *RUNTIME_STRATEGIES,
    *LEGACY_EVALUATION_STRATEGIES,
)

_RUNTIME_ALIASES: dict[str, RuntimeGenerationStrategy] = {
    "baseline": "baseline",
    "domain_curriculum_prior": "domain_curriculum_prior",
    "curriculum_prior": "domain_curriculum_prior",
    "domain_prior": "domain_curriculum_prior",
    "curriculum": "domain_curriculum_prior",
    "domain_prior_edge_classifier": "domain_prior_edge_classifier",
    "edge_classifier": "domain_prior_edge_classifier",
    "constrained_dependency": "domain_prior_edge_classifier",
    "constrained_edge_classifier": "domain_prior_edge_classifier",
}

_LEGACY_ALIASES: dict[str, LegacyEvaluationStrategy] = {
    "concept_first": "concept_first",
    "conceptfirst": "concept_first",
    "concept_first_pruned": "concept_first_pruned",
    "conceptfirst_pruned": "concept_first_pruned",
    "concept_first_prune": "concept_first_pruned",
    "pruned": "concept_first_pruned",
    "baseline_coverage_recovery": "baseline_coverage_recovery",
    "coverage_recovery": "baseline_coverage_recovery",
    "baselinecoveragerecovery": "baseline_coverage_recovery",
    "coverage": "baseline_coverage_recovery",
}


def _normalize_key(raw: str) -> str:
    return raw.strip().casefold().replace("-", "_")


def resolve_generation_strategy(
    strategy: str | None = None,
    *,
    allow_legacy: bool = False,
) -> GenerationStrategy:
    """Resolve strategy from argument or ``SYNAPSE_GENERATION_STRATEGY`` (default: baseline).

    Product callers must leave ``allow_legacy=False`` (default). Evaluation code that
    still exercises closed experiments may pass ``allow_legacy=True``.
    """
    raw = (
        strategy
        if strategy is not None
        else os.environ.get("SYNAPSE_GENERATION_STRATEGY") or "baseline"
    )
    key = _normalize_key(str(raw))
    if key in _RUNTIME_ALIASES:
        return _RUNTIME_ALIASES[key]
    if key in _LEGACY_ALIASES:
        legacy = _LEGACY_ALIASES[key]
        if not allow_legacy:
            raise ValueError(
                f"Generation strategy {legacy!r} is closed / evaluation-only. "
                f"Product strategies: {list(RUNTIME_STRATEGIES)}. "
                "Use evaluation adapters for historical experiments."
            )
        return legacy
    raise ValueError(
        f"Unknown generation strategy {strategy!r}; "
        f"product strategies: {list(RUNTIME_STRATEGIES)}"
        + (f"; legacy evaluation: {list(LEGACY_EVALUATION_STRATEGIES)}" if allow_legacy else "")
    )


def resolve_runtime_generation_strategy(strategy: str | None = None) -> RuntimeGenerationStrategy:
    """Product ingest / API resolver — never returns legacy closed experiments."""
    resolved = resolve_generation_strategy(strategy, allow_legacy=False)
    return resolved  # type: ignore[return-type]


def resolve_evaluation_generation_strategy(strategy: str | None = None) -> GenerationStrategy:
    """Evaluation / historical reproducibility — allows closed experiment strategies."""
    return resolve_generation_strategy(strategy, allow_legacy=True)


def strategy_enables_inventory_pruning(strategy: GenerationStrategy | str) -> bool:
    return resolve_evaluation_generation_strategy(strategy) == "concept_first_pruned"


def strategy_enables_coverage_recovery(strategy: GenerationStrategy | str) -> bool:
    return resolve_evaluation_generation_strategy(strategy) == "baseline_coverage_recovery"


def strategy_enables_domain_curriculum_prior(strategy: GenerationStrategy | str) -> bool:
    return resolve_generation_strategy(strategy, allow_legacy=True) == "domain_curriculum_prior"


def strategy_enables_domain_prior_edge_classifier(strategy: GenerationStrategy | str) -> bool:
    return resolve_generation_strategy(strategy, allow_legacy=True) == "domain_prior_edge_classifier"


def strategy_uses_domain_inventory(strategy: GenerationStrategy | str) -> bool:
    resolved = resolve_generation_strategy(strategy, allow_legacy=True)
    return resolved in ("domain_curriculum_prior", "domain_prior_edge_classifier")
