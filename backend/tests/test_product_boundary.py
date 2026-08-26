"""Product boundary smoke tests (no LLM)."""

from __future__ import annotations

import pytest

from app.curriculum.inventory import inventory_file_hash, load_domain_inventory
from app.services.generation_strategy import (
    RUNTIME_STRATEGIES,
    resolve_evaluation_generation_strategy,
    resolve_generation_strategy,
    resolve_runtime_generation_strategy,
)


from pathlib import Path

CURRICULUM = Path(__file__).resolve().parents[2] / "data" / "curriculum"


def test_runtime_strategies_are_small_and_explicit():
    assert RUNTIME_STRATEGIES == (
        "baseline",
        "domain_curriculum_prior",
        "domain_prior_edge_classifier",
    )
    assert resolve_runtime_generation_strategy(None) == "baseline"


def test_closed_experiments_rejected_by_product_resolver():
    for legacy in ("concept_first", "concept_first_pruned", "baseline_coverage_recovery"):
        with pytest.raises(ValueError, match="evaluation-only"):
            resolve_generation_strategy(legacy)
        assert resolve_evaluation_generation_strategy(legacy) == legacy


def test_databases_and_data_engineering_resolve_to_v2():
    assert load_domain_inventory("databases").version == "v2"
    assert load_domain_inventory("data_engineering").version == "v2"
    assert load_domain_inventory("compiler_construction").version == "v1"


def test_frozen_v1_hashes_for_databases_and_de():
    assert (
        inventory_file_hash(CURRICULUM / "databases_v1.json")
        == "28aadc98baf0229ee191c7f532a6b9d0ba8191c4a1add2a5b55249cce5e0e79f"
    )
    assert (
        inventory_file_hash(CURRICULUM / "data_engineering_v1.json")
        == "68ca9fcd723efb476dd51b502cf4a3e97e8711107b633706e255ffdc2e1c8533"
    )
