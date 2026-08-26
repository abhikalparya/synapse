"""Tests for databases_v2 / data_engineering_v2 inventory iteration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.curriculum.inventory import (
    find_redundant_transitive_edges,
    inventory_file_hash,
    inventory_health_report,
    load_domain_inventory,
    load_experiment_config,
    load_inventory,
    validate_inventory_dict,
)
from app.services.generation_strategy import resolve_generation_strategy

CURRICULUM = Path(__file__).resolve().parents[2] / "data" / "curriculum"

V1_HASHES = {
    "databases_v1.json": "28aadc98baf0229ee191c7f532a6b9d0ba8191c4a1add2a5b55249cce5e0e79f",
    "data_engineering_v1.json": "68ca9fcd723efb476dd51b502cf4a3e97e8711107b633706e255ffdc2e1c8533",
}

# Overview composition edges may be transitively reachable via stage/mechanism chains
# while still being independently direct prerequisites of the overview concept.
ALLOWED_TRANSITIVE_SHORTCUTS = {
    "databases": {
        ("db.distributed_db", "db.replication"),
        ("db.schema_design", "db.tables"),
    },
    "data_engineering": {
        ("de.etl", "de.extraction"),
        ("de.etl", "de.transformation"),
    },
}


@pytest.mark.parametrize("domain", ["databases", "data_engineering"])
def test_v2_schema_validity(domain: str):
    inv = load_inventory(CURRICULUM / f"{domain}_v2.json")
    health = inventory_health_report(inv)
    assert health["validation_status"] == "valid"
    assert health["cycle_count"] == 0
    assert inv.version == "v2"


@pytest.mark.parametrize("filename,expected", list(V1_HASHES.items()))
def test_v1_remains_unchanged(filename: str, expected: str):
    path = CURRICULUM / filename
    assert path.is_file()
    assert inventory_file_hash(path) == expected


@pytest.mark.parametrize("domain", ["databases", "data_engineering"])
def test_v2_version_metadata_and_changelog(domain: str):
    inv = load_inventory(CURRICULUM / f"{domain}_v2.json")
    assert inv.version == "v2"
    assert inv.raw.get("parent_version") == "v1"
    assert inv.review_status == "reviewed"
    changelog = inv.raw.get("change_log")
    assert isinstance(changelog, list) and len(changelog) >= 3
    for row in changelog:
        for key in ("change_type", "old", "new", "reason", "rationale"):
            assert key in row and str(row[key]).strip()


@pytest.mark.parametrize("domain", ["databases", "data_engineering"])
def test_v2_prerequisites_acyclic_and_known(domain: str):
    raw = json.loads((CURRICULUM / f"{domain}_v2.json").read_text(encoding="utf-8"))
    errs = validate_inventory_dict(raw)
    assert not any("cycle" in e for e in errs)
    assert not any("unknown prerequisite" in e for e in errs)
    assert not any("ambiguous alias" in e for e in errs)


@pytest.mark.parametrize("domain", ["databases", "data_engineering"])
def test_v2_directness_constraints(domain: str):
    inv = load_inventory(CURRICULUM / f"{domain}_v2.json")
    edges = [(c.id, p) for c in inv.concepts for p in c.prerequisite_ids]
    shortcuts = set(find_redundant_transitive_edges(edges))
    unexpected = shortcuts - ALLOWED_TRANSITIVE_SHORTCUTS[domain]
    assert not unexpected, f"unexpected transitive shortcuts: {unexpected}"


@pytest.mark.parametrize("domain", ["databases", "data_engineering"])
def test_v2_loads_independently_and_resolver_selects_v2(domain: str):
    inv = load_inventory(CURRICULUM / f"{domain}_v2.json")
    assert inv.domain == domain
    resolved = load_domain_inventory(domain)
    assert resolved.version == "v2"
    assert resolved.path.name == f"{domain}_v2.json"
    cfg = load_experiment_config()
    assert cfg["inventory_files"][domain] == f"{domain}_v2.json"


def test_baseline_unchanged():
    assert resolve_generation_strategy(None) == "baseline"
    assert resolve_generation_strategy("domain_curriculum_prior") == "domain_curriculum_prior"


def test_find_redundant_transitive_edges_detects_shortcut():
    edges = [("a", "b"), ("b", "c"), ("a", "c")]
    assert ("a", "c") in find_redundant_transitive_edges(edges)
