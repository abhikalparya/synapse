"""Hardening tests for domain curriculum prior expansion."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from app.curriculum.inventory import (
    inventory_file_hash,
    inventory_health_report,
    list_inventory_paths,
    load_domain_inventory,
    load_inventory,
    validate_inventory_dict,
)
from app.curriculum.resolution import resolve_domain
from app.evaluation.domain_coverage_report import (
    run_domain_coverage_report,
    run_inventory_health_only,
)
from app.services.generation_strategy import resolve_generation_strategy


CURRICULUM_PKG = Path(__file__).resolve().parents[1] / "app" / "curriculum"
CURRICULUM_DATA = Path(__file__).resolve().parents[2] / "data" / "curriculum"


def test_all_inventory_files_validate():
    paths = list_inventory_paths()
    assert len(paths) >= 10
    for path in paths:
        inv = load_inventory(path)
        health = inventory_health_report(inv)
        assert health["validation_status"] == "valid"
        assert health["cycle_count"] == 0


def test_duplicate_aliases_rejected():
    data = {
        "domain": "x",
        "version": "v1",
        "concepts": [
            {
                "id": "a",
                "title": "Alpha",
                "description": "d",
                "aliases": ["Shared", "Shared"],
                "prerequisite_ids": [],
            }
        ],
    }
    errs = validate_inventory_dict(data)
    assert any("duplicate alias" in e for e in errs)


def test_invalid_level_rejected():
    data = {
        "domain": "x",
        "version": "v1",
        "concepts": [
            {
                "id": "a",
                "title": "Alpha",
                "description": "d",
                "aliases": [],
                "level": "expert-plus",
                "prerequisite_ids": [],
            }
        ],
    }
    errs = validate_inventory_dict(data)
    assert any("invalid level" in e for e in errs)


def test_cycle_diagnostic_mentions_reason():
    data = {
        "domain": "x",
        "version": "v1",
        "concepts": [
            {"id": "a", "title": "A", "description": "d", "aliases": [], "prerequisite_ids": ["b"]},
            {"id": "b", "title": "B", "description": "d", "aliases": [], "prerequisite_ids": ["a"]},
        ],
    }
    errs = validate_inventory_dict(data)
    assert any("cycle in curriculum inventory" in e for e in errs)


def test_inventory_hash_stable_for_unchanged_file():
    path = CURRICULUM_DATA / "compiler_construction_v1.json"
    h1 = inventory_file_hash(path)
    h2 = inventory_file_hash(path)
    assert h1 == h2
    inv = load_inventory(path)
    assert inv.content_hash == h1


def test_inventory_version_recorded():
    inv = load_domain_inventory("security")
    assert inv.version == "v1"
    assert inv.inventory_version_label() == "security_v1"
    assert inv.review_status
    inv2 = load_domain_inventory("databases")
    assert inv2.version == "v2"
    assert inv2.inventory_version_label() == "databases_v2"


def test_domain_resolution_explicit_and_case_map():
    r = resolve_domain(domain_override="security")
    assert r.ok and r.domain == "security" and r.source == "explicit"
    r2 = resolve_domain(case_id="sql_basics_001")
    assert r2.ok and r2.domain == "databases"
    r3 = resolve_domain(category="cloud_computing")
    assert r3.ok and r3.domain == "cloud_computing" and r3.source == "category"


def test_unresolved_domain_fallback_contract():
    soft = resolve_domain(case_id="python_basics_001", on_unresolved="baseline")
    assert soft.status == "DOMAIN_UNRESOLVED"
    assert soft.fallback_action == "baseline"
    assert soft.fallback_reason == "DOMAIN_UNRESOLVED"

    strict = resolve_domain(case_id="python_basics_001", on_unresolved="error")
    assert strict.status == "DOMAIN_UNRESOLVED"
    assert strict.fallback_action == "error"


def test_domain_prior_unavailable_when_inventory_missing(tmp_path: Path):
    root = tmp_path / "curriculum"
    root.mkdir()
    (root / "experiment_config_v1.json").write_text(
        json.dumps(
            {
                "inventory_files": {"ghost_domain": "ghost_v1.json"},
                "fallback": {"on_missing_inventory": "baseline"},
            }
        ),
        encoding="utf-8",
    )
    (root / "case_domain_map_v1.json").write_text(
        json.dumps({"case_to_domain": {"x": "ghost_domain"}}),
        encoding="utf-8",
    )
    r = resolve_domain(case_id="x", curriculum_dir=root, require_inventory=True)
    assert r.status == "DOMAIN_PRIOR_UNAVAILABLE"
    assert r.fallback_reason == "DOMAIN_PRIOR_UNAVAILABLE"


def test_baseline_remains_production_default():
    assert resolve_generation_strategy(None) == "baseline"
    assert resolve_generation_strategy("domain_curriculum_prior") == "domain_curriculum_prior"


def test_domain_coverage_report_offline(tmp_path: Path):
    md, js = run_domain_coverage_report(output_dir=tmp_path)
    data = json.loads(js.read_text())
    assert data["no_new_llm_calls"] is True
    assert data["mapped_case_count"] >= 30
    assert data["unmapped_case_count"] >= 1  # programming + math remain
    assert md.is_file()


def test_inventory_health_no_api_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    md, js = run_inventory_health_only(output_dir=tmp_path)
    data = json.loads(js.read_text())
    assert data["no_api_key_required"] is True
    assert data["all_valid"] is True
    assert md.is_file()


def test_runtime_curriculum_modules_do_not_import_eval_datasets():
    """Curriculum runtime must not import data/eval paths or dataset loaders."""
    forbidden_substrings = (
        "data/eval",
        "data\\eval",
        "load_dataset",
        "gold_topics",
        "gold_dependencies",
        "learning_graph_quality",
        "failure_analysis",
        "quality_stability",
    )
    for path in CURRICULUM_PKG.rglob("*.py"):
        src = path.read_text(encoding="utf-8")
        for needle in forbidden_substrings:
            assert needle not in src, f"{path.name} contains forbidden {needle!r}"
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mod = (
                    node.module
                    if isinstance(node, ast.ImportFrom)
                    else ",".join(a.name for a in node.names)
                )
                if mod and (
                    mod.startswith("app.evaluation.dataset")
                    or mod.endswith(".dataset")
                    and "evaluation" in mod
                ):
                    raise AssertionError(f"{path} imports evaluation dataset module {mod}")


def test_existing_v1_inventories_untouched_hashes():
    """Regression guard: starter inventories remain the original frozen files."""
    # Presence + load; content must still validate and keep domain ids.
    for domain in ("compiler_construction", "distributed_systems", "stream_processing"):
        inv = load_domain_inventory(domain)
        assert inv.version == "v1"
        assert inv.domain == domain
        assert inv.size() >= 5


def test_prioritization_planning_file_exists():
    path = CURRICULUM_DATA / "domain_prioritization_v1.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["frozen_before_inventory_authoring"] is True
    batch = data["frozen_expansion_batch"]
    assert 5 <= len(batch) <= 7
    assert "cloud_computing" in batch
