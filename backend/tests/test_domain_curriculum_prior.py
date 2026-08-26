"""Domain curriculum prior tests (no API / no LLM for unit tests)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.curriculum.inventory import (
    InventoryValidationError,
    load_domain_inventory,
    load_inventory,
    resolve_domain_for_case,
    validate_inventory_dict,
)
from app.curriculum.selection import parse_and_validate_selection
from app.evaluation.curriculum_inventory_check import run_curriculum_inventory_check
from app.services.generation_strategy import resolve_generation_strategy
from app.services.topics import would_create_cycle


CURRICULUM = Path(__file__).resolve().parents[2] / "data" / "curriculum"


def test_inventory_schema_validation_ok():
    inv = load_inventory(CURRICULUM / "compiler_construction_v1.json")
    assert inv.domain == "compiler_construction"
    assert inv.size() >= 5


def test_duplicate_concept_ids_rejected():
    data = {
        "domain": "x",
        "version": "v1",
        "concepts": [
            {"id": "a", "title": "A", "description": "d", "aliases": [], "prerequisite_ids": []},
            {"id": "a", "title": "B", "description": "d", "aliases": [], "prerequisite_ids": []},
        ],
    }
    errs = validate_inventory_dict(data)
    assert any("duplicate concept id" in e for e in errs)


def test_duplicate_titles_rejected():
    data = {
        "domain": "x",
        "version": "v1",
        "concepts": [
            {"id": "a", "title": "Same", "description": "d", "aliases": [], "prerequisite_ids": []},
            {"id": "b", "title": "Same", "description": "d", "aliases": [], "prerequisite_ids": []},
        ],
    }
    errs = validate_inventory_dict(data)
    assert any("duplicate" in e and "title" in e for e in errs)


def test_cyclic_inventory_prerequisites_rejected():
    data = {
        "domain": "x",
        "version": "v1",
        "concepts": [
            {"id": "a", "title": "A", "description": "d", "aliases": [], "prerequisite_ids": ["b"]},
            {"id": "b", "title": "B", "description": "d", "aliases": [], "prerequisite_ids": ["a"]},
        ],
    }
    errs = validate_inventory_dict(data)
    assert any("cycle" in e for e in errs)


def test_unknown_inventory_prerequisite_rejected():
    data = {
        "domain": "x",
        "version": "v1",
        "concepts": [
            {"id": "a", "title": "A", "description": "d", "aliases": [], "prerequisite_ids": ["missing"]},
        ],
    }
    errs = validate_inventory_dict(data)
    assert any("unknown prerequisite" in e for e in errs)


def test_domain_selection_and_unknown_domain():
    assert resolve_domain_for_case("compilers_001") == "compiler_construction"
    with pytest.raises(ValueError, match="DOMAIN_UNRESOLVED"):
        resolve_domain_for_case("python_basics_001")


def test_inventory_coverage_gate_offline(tmp_path: Path):
    md, js = run_curriculum_inventory_check(output_dir=tmp_path)
    data = json.loads(js.read_text())
    assert data["no_new_llm_calls"] is True
    assert md.is_file()
    assert len(data["domains"]) >= 10
    assert data["all_gates_pass"] is True
    for d in data["domains"]:
        assert d["gate"]["pass"] is True


def test_missing_required_concept_in_inventory_reported(tmp_path: Path):
    # Tiny inventory missing Parsing
    inv = {
        "domain": "compiler_construction",
        "version": "vtest",
        "concepts": [
            {
                "id": "compiler.lexing",
                "title": "Lexical Analysis",
                "description": "tokenize",
                "aliases": [],
                "prerequisite_ids": [],
            }
        ],
    }
    root = tmp_path / "curriculum"
    root.mkdir()
    (root / "compiler_construction_v1.json").write_text(json.dumps(inv), encoding="utf-8")
    (root / "distributed_systems_v1.json").write_text(
        (CURRICULUM / "distributed_systems_v1.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (root / "stream_processing_v1.json").write_text(
        (CURRICULUM / "stream_processing_v1.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (root / "case_domain_map_v1.json").write_text(
        (CURRICULUM / "case_domain_map_v1.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (root / "experiment_config_v1.json").write_text(
        (CURRICULUM / "experiment_config_v1.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    _, js = run_curriculum_inventory_check(
        curriculum_dir=root,
        output_dir=tmp_path / "out",
        domains=["compiler_construction"],
    )
    data = json.loads(js.read_text())
    assert data["all_gates_pass"] is False
    assert data["domains"][0]["gate"]["result"] == "DOMAIN_INVENTORY_INSUFFICIENT"


def test_valid_concept_selection_and_unknown_rejected():
    inv = load_domain_inventory("compiler_construction")
    raw = json.dumps(
        {
            "selected_concepts": [
                {
                    "concept_id": "compiler.parsing",
                    "kind": "REQUIRED",
                    "reason": "needed",
                    "confidence": 0.9,
                },
                {
                    "concept_id": "not.a.real.id",
                    "kind": "REQUIRED",
                    "reason": "x",
                    "confidence": 0.9,
                },
                {"title": "Invented Topic", "kind": "REQUIRED", "confidence": 0.9},
            ]
        }
    )
    sel = parse_and_validate_selection(raw, inv, max_required=8, max_selected=8)
    assert [s.concept_id for s in sel.selected] == ["compiler.parsing"]
    assert "not.a.real.id" in sel.rejected_unknown_ids
    assert "Invented Topic" in sel.rejected_arbitrary_titles


def test_selection_cap():
    inv = load_domain_inventory("compiler_construction")
    rows = [
        {"concept_id": c.id, "kind": "REQUIRED", "confidence": 0.5 + i * 0.01}
        for i, c in enumerate(inv.concepts)
    ]
    sel = parse_and_validate_selection(
        json.dumps({"selected_concepts": rows}),
        inv,
        max_required=3,
        max_selected=3,
    )
    assert len(sel.selected) == 3
    assert sel.truncated is True


def test_no_concept_invention_in_pipeline_parse():
    inv = load_domain_inventory("stream_processing")
    sel = parse_and_validate_selection(
        json.dumps(
            {
                "selected_concepts": [
                    {"concept_id": "stream.kafka", "kind": "REQUIRED", "confidence": 0.8}
                ]
            }
        ),
        inv,
        max_required=5,
        max_selected=5,
    )
    assert sel.selected[0].title == "Kafka"
    assert all(s.concept_id in inv.by_id() for s in sel.selected)


def test_dependency_constrained_to_selected_and_dag():
    from app.evaluation.persistent_failure_attribution import has_prerequisite_path

    # Selected-only dependency space: path checks use title edges
    remapped = [("Parsing", "Lexical Analysis"), ("Semantic Analysis", "Parsing")]
    assert has_prerequisite_path(remapped, "Semantic Analysis", "Lexical Analysis")
    assert not has_prerequisite_path(remapped, "Lexical Analysis", "Semantic Analysis")
    # Existing DB cycle helper signature remains usable
    assert would_create_cycle("a", "b", []) is False
    assert would_create_cycle("a", "b", [{"from_topic_id": "b", "to_topic_id": "a"}]) is True


def test_strategy_opt_in_default_baseline():
    assert resolve_generation_strategy(None) == "baseline"
    assert resolve_generation_strategy("domain_curriculum_prior") == "domain_curriculum_prior"
    assert resolve_generation_strategy("curriculum_prior") == "domain_curriculum_prior"


def test_ambiguous_alias_rejected():
    data = {
        "domain": "x",
        "version": "v1",
        "concepts": [
            {"id": "a", "title": "Alpha", "description": "d", "aliases": ["Shared"], "prerequisite_ids": []},
            {"id": "b", "title": "Beta", "description": "d", "aliases": ["Shared"], "prerequisite_ids": []},
        ],
    }
    errs = validate_inventory_dict(data)
    assert any("ambiguous alias" in e for e in errs)


def test_inventory_check_no_api_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    run_curriculum_inventory_check(output_dir=tmp_path)


def test_load_rejects_invalid_file(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text(
        json.dumps(
            {
                "domain": "x",
                "version": "v1",
                "concepts": [{"id": "a", "title": "A", "description": "", "prerequisite_ids": []}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(InventoryValidationError):
        load_inventory(p)
