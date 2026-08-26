"""Inventory pruning tests (no API / no LLM)."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from app.evaluation.inventory_pruning_analysis import run_inventory_pruning_analysis
from app.evaluation.metrics import score_graph
from app.evaluation.schemas import EvalExample, GeneratedGraph
from app.services.generation_strategy import resolve_generation_strategy
from app.services.inventory_pruning import prune_inventory


def test_exact_normalized_duplicates_not_retained_twice():
    r = prune_inventory(["Git", "git", "GIT"], "Learn Git", config_name="exact_duplicate")
    assert r.kept_concept_count == 1
    assert r.kept_concepts[0] == "Git"
    assert r.pruned_concept_count == 2
    assert all(d.reason == "DUPLICATE" for d in r.decisions if d.decision == "PRUNE")


def test_near_duplicates_deterministic():
    r = prune_inventory(
        ["Incident Response Playbook", "Incident Response Playbooks Guide"],
        "Learn security operations",
        config_name="near_duplicate",
        near_duplicate_similarity=0.7,
    )
    assert len(r.kept_concepts) == 1
    assert r.kept_concepts[0] == "Incident Response Playbook"
    assert any(d.reason == "NEAR_DUPLICATE" for d in r.decisions if d.decision == "PRUNE")


def test_malformed_concepts_removed():
    r = prune_inventory(
        ["Variables", "!!!", "", "  ", "Functions"],
        "Learn Python programming",
        config_name="malformed_and_filler",
    )
    assert "Variables" in r.kept_concepts and "Functions" in r.kept_concepts
    assert any(d.reason == "MALFORMED" for d in r.decisions if d.decision == "PRUNE")


def test_generic_filler_only_for_defined_conditions():
    r = prune_inventory(
        ["Module 1", "Advanced Topics", "Variables", "Git"],
        "Learn Git",
        config_name="malformed_and_filler",
    )
    assert "Module 1" not in r.kept_concepts
    assert "Advanced Topics" not in r.kept_concepts
    assert "Variables" in r.kept_concepts
    assert "Git" in r.kept_concepts
    assert any(d.reason == "GENERIC_FILLER" for d in r.decisions)


def test_objective_mismatch_deterministic():
    r = prune_inventory(
        ["Git", "Branching", "Quantum Chromodynamics"],
        "Learn collaborative Git workflows",
        config_name="objective_mismatch",
    )
    # Git overlaps; Quantum Chromodynamics should prune if weakly related to peers
    assert "Git" in r.kept_concepts
    pruned = {d.original_title: d.reason for d in r.decisions if d.decision == "PRUNE"}
    assert "Quantum Chromodynamics" in pruned
    assert pruned["Quantum Chromodynamics"] == "OBJECTIVE_MISMATCH"


def test_short_useful_concept_not_removed_for_length_alone():
    r = prune_inventory(
        ["Git", "SSL"],
        "Learn Git and SSL",
        config_name="combined_conservative",
    )
    assert "Git" in r.kept_concepts
    assert "SSL" in r.kept_concepts


def test_empty_pruning_falls_back_safely():
    # All generic fillers → would empty → fallback
    r = prune_inventory(
        ["Module 1", "Lesson 2", "Advanced Topics"],
        "Learn anything unrelated zzz",
        config_name="combined_conservative",
    )
    assert r.fallback_to_original_inventory is True
    assert r.kept_concepts == ["Module 1", "Lesson 2", "Advanced Topics"]
    assert r.pruned_concept_count == 0


def test_every_pruned_concept_has_audit_reason():
    r = prune_inventory(
        ["Git", "git", "Module 1", "Underwater Basket Weaving"],
        "Learn Git",
        config_name="combined_conservative",
    )
    for d in r.decisions:
        if d.decision == "PRUNE":
            assert d.reason in {
                "DUPLICATE",
                "NEAR_DUPLICATE",
                "GENERIC_FILLER",
                "LOW_INFORMATION",
                "OBJECTIVE_MISMATCH",
                "REDUNDANT_CONCEPT",
                "MALFORMED",
                "UNKNOWN",
            }
            assert d.detail


def test_pruning_preserves_stable_ordering():
    r = prune_inventory(
        ["Zebra", "Apple", "Mango", "apple"],
        "Learn Apple Mango Zebra fruit taxonomy",
        config_name="exact_duplicate",
    )
    assert r.kept_concepts == ["Zebra", "Apple", "Mango"]


def test_offline_replay_requires_no_api_key(tmp_path: Path):
    ds = Path(__file__).resolve().parents[2] / "data" / "eval" / "learning_graph_quality_v1.jsonl"
    first = json.loads(ds.read_text(encoding="utf-8").splitlines()[0])
    eid = first["id"]
    artifact = {
        "dataset": "learning_graph_quality_v1",
        "systems": {
            "synapse": {
                "example_results": [
                    {
                        "example_id": eid,
                        "repetition": 0,
                        "parse_ok": True,
                        "generated_topics": list(first.get("gold_topics") or ["A"])[:3],
                        "generated_dependencies": [],
                    }
                ]
            },
            "concept_first": {
                "example_results": [
                    {
                        "example_id": eid,
                        "repetition": 0,
                        "parse_ok": True,
                        "generated_topics": list(first.get("gold_topics") or ["A"])[:2]
                        + ["Module 1", "Unrelated Quantum Foam"],
                        "generated_dependencies": [],
                        "generation_meta": {
                            "candidate_concepts": [
                                {"title": t}
                                for t in list(first.get("gold_topics") or ["A"])[:2]
                                + ["Module 1", "Unrelated Quantum Foam"]
                            ],
                            "normalized_inventory": list(first.get("gold_topics") or ["A"])[:2]
                            + ["Module 1", "Unrelated Quantum Foam"],
                        },
                    }
                ]
            },
        },
    }
    path = tmp_path / "q.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    out = tmp_path / "fa"
    out.mkdir()
    with patch("app.services.llm.call_llm_detailed") as llm:
        md, js = run_inventory_pruning_analysis(path, dataset_path=ds, output_dir=out)
        llm.assert_not_called()
    assert md.is_file() and js.is_file()
    payload = json.loads(js.read_text(encoding="utf-8"))
    assert payload["llm_calls"] == "NO_NEW_LLM_CALLS"
    assert payload["evaluation_stage"] == "OFFLINE_REPLAY"
    assert payload["live_end_to_end"]["status"] == "NOT_RUN"


def test_runtime_pruning_source_never_accesses_gold():
    src = (Path(__file__).resolve().parents[1] / "app" / "services" / "inventory_pruning.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(src)
    # No references to gold/eval datasets in the pruning service module body beyond metrics helpers
    forbidden = {"gold_topics", "gold_dependencies", "load_dataset", "EvalExample", "curated_alias"}
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert not (names & forbidden)


def test_concept_first_unchanged_when_pruning_disabled():
    r = prune_inventory(
        ["Variables", "Functions", "Ghost"],
        "Learn Python",
        config_name="no_pruning",
    )
    assert r.kept_concepts == ["Variables", "Functions", "Ghost"]
    assert r.pruned_concept_count == 0


def test_concept_first_pruned_is_explicit_opt_in():
    from app.services.generation_strategy import resolve_evaluation_generation_strategy

    assert resolve_generation_strategy(None) == "baseline"
    with pytest.raises(ValueError, match="evaluation-only"):
        resolve_generation_strategy("concept_first")
    assert resolve_evaluation_generation_strategy("concept_first") == "concept_first"
    assert resolve_evaluation_generation_strategy("concept_first_pruned") == "concept_first_pruned"
    assert resolve_generation_strategy("baseline") == "baseline"


def test_existing_evaluation_metrics_unchanged():
    ex = EvalExample(
        id="t",
        category="programming",
        difficulty="beginner",
        goal="Learn Python",
        gold_topics=["Variables", "Functions"],
        gold_dependencies=[("Functions", "Variables")],
    )
    graph = GeneratedGraph(
        topics=["Variables", "Functions"],
        dependencies=[("Functions", "Variables")],
    )
    before = score_graph(ex, graph)
    _ = prune_inventory(["Variables", "Functions"], "Learn Python", config_name="combined_conservative")
    after = score_graph(ex, graph)
    assert before.topic_f1 == after.topic_f1
    assert before.required_edge_f1 == after.required_edge_f1
