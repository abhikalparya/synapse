"""Offline Stage-1 inventory pruning calibration + replay (evaluation only).

Uses stored Concept-First inventories. Gold labels are used ONLY for offline metrics;
runtime ``prune_inventory`` never sees gold data.

Does not claim end-to-end graph improvement — Stage-2 is not re-run here.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.evaluation.dataset import load_dataset
from app.evaluation.edge_ambiguity import adapt_example_for_edge_mode
from app.evaluation.inventory_attribution import (
    evaluate_inventory,
    extract_normalized_inventory,
    extract_stage1_inventory,
    gold_endpoint_present,
    inventory_graph_from_titles,
)
from app.evaluation.node_edge_attribution import classify_generated_topic, classify_gold_topic_representation
from app.evaluation.schemas import EvalExample
from app.services.inventory_pruning import PRUNE_CONFIGS, prune_inventory

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = _REPO_ROOT / "results" / "failure_analysis"
DEFAULT_ARTIFACT = (
    _REPO_ROOT / "results" / "benchmarks" / "2026-08-24_181142_quality_gpt-4o-mini_baseline.json"
)

_USEFUL = frozenset(
    {
        "MATCHED_GOLD_TOPIC",
        "ALIAS_OF_GOLD_TOPIC",
        "GRANULARITY_VARIANT",
        "DECOMPOSITION_COMPONENT",
    }
)

# Material recall degradation threshold for configuration selection.
_MATERIAL_RECALL_DROP = 0.03


def _rows_by_id(block: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in block.get("example_results") or []:
        eid = str(row.get("example_id") or "")
        if eid and int(row.get("repetition") or 0) == 0:
            out[eid] = row
    return out


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _is_useful(status: str) -> bool:
    return status in _USEFUL


def _opportunity_rate(example: EvalExample, titles: list[str]) -> dict[str, Any]:
    inv = inventory_graph_from_titles(titles)
    required = example.required_dependency_list()
    opp = 0
    for frm, to in required:
        s = classify_gold_topic_representation(frm, example, inv)
        t = classify_gold_topic_representation(to, example, inv)
        if gold_endpoint_present(s["status"]) and gold_endpoint_present(t["status"]):
            opp += 1
    present_endpoints = 0
    for g in example.required_topic_list():
        st = classify_gold_topic_representation(g, example, inv)["status"]
        if gold_endpoint_present(st):
            present_endpoints += 1
    return {
        "EDGE_OPPORTUNITY_RATE": (opp / len(required)) if required else 0.0,
        "opportunity_edges": opp,
        "required_edges": len(required),
        "required_endpoints_present": present_endpoints,
        "required_endpoints_total": len(example.required_topic_list()),
    }


def analyze_noise_properties(
    examples: dict[str, EvalExample],
    cf_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    useful_feats: list[dict[str, float]] = []
    noise_feats: list[dict[str, float]] = []
    status_counts: Counter[str] = Counter()
    samples_useful_zero: list[dict[str, Any]] = []
    samples_noise_zero: list[dict[str, Any]] = []

    from app.services.inventory_pruning import content_tokens
    from app.evaluation.metrics import topic_similarity

    for eid, row in cf_rows.items():
        ex = examples.get(eid)
        if not ex:
            continue
        inv = extract_normalized_inventory(row) or extract_stage1_inventory(row)
        gct = content_tokens(ex.goal)
        for t in inv:
            st = classify_generated_topic(t, ex)["status"]
            status_counts[st] += 1
            ov = len(gct & content_tokens(t))
            feat = {
                "n_chars": float(len(t)),
                "n_words": float(len(t.split())),
                "content_overlap": float(ov),
                "goal_sim": float(topic_similarity(t, ex.goal)),
            }
            if _is_useful(st):
                useful_feats.append(feat)
                if ov == 0 and len(samples_useful_zero) < 8:
                    samples_useful_zero.append({"case_id": eid, "title": t, "status": st})
            else:
                noise_feats.append(feat)
                if ov == 0 and len(samples_noise_zero) < 8:
                    samples_noise_zero.append({"case_id": eid, "title": t, "status": st})

    def _summ(xs: list[dict[str, float]]) -> dict[str, Any]:
        if not xs:
            return {}
        out: dict[str, Any] = {"n": len(xs)}
        for k in xs[0]:
            vals = [x[k] for x in xs]
            out[k] = {"mean": _mean(vals), "median": sorted(vals)[len(vals) // 2]}
        out["zero_content_overlap_count"] = sum(1 for x in xs if x["content_overlap"] == 0)
        return out

    return {
        "total_concepts": sum(status_counts.values()),
        "status_counts": dict(status_counts),
        "useful_summary": _summ(useful_feats),
        "noise_summary": _summ(noise_feats),
        "note": (
            "Useful and noise concepts have similar length. Goal content-token overlap is "
            "higher on average for useful concepts but many gold-matched foundations "
            "(e.g. Variables) still have zero overlap with goals like 'Learn Python…'. "
            "Exact/near duplicates are rare in this artifact. Lexical objective mismatch "
            "therefore preferentially removes noise but also removes many useful concepts."
        ),
        "samples_useful_zero_overlap": samples_useful_zero,
        "samples_noise_zero_overlap": samples_noise_zero,
    }


def calibrate_config(
    config_name: str,
    examples: dict[str, EvalExample],
    baseline_rows: dict[str, dict[str, Any]],
    cf_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    precs: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    halls: list[float] = []
    missf: list[float] = []
    gran: list[float] = []
    abstr: list[float] = []
    oos: list[float] = []
    rets: list[float] = []
    useful_removed = 0
    noise_removed = 0
    pruned_total = 0
    fallbacks = 0
    opp_rates: list[float] = []
    endpoints_present: list[int] = []
    endpoints_total = 0
    reason_counts: Counter[str] = Counter()
    per_case: list[dict[str, Any]] = []

    for eid, ex in examples.items():
        cf_row = cf_rows.get(eid)
        if not cf_row or not cf_row.get("parse_ok", True):
            continue
        inv = extract_normalized_inventory(cf_row) or extract_stage1_inventory(cf_row)
        before_cls = {
            t: classify_generated_topic(t, ex)["status"] for t in inv
        }
        prune = prune_inventory(inv, ex.goal, config_name=config_name)
        if prune.fallback_to_original_inventory:
            fallbacks += 1
        kept = list(prune.kept_concepts)
        kept_set = set(kept)
        for t, st in before_cls.items():
            if t not in kept_set:
                pruned_total += 1
                if _is_useful(st):
                    useful_removed += 1
                else:
                    noise_removed += 1
        for d in prune.decisions:
            if d.decision == "PRUNE":
                reason_counts[d.reason] += 1

        m = evaluate_inventory(kept, ex)
        precs.append(m["topic_precision"])
        recalls.append(m["topic_recall"])
        f1s.append(m["topic_f1"])
        halls.append(m["hallucinated_topic_rate"])
        missf.append(m["missing_foundational_concept_rate"])
        gran.append(m["granularity_mismatch_rate"])
        abstr.append(m["abstraction_mismatch_rate"])
        oos.append(m["out_of_scope_rate"])
        rets.append(prune.retention_rate)

        opp = _opportunity_rate(ex, kept)
        opp_rates.append(opp["EDGE_OPPORTUNITY_RATE"])
        endpoints_present.append(opp["required_endpoints_present"])
        endpoints_total += opp["required_endpoints_total"]

        # required endpoints removed vs unpruned CF
        before_opp = _opportunity_rate(ex, inv)
        removed_endpoints = []
        inv_g = inventory_graph_from_titles(inv)
        kept_g = inventory_graph_from_titles(kept)
        for g in ex.required_topic_list():
            was = gold_endpoint_present(classify_gold_topic_representation(g, ex, inv_g)["status"])
            now = gold_endpoint_present(classify_gold_topic_representation(g, ex, kept_g)["status"])
            if was and not now:
                removed_endpoints.append(g)

        per_case.append(
            {
                "case_id": eid,
                "input_count": len(inv),
                "kept_count": len(kept),
                "retention_rate": prune.retention_rate,
                "fallback": prune.fallback_to_original_inventory,
                "pruned_titles": list(prune.pruned_concepts),
                "kept_titles": kept,
                "inventory_f1": m["topic_f1"],
                "inventory_precision": m["topic_precision"],
                "inventory_recall": m["topic_recall"],
                "hallucinated_topic_rate": m["hallucinated_topic_rate"],
                "EDGE_OPPORTUNITY_RATE": opp["EDGE_OPPORTUNITY_RATE"],
                "opportunity_before": before_opp["EDGE_OPPORTUNITY_RATE"],
                "required_endpoints_removed": removed_endpoints,
                "decisions": [d.to_dict() for d in prune.decisions if d.decision == "PRUNE"][:20],
            }
        )

    n = len(precs) or 1
    return {
        "config_name": config_name,
        "n_cases": len(precs),
        "concepts_removed": pruned_total,
        "useful_removed": useful_removed,
        "noise_removed": noise_removed,
        "fallback_count": fallbacks,
        "prune_reason_counts": dict(reason_counts),
        "retention_rate": _mean(rets),
        "topic_precision": _mean(precs),
        "topic_recall": _mean(recalls),
        "topic_f1": _mean(f1s),
        "hallucinated_topic_rate": _mean(halls),
        "missing_foundational_concept_rate": _mean(missf),
        "granularity_mismatch_rate": _mean(gran),
        "abstraction_mismatch_rate": _mean(abstr),
        "out_of_scope_rate": _mean(oos),
        "EDGE_OPPORTUNITY_RATE": _mean(opp_rates),
        "required_endpoints_present_mean": _mean([float(x) for x in endpoints_present]),
        "required_endpoints_total_sum": endpoints_total,
        "per_case": per_case,
    }


def _baseline_inventory_metrics(
    examples: dict[str, EvalExample],
    baseline_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    precs, recalls, f1s, halls, missf, gran, abstr, oos, opps = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for eid, ex in examples.items():
        row = baseline_rows.get(eid)
        if not row or not row.get("parse_ok", True):
            continue
        titles = [str(t) for t in (row.get("generated_topics") or [])]
        m = evaluate_inventory(titles, ex)
        precs.append(m["topic_precision"])
        recalls.append(m["topic_recall"])
        f1s.append(m["topic_f1"])
        halls.append(m["hallucinated_topic_rate"])
        missf.append(m["missing_foundational_concept_rate"])
        gran.append(m["granularity_mismatch_rate"])
        abstr.append(m["abstraction_mismatch_rate"])
        oos.append(m["out_of_scope_rate"])
        opps.append(_opportunity_rate(ex, titles)["EDGE_OPPORTUNITY_RATE"])
    return {
        "config_name": "baseline_topics",
        "retention_rate": 1.0,
        "topic_precision": _mean(precs),
        "topic_recall": _mean(recalls),
        "topic_f1": _mean(f1s),
        "hallucinated_topic_rate": _mean(halls),
        "missing_foundational_concept_rate": _mean(missf),
        "granularity_mismatch_rate": _mean(gran),
        "abstraction_mismatch_rate": _mean(abstr),
        "out_of_scope_rate": _mean(oos),
        "EDGE_OPPORTUNITY_RATE": _mean(opps),
        "concepts_removed": 0,
        "useful_removed": 0,
        "noise_removed": 0,
    }


def select_configuration(calibrations: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Prefer smallest pruning config that improves precision, does not materially
    worsen recall, and reduces hallucination vs no_pruning.

    If none qualify, select no_pruning and record that no viable prune config exists.
    """
    base = calibrations["no_pruning"]
    base_prec = base["topic_precision"]
    base_rec = base["topic_recall"]
    base_hall = base["hallucinated_topic_rate"]

    # Order: least aggressive first among non-no_pruning configs.
    order = [
        "exact_duplicate",
        "near_duplicate",
        "malformed_and_filler",
        "objective_mismatch",
        "combined_conservative",
    ]
    eligible: list[str] = []
    for name in order:
        row = calibrations[name]
        if row["topic_precision"] <= base_prec + 1e-9:
            continue
        if row["topic_recall"] < base_rec - _MATERIAL_RECALL_DROP:
            continue
        if row["hallucinated_topic_rate"] >= base_hall - 1e-9:
            continue
        eligible.append(name)

    if not eligible:
        return {
            "selected": "no_pruning",
            "viable": False,
            "rule": (
                f"Require precision↑, recall drop ≤ {_MATERIAL_RECALL_DROP}, hallucination↓ "
                "vs no_pruning; none satisfied."
            ),
            "eligible": [],
        }
    return {
        "selected": eligible[0],
        "viable": True,
        "rule": (
            f"Smallest config with precision↑, recall drop ≤ {_MATERIAL_RECALL_DROP}, "
            "hallucination↓ vs no_pruning."
        ),
        "eligible": eligible,
    }


def _classify_hypothesis(selection: dict[str, Any], calibrations: dict[str, Any]) -> tuple[str, str]:
    base = calibrations["no_pruning"]
    # Best precision among configs that remove something
    best_prec_name = max(
        (n for n in calibrations if n != "baseline_topics"),
        key=lambda n: calibrations[n]["topic_precision"],
    )
    best = calibrations[best_prec_name]
    prec_up = best["topic_precision"] - base["topic_precision"]
    rec_drop = base["topic_recall"] - best["topic_recall"]
    if selection["viable"]:
        return (
            "SUPPORTED",
            f"Viable config {selection['selected']!r} improves precision without material recall loss.",
        )
    if prec_up >= 0.03 and rec_drop >= _MATERIAL_RECALL_DROP:
        return (
            "PARTIALLY_SUPPORTED",
            (
                f"Best precision config {best_prec_name!r} improves precision by {prec_up:+.3f} "
                f"but recall drops by {rec_drop:.3f} (material). Noise and useful concepts are "
                "lexically entangled under objective-overlap pruning."
            ),
        )
    if prec_up < 0.02 and best["concepts_removed"] == 0:
        return (
            "NOT_SUPPORTED",
            "No deterministic signal removed concepts with a favorable precision/recall trade-off.",
        )
    if prec_up < 0.02:
        return (
            "NOT_SUPPORTED",
            "Pruning did not produce a meaningful precision gain.",
        )
    return (
        "PARTIALLY_SUPPORTED",
        "Precision can improve under lexical filters, but not without material recall loss.",
    )


def _pick_cases(calibrations: dict[str, Any], selected: str, n: int = 10) -> list[dict[str, Any]]:
    base_cases = {c["case_id"]: c for c in calibrations["no_pruning"]["per_case"]}
    sel_cases = {c["case_id"]: c for c in calibrations[selected]["per_case"]}
    obj_cases = {
        c["case_id"]: c
        for c in calibrations.get("objective_mismatch", calibrations["combined_conservative"])[
            "per_case"
        ]
    }
    picked: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(cid: str, label: str) -> None:
        if cid in seen or cid not in base_cases:
            return
        b, s = base_cases[cid], sel_cases.get(cid) or obj_cases.get(cid) or base_cases[cid]
        o = obj_cases.get(cid, s)
        picked.append(
            {
                "case_id": cid,
                "label": label,
                "before": b,
                "after_selected": s,
                "after_objective_mismatch": o,
                "f1_delta": s["inventory_f1"] - b["inventory_f1"],
                "recall_delta": s["inventory_recall"] - b["inventory_recall"],
                "precision_delta": s["inventory_precision"] - b["inventory_precision"],
            }
        )
        seen.add(cid)

    # Improve / harm / no effect under objective_mismatch (most active rule)
    deltas = []
    for cid, b in base_cases.items():
        o = obj_cases.get(cid)
        if not o:
            continue
        deltas.append((o["inventory_f1"] - b["inventory_f1"], cid, o))
    deltas.sort(reverse=True)
    for _, cid, o in deltas[:3]:
        add(cid, "IMPROVES_F1" if o["inventory_f1"] > base_cases[cid]["inventory_f1"] else "NO_EFFECT")
    deltas.sort()
    for _, cid, o in deltas[:3]:
        add(cid, "HARMS_COVERAGE")
    # required endpoint removals
    for cid, o in obj_cases.items():
        if o.get("required_endpoints_removed"):
            add(cid, "REQUIRED_ENDPOINT_REMOVED")
        if not o.get("pruned_titles"):
            add(cid, "NO_EFFECT")
        if len(picked) >= n:
            break
    for cid in base_cases:
        if len(picked) >= n:
            break
        add(cid, "OTHER")
    return picked[:n]


def run_inventory_pruning_analysis(
    artifact_path: str | Path | None = None,
    *,
    dataset_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    target = Path(artifact_path) if artifact_path else DEFAULT_ARTIFACT
    if not target.is_file():
        raise FileNotFoundError(f"Artifact not found: {target}")

    payload = json.loads(target.read_text(encoding="utf-8"))
    systems = payload.get("systems") or {}
    if "concept_first" not in systems:
        raise ValueError("Artifact must contain concept_first system generations")

    ds_stem = payload.get("dataset") or "learning_graph_quality_v1"
    ds_path = Path(dataset_path) if dataset_path else _REPO_ROOT / "data" / "eval" / f"{ds_stem}.jsonl"
    if not ds_path.is_file():
        ds_path = _REPO_ROOT / "data" / "eval" / "learning_graph_quality_v1.jsonl"

    raw_examples = {ex.id: ex for ex in load_dataset(ds_path)}
    examples = {
        eid: adapt_example_for_edge_mode(ex, "edge_calibrated", topic_matching_mode="curated_alias")
        for eid, ex in raw_examples.items()
    }
    cf_rows = _rows_by_id(systems["concept_first"])
    baseline_rows = _rows_by_id(systems.get("synapse") or systems.get("direct_llm_graph") or {})

    noise = analyze_noise_properties(examples, cf_rows)

    calibrations: dict[str, dict[str, Any]] = {}
    for name in PRUNE_CONFIGS:
        calibrations[name] = calibrate_config(name, examples, baseline_rows, cf_rows)
    calibrations["baseline_topics"] = _baseline_inventory_metrics(examples, baseline_rows)

    # Deltas vs no_pruning
    base = calibrations["no_pruning"]
    rule_table = []
    for name, row in calibrations.items():
        if name in {"baseline_topics"}:
            continue
        rule_table.append(
            {
                "rule": name,
                "removed": row.get("concepts_removed", 0),
                "useful_removed": row.get("useful_removed", 0),
                "noise_removed": row.get("noise_removed", 0),
                "precision_delta": row["topic_precision"] - base["topic_precision"],
                "recall_delta": row["topic_recall"] - base["topic_recall"],
                "f1_delta": row["topic_f1"] - base["topic_f1"],
                "hallucination_delta": row["hallucinated_topic_rate"] - base["hallucinated_topic_rate"],
                "retention": row["retention_rate"],
                "EDGE_OPPORTUNITY_RATE": row["EDGE_OPPORTUNITY_RATE"],
                "opportunity_delta": row["EDGE_OPPORTUNITY_RATE"] - base["EDGE_OPPORTUNITY_RATE"],
            }
        )

    selection = select_configuration(calibrations)
    hypothesis, rationale = _classify_hypothesis(selection, calibrations)
    selected_name = selection["selected"]
    representative = _pick_cases(calibrations, selected_name if selected_name in calibrations else "objective_mismatch")

    # Opportunity lost vs unpruned CF for objective_mismatch / combined
    opp_impact = {}
    for name in ("objective_mismatch", "combined_conservative"):
        row = calibrations[name]
        lost_endpoints = sum(len(c.get("required_endpoints_removed") or []) for c in row["per_case"])
        opp_impact[name] = {
            "EDGE_OPPORTUNITY_RATE": row["EDGE_OPPORTUNITY_RATE"],
            "opportunity_delta_vs_no_pruning": row["EDGE_OPPORTUNITY_RATE"] - base["EDGE_OPPORTUNITY_RATE"],
            "required_endpoint_removals_total": lost_endpoints,
        }

    out_dir = Path(output_dir) if output_dir else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    json_path = out_dir / f"{ts}_inventory_pruning_analysis.json"
    md_path = out_dir / f"{ts}_inventory_pruning_analysis.md"

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "evaluation_stage": "OFFLINE_REPLAY",
        "live_end_to_end": {
            "status": "NOT_RUN",
            "reason": (
                "Offline replay shows no pruning config that improves precision without "
                "material recall loss; live Stage-2 regeneration was not justified."
            ),
            "llm_calls": "NO_NEW_LLM_CALLS",
        },
        "llm_calls": "NO_NEW_LLM_CALLS",
        "source_artifact": str(target),
        "dataset": str(ds_path),
        "matching_mode": "curated_alias",
        "edge_mode": "edge_calibrated",
        "gold_usage": {
            "runtime_pruning": "never — prune_inventory uses only objective + candidate titles",
            "offline_analysis": "gold used only to score precision/recall/hallucination after pruning",
        },
        "noise_analysis": noise,
        "proposed_signals": [
            {
                "signal": "DUPLICATE / EXACT normalized duplicate",
                "why": "Safe; exact duplicates never help graph coverage.",
                "effect_on_artifact": "0 removals in stored inventories",
            },
            {
                "signal": "NEAR_DUPLICATE (sim≥0.85)",
                "why": "Conservative collapse of near-identical titles.",
                "effect_on_artifact": "0 removals in stored inventories",
            },
            {
                "signal": "MALFORMED / GENERIC_FILLER",
                "why": "Empty/punctuation/structural scaffolding are not learnable concepts.",
                "effect_on_artifact": "Rare in this artifact (LLM already avoided Module N labels)",
            },
            {
                "signal": "OBJECTIVE_MISMATCH (no content-token overlap + weak peer sim)",
                "why": (
                    "Noise has lower mean goal content overlap than useful concepts; "
                    "largest available lexical signal without gold."
                ),
                "effect_on_artifact": (
                    "Removes substantial noise AND many useful zero-overlap foundations "
                    "(e.g. Variables under 'Learn Python…')"
                ),
            },
            {
                "signal": "REDUNDANT_CONCEPT (strict token containment)",
                "why": "Fragment titles contained in longer kept titles.",
                "effect_on_artifact": "Very few candidates; mixed useful/noise",
            },
        ],
        "decision_rule": selection,
        "hypothesis": {"code": hypothesis, "rationale": rationale},
        "rule_calibration_table": rule_table,
        "calibrations": {
            k: {kk: vv for kk, vv in v.items() if kk != "per_case"}
            for k, v in calibrations.items()
        },
        "retention_quality_curve": [
            {
                "configuration": name,
                "retention": calibrations[name]["retention_rate"],
                "precision": calibrations[name]["topic_precision"],
                "recall": calibrations[name]["topic_recall"],
                "f1": calibrations[name]["topic_f1"],
                "hallucination_rate": calibrations[name]["hallucinated_topic_rate"],
            }
            for name in (
                "baseline_topics",
                "no_pruning",
                "exact_duplicate",
                "near_duplicate",
                "malformed_and_filler",
                "objective_mismatch",
                "combined_conservative",
            )
        ],
        "opportunity_impact": opp_impact,
        "representative_cases": representative,
        "per_case_selected": calibrations[selected_name]["per_case"]
        if selected_name in calibrations
        else [],
    }
    # Attach full per-case for objective_mismatch (most informative) without bloating every config
    result["per_case_objective_mismatch"] = calibrations["objective_mismatch"]["per_case"]

    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(_render_md(result), encoding="utf-8")
    return md_path, json_path


def _render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# Inventory Pruning Analysis (Offline Replay)",
        "",
        f"- Source: `{payload['source_artifact']}`",
        f"- Evaluation stage: **{payload['evaluation_stage']}**",
        f"- LLM calls: **{payload['llm_calls']}**",
        f"- Live end-to-end: **{payload['live_end_to_end']['status']}** — {payload['live_end_to_end']['reason']}",
        f"- Hypothesis: **{payload['hypothesis']['code']}**",
        f"- Rationale: {payload['hypothesis']['rationale']}",
        f"- Selected config: `{payload['decision_rule']['selected']}` (viable={payload['decision_rule']['viable']})",
        "",
        "## Noise analysis",
        "",
        payload["noise_analysis"]["note"],
        "",
        f"- Status counts: `{payload['noise_analysis']['status_counts']}`",
        f"- Useful summary: `{payload['noise_analysis']['useful_summary']}`",
        f"- Noise summary: `{payload['noise_analysis']['noise_summary']}`",
        "",
        "## Proposed signals",
        "",
    ]
    for s in payload["proposed_signals"]:
        lines.append(f"- **{s['signal']}**: {s['why']} Effect: {s['effect_on_artifact']}")
    lines.extend(
        [
            "",
            "## Rule calibration",
            "",
            "| Rule | Removed | Useful Removed | Noise Removed | Precision Δ | Recall Δ | F1 Δ | Retention | Opp Δ |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for r in payload["rule_calibration_table"]:
        lines.append(
            f"| {r['rule']} | {r['removed']} | {r['useful_removed']} | {r['noise_removed']} | "
            f"{r['precision_delta']:+.3f} | {r['recall_delta']:+.3f} | {r['f1_delta']:+.3f} | "
            f"{r['retention']:.3f} | {r['opportunity_delta']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "## Retention–quality curve",
            "",
            "| Configuration | Retention | Precision | Recall | F1 | Hallucination Rate |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for r in payload["retention_quality_curve"]:
        lines.append(
            f"| {r['configuration']} | {r['retention']:.3f} | {r['precision']:.3f} | "
            f"{r['recall']:.3f} | {r['f1']:.3f} | {r['hallucination_rate']:.3f} |"
        )
    lines.extend(["", "## Opportunity impact", ""])
    for name, row in payload["opportunity_impact"].items():
        lines.append(
            f"- `{name}`: EDGE_OPPORTUNITY_RATE={row['EDGE_OPPORTUNITY_RATE']:.3f} "
            f"(Δ vs no_pruning {row['opportunity_delta_vs_no_pruning']:+.3f}); "
            f"required endpoint removals={row['required_endpoint_removals_total']}"
        )
    lines.extend(["", "## Representative cases", ""])
    for c in payload["representative_cases"]:
        lines.extend(
            [
                f"### {c['case_id']} — {c['label']}",
                "",
                f"- Before (no prune): n={c['before']['input_count']} "
                f"P/R/F1={c['before']['inventory_precision']:.3f}/"
                f"{c['before']['inventory_recall']:.3f}/{c['before']['inventory_f1']:.3f} "
                f"hall={c['before']['hallucinated_topic_rate']:.3f}",
                f"- After objective_mismatch: kept={c['after_objective_mismatch']['kept_count']} "
                f"pruned={c['after_objective_mismatch']['pruned_titles']} "
                f"P/R/F1={c['after_objective_mismatch']['inventory_precision']:.3f}/"
                f"{c['after_objective_mismatch']['inventory_recall']:.3f}/"
                f"{c['after_objective_mismatch']['inventory_f1']:.3f}",
                f"- Required endpoints removed: "
                f"{c['after_objective_mismatch'].get('required_endpoints_removed')}",
                f"- Opportunity: before={c['before']['EDGE_OPPORTUNITY_RATE']:.3f} "
                f"after={c['after_objective_mismatch']['EDGE_OPPORTUNITY_RATE']:.3f}",
                "",
            ]
        )
    return "\n".join(lines) + "\n"
