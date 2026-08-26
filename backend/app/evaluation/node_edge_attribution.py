"""Node vs relationship error attribution (diagnostic only; no score changes).

Baseline interpretation: curated_alias topic matching + edge_calibrated gold edges.
Does not modify Topic F1, Dependency F1, aliases, or acceptable-alternative rules.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.evaluation.dataset import load_dataset
from app.evaluation.edge_ambiguity import adapt_example_for_edge_mode
from app.evaluation.inspect import _graph_from_row
from app.evaluation.metrics import (
    _alias_index,
    compare_graphs,
    match_topic,
    normalize_topic,
    topic_similarity,
    topic_tokens,
)
from app.evaluation.schemas import EvalExample, GeneratedGraph
from app.evaluation.topic_equivalence import (
    _ABSTRACTION_UMBRELLAS,
    _DECOMPOSITION_PARTS,
    _strip_boilerplate,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_NODE_REP_PATH = _REPO_ROOT / "data" / "eval" / "node_representation_v1.json"

# --- Gold-topic representation statuses ---
GOLD_NODE_STATUSES = (
    "EXACT_MATCH",
    "ALIAS_MATCH",
    "GRANULARITY_VARIANT",
    "DECOMPOSED",
    "ABSTRACTED",
    "RELATED_BUT_DISTINCT",
    "MISSING",
    "UNKNOWN",
)

# --- Generated-topic statuses ---
GEN_NODE_STATUSES = (
    "MATCHED_GOLD_TOPIC",
    "ALIAS_OF_GOLD_TOPIC",
    "GRANULARITY_VARIANT",
    "DECOMPOSITION_COMPONENT",
    "ABSTRACTION_VARIANT",
    "RELATED_BUT_DISTINCT",
    "OUT_OF_SCOPE",
    "GENUINE_HALLUCINATION",
    "UNKNOWN",
)

MISSING_EDGE_ATTRS = (
    "EDGE_OMISSION",
    "SOURCE_ENDPOINT_MISSING",
    "TARGET_ENDPOINT_MISSING",
    "BOTH_ENDPOINTS_MISSING",
    "ENDPOINT_GRANULARITY_MISMATCH",
    "ENDPOINT_DECOMPOSITION",
    "ENDPOINT_ABSTRACTION_MISMATCH",
    "ENDPOINT_UNMATCHED",
    "UNKNOWN",
)

INVALID_EDGE_ATTRS = (
    "BOTH_ENDPOINTS_VALID_BUT_EDGE_INVALID",
    "SOURCE_ENDPOINT_OUT_OF_SCOPE",
    "TARGET_ENDPOINT_OUT_OF_SCOPE",
    "BOTH_ENDPOINTS_OUT_OF_SCOPE",
    "ENDPOINT_GRANULARITY_DRIFT",
    "ENDPOINT_DECOMPOSITION_DRIFT",
    "ENDPOINT_ABSTRACTION_DRIFT",
    "CURRICULUM_SCOPE_DRIFT",
    "UNKNOWN",
)

_PRESENT_GOLD = frozenset({"EXACT_MATCH", "ALIAS_MATCH"})
_GRAN_GOLD = frozenset({"GRANULARITY_VARIANT"})
_DECOMP_GOLD = frozenset({"DECOMPOSED"})
_ABS_GOLD = frozenset({"ABSTRACTED"})
_MISSING_GOLD = frozenset({"MISSING", "RELATED_BUT_DISTINCT", "UNKNOWN"})

_IN_SCOPE_GEN = frozenset(
    {"MATCHED_GOLD_TOPIC", "ALIAS_OF_GOLD_TOPIC", "GRANULARITY_VARIANT", "DECOMPOSITION_COMPONENT"},
)
_OPTIONALISH_GEN = frozenset({"ABSTRACTION_VARIANT", "RELATED_BUT_DISTINCT", "OUT_OF_SCOPE"})


def load_node_representation_map(path: str | Path | None = None) -> dict[str, Any]:
    target = Path(path) if path else DEFAULT_NODE_REP_PATH
    if not target.is_file():
        return {"version": "node_representation_v1", "granularity": [], "decomposition": [], "abstraction": []}
    data = json.loads(target.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {"granularity": [], "decomposition": [], "abstraction": []}


def _explicit_granularity_pair(gold: str, gen: str, rep: dict[str, Any]) -> bool:
    gn, tn = normalize_topic(gold), normalize_topic(gen)
    for row in rep.get("granularity") or []:
        if not row.get("approved"):
            continue
        if normalize_topic(str(row.get("canonical") or "")) == gn and normalize_topic(str(row.get("variant") or "")) == tn:
            return True
        if normalize_topic(str(row.get("canonical") or "")) == tn and normalize_topic(str(row.get("variant") or "")) == gn:
            return True
    return False


def _gens_mapping_to_gold(gold: str, example: EvalExample, graph: GeneratedGraph) -> list[str]:
    gnorm = normalize_topic(gold)
    out: list[str] = []
    for t in graph.topics:
        hit = match_topic(t, example)
        if hit is not None and normalize_topic(hit) == gnorm:
            out.append(t)
    return out


def classify_gold_topic_representation(
    gold: str,
    example: EvalExample,
    graph: GeneratedGraph,
    *,
    rep_map: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify how a gold topic appears in the generated graph (attribution only)."""
    rep = rep_map if rep_map is not None else load_node_representation_map()
    index = _alias_index(example)
    gnorm = normalize_topic(gold)
    mapped = _gens_mapping_to_gold(gold, example, graph)

    # 1–2 Exact / alias
    for t in mapped:
        tn = normalize_topic(t)
        if tn == gnorm:
            return {"status": "EXACT_MATCH", "candidates": [t], "reason": "Normalized titles are identical."}
        # alias index maps surface -> canonical; gold itself is also in index
        if tn in index and normalize_topic(index[tn]) == gnorm and tn != gnorm:
            # surface is alias, not the gold spelling
            gold_aliases = {normalize_topic(a) for a in example.topic_aliases.get(gold, [])}
            if tn in gold_aliases:
                return {"status": "ALIAS_MATCH", "candidates": [t], "reason": "Matched via curated/exact alias index."}

    # Fuzzy-mapped gens that are not exact/alias → granularity if containment or explicit map
    for t in mapped:
        if _explicit_granularity_pair(gold, t, rep):
            return {
                "status": "GRANULARITY_VARIANT",
                "candidates": [t],
                "reason": "Explicit reviewed granularity mapping.",
            }
        gt, tt = topic_tokens(gold), topic_tokens(t)
        if gt and tt and (gt < tt or tt < gt):
            return {
                "status": "GRANULARITY_VARIANT",
                "candidates": [t],
                "reason": "Endpoint matched only via fuzzy containment (granularity), not exact/alias identity.",
            }
        return {
            "status": "UNKNOWN",
            "candidates": [t],
            "reason": "Fuzzy topic match without clear exact/alias/granularity evidence.",
        }

    # 3 Decomposition: multiple (or one) listed parts present, gold not matched
    parts = _DECOMPOSITION_PARTS.get(gnorm)
    if parts:
        found = [t for t in graph.topics if normalize_topic(t) in parts]
        if len(found) >= 2:
            return {
                "status": "DECOMPOSED",
                "candidates": found,
                "reason": "Gold umbrella represented by multiple listed part-concepts.",
            }
        if len(found) == 1:
            return {
                "status": "GRANULARITY_VARIANT",
                "candidates": found,
                "reason": "Single listed decomposition part present without full umbrella match.",
            }

    # Explicit decomposition rows
    for row in rep.get("decomposition") or []:
        if not row.get("approved"):
            continue
        if normalize_topic(str(row.get("canonical") or "")) != gnorm:
            continue
        parts_list = [normalize_topic(str(p)) for p in (row.get("parts") or [])]
        found = [t for t in graph.topics if normalize_topic(t) in parts_list]
        if len(found) >= 2:
            return {"status": "DECOMPOSED", "candidates": found, "reason": "Explicit reviewed decomposition mapping."}

    # 4 Abstraction: generated umbrella present
    for t in graph.topics:
        tn = normalize_topic(t)
        stripped = _strip_boilerplate(tn)
        if tn in _ABSTRACTION_UMBRELLAS or stripped in _ABSTRACTION_UMBRELLAS:
            # only if gold is more specific
            if gnorm not in _ABSTRACTION_UMBRELLAS:
                return {
                    "status": "ABSTRACTED",
                    "candidates": [t],
                    "reason": "Broader umbrella topic present instead of specific gold concept.",
                }
        for row in rep.get("abstraction") or []:
            if not row.get("approved"):
                continue
            if normalize_topic(str(row.get("specific") or "")) == gnorm and normalize_topic(str(row.get("umbrella") or "")) == tn:
                return {"status": "ABSTRACTED", "candidates": [t], "reason": "Explicit reviewed abstraction mapping."}

    # 5 Related-but-distinct
    best_t, best_s = "", 0.0
    for t in graph.topics:
        s = topic_similarity(gold, t)
        if s > best_s:
            best_s, best_t = s, t
    if best_t and 0.15 <= best_s < 0.5 and (topic_tokens(gold) & topic_tokens(best_t)):
        return {
            "status": "RELATED_BUT_DISTINCT",
            "candidates": [best_t],
            "reason": "Shares vocabulary with a generated topic but is not an approved equivalent.",
        }

    if not graph.topics:
        return {"status": "MISSING", "candidates": [], "reason": "No generated topics."}

    if best_s <= 0.0:
        return {"status": "MISSING", "candidates": [], "reason": "No overlapping generated title under deterministic similarity."}

    return {
        "status": "UNKNOWN",
        "candidates": [best_t] if best_t else [],
        "reason": "Insufficient deterministic evidence for a safer gold-representation category.",
    }


def classify_generated_topic(
    title: str,
    example: EvalExample,
    *,
    rep_map: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify a generated topic relative to the gold/optional pool (attribution only)."""
    rep = rep_map if rep_map is not None else load_node_representation_map()
    index = _alias_index(example)
    tn = normalize_topic(title)
    hit = match_topic(title, example)

    if hit is not None:
        hnorm = normalize_topic(hit)
        if tn == hnorm:
            return {"status": "MATCHED_GOLD_TOPIC", "canonical": hit, "reason": "Exact normalized match to gold/optional title."}
        gold_aliases = {normalize_topic(a) for a in example.topic_aliases.get(hit, [])}
        if tn in gold_aliases or (tn in index and normalize_topic(index[tn]) == hnorm and tn != hnorm):
            # Distinguish: if tn is gold spelling vs alias — already handled exact
            if tn in gold_aliases:
                return {"status": "ALIAS_OF_GOLD_TOPIC", "canonical": hit, "reason": "Exact curated/dataset alias of a gold topic."}
        if _explicit_granularity_pair(hit, title, rep):
            return {"status": "GRANULARITY_VARIANT", "canonical": hit, "reason": "Explicit granularity variant of a gold topic."}
        gt, tt = topic_tokens(hit), topic_tokens(title)
        if gt and tt and (gt < tt or tt < gt):
            return {"status": "GRANULARITY_VARIANT", "canonical": hit, "reason": "Fuzzy containment match (granularity)."}
        return {"status": "MATCHED_GOLD_TOPIC", "canonical": hit, "reason": "Matched to gold/optional via deterministic matcher."}

    # Decomposition component of some gold umbrella
    for umbrella, parts in _DECOMPOSITION_PARTS.items():
        if tn in parts:
            for g in [*example.required_topic_list(), *example.optional_topic_list(), *example.gold_topics]:
                if normalize_topic(g) == umbrella:
                    return {
                        "status": "DECOMPOSITION_COMPONENT",
                        "canonical": g,
                        "reason": "Listed part of a gold umbrella concept.",
                    }

    stripped = _strip_boilerplate(tn)
    if tn in _ABSTRACTION_UMBRELLAS or stripped in _ABSTRACTION_UMBRELLAS:
        return {"status": "ABSTRACTION_VARIANT", "canonical": None, "reason": "Broader umbrella title outside exact gold match."}

    best_g, best_s = "", 0.0
    for g in [*example.required_topic_list(), *example.optional_topic_list(), *example.gold_topics]:
        s = topic_similarity(title, g)
        if s > best_s:
            best_s, best_g = s, g
    if best_g and 0.15 <= best_s < 0.5 and (topic_tokens(title) & topic_tokens(best_g)):
        return {
            "status": "RELATED_BUT_DISTINCT",
            "canonical": best_g,
            "reason": "Related vocabulary to a gold topic without approved equivalence.",
        }

    # Curriculum / goal overlap → out of scope rather than pure hallucination
    if topic_tokens(title) & topic_tokens(example.goal):
        return {
            "status": "OUT_OF_SCOPE",
            "canonical": None,
            "reason": "Shares goal vocabulary but is not an in-scope gold/optional topic.",
        }

    if best_s <= 0.0:
        return {"status": "GENUINE_HALLUCINATION", "canonical": None, "reason": "No overlap with gold/optional topics or goal tokens."}

    return {"status": "UNKNOWN", "canonical": best_g or None, "reason": "Insufficient evidence for a safer generated-topic category."}


def _endpoint_bucket(status: str) -> str:
    if status in _PRESENT_GOLD:
        return "present"
    if status in _GRAN_GOLD:
        return "granularity"
    if status in _DECOMP_GOLD:
        return "decomposition"
    if status in _ABS_GOLD:
        return "abstraction"
    if status == "RELATED_BUT_DISTINCT":
        return "unmatched"
    if status == "MISSING":
        return "missing"
    return "unknown"


def attribute_missing_required_edge(
    frm: str,
    to: str,
    example: EvalExample,
    graph: GeneratedGraph,
    *,
    rep_map: dict[str, Any] | None = None,
) -> dict[str, Any]:
    src = classify_gold_topic_representation(frm, example, graph, rep_map=rep_map)
    tgt = classify_gold_topic_representation(to, example, graph, rep_map=rep_map)
    sb, tb = _endpoint_bucket(src["status"]), _endpoint_bucket(tgt["status"])

    secondary: list[str] = []
    if sb == "present" and tb == "present":
        primary = "EDGE_OMISSION"
        expl = "Both required endpoints are present (exact/alias), but the required edge is absent."
    elif sb == "missing" and tb == "missing":
        primary = "BOTH_ENDPOINTS_MISSING"
        expl = "Neither required endpoint is represented in the generated topic set."
    elif sb == "missing" and tb == "present":
        primary = "SOURCE_ENDPOINT_MISSING"
        expl = "Required source endpoint is missing; target is present."
    elif sb == "present" and tb == "missing":
        primary = "TARGET_ENDPOINT_MISSING"
        expl = "Required target endpoint is missing; source is present."
    elif sb == "granularity" or tb == "granularity":
        primary = "ENDPOINT_GRANULARITY_MISMATCH"
        expl = "At least one endpoint is only represented at different granularity."
        if sb == "missing" or tb == "missing":
            secondary.append("Other endpoint also missing or unmatched.")
    elif sb == "decomposition" or tb == "decomposition":
        primary = "ENDPOINT_DECOMPOSITION"
        expl = "At least one required endpoint appears decomposed into part-concepts."
    elif sb == "abstraction" or tb == "abstraction":
        primary = "ENDPOINT_ABSTRACTION_MISMATCH"
        expl = "At least one required endpoint is abstracted to a broader umbrella."
    elif sb == "unmatched" or tb == "unmatched":
        primary = "ENDPOINT_UNMATCHED"
        expl = "A related generated topic exists for an endpoint but is not an approved equivalent."
    elif sb == "unknown" or tb == "unknown":
        # If one clear missing and other unknown → prefer missing attributions
        if sb == "missing":
            primary = "SOURCE_ENDPOINT_MISSING"
            expl = "Source missing; target representation uncertain."
        elif tb == "missing":
            primary = "TARGET_ENDPOINT_MISSING"
            expl = "Target missing; source representation uncertain."
        else:
            primary = "UNKNOWN"
            expl = "Insufficient deterministic evidence to attribute this missing edge."
    else:
        # mixed present+missing already handled; present+unmatched etc.
        if sb == "present" and tb != "present":
            if tb == "missing":
                primary = "TARGET_ENDPOINT_MISSING"
            elif tb == "unmatched":
                primary = "ENDPOINT_UNMATCHED"
            else:
                primary = "UNKNOWN"
            expl = f"Source present; target status={tgt['status']}."
        elif tb == "present" and sb != "present":
            if sb == "missing":
                primary = "SOURCE_ENDPOINT_MISSING"
            elif sb == "unmatched":
                primary = "ENDPOINT_UNMATCHED"
            else:
                primary = "UNKNOWN"
            expl = f"Target present; source status={src['status']}."
        else:
            primary = "UNKNOWN"
            expl = "Mixed endpoint statuses without a more specific rule."

    assert primary in MISSING_EDGE_ATTRS
    return {
        "case_id": example.id,
        "required_edge": [frm, to],
        "source_status": src["status"],
        "target_status": tgt["status"],
        "primary_attribution": primary,
        "secondary_observations": secondary,
        "generated_topics": list(graph.topics),
        "relevant_candidate_topics": sorted(set((src.get("candidates") or []) + (tgt.get("candidates") or []))),
        "explanation": expl,
    }


def _gen_in_scope(status: str) -> bool:
    return status in _IN_SCOPE_GEN


def attribute_invalid_extra_edge(
    frm: str,
    to: str,
    example: EvalExample,
    graph: GeneratedGraph,
    *,
    rep_map: dict[str, Any] | None = None,
) -> dict[str, Any]:
    src = classify_generated_topic(frm, example, rep_map=rep_map)
    tgt = classify_generated_topic(to, example, rep_map=rep_map)
    ss, ts = src["status"], tgt["status"]

    required_norms = {normalize_topic(t) for t in example.required_topic_list()}
    optional_norms = {normalize_topic(t) for t in example.optional_topic_list()}

    def _canon_bucket(st: dict[str, Any]) -> str:
        can = st.get("canonical")
        if can and normalize_topic(str(can)) in required_norms:
            return "required"
        if can and normalize_topic(str(can)) in optional_norms:
            return "optional"
        return "none"

    sb, tb = _canon_bucket(src), _canon_bucket(tgt)

    if ss == "GRANULARITY_VARIANT" or ts == "GRANULARITY_VARIANT":
        primary = "ENDPOINT_GRANULARITY_DRIFT"
        expl = "Invalid extra edge primarily reflects endpoint granularity drift."
    elif ss == "DECOMPOSITION_COMPONENT" or ts == "DECOMPOSITION_COMPONENT":
        primary = "ENDPOINT_DECOMPOSITION_DRIFT"
        expl = "Invalid extra edge involves a decomposition component of a gold concept."
    elif ss == "ABSTRACTION_VARIANT" or ts == "ABSTRACTION_VARIANT":
        primary = "ENDPOINT_ABSTRACTION_DRIFT"
        expl = "Invalid extra edge involves a broader abstraction than the gold concept."
    elif _gen_in_scope(ss) and _gen_in_scope(ts):
        if sb == "optional" or tb == "optional":
            primary = "CURRICULUM_SCOPE_DRIFT"
            expl = (
                "Endpoints map to expected optional/curriculum concepts, but the edge is outside "
                "the required gold structure."
            )
        else:
            primary = "BOTH_ENDPOINTS_VALID_BUT_EDGE_INVALID"
            expl = "Both endpoints correspond to expected concepts, but the relationship is not required/acceptable."
    elif not _gen_in_scope(ss) and not _gen_in_scope(ts):
        if ss in _OPTIONALISH_GEN or ts in _OPTIONALISH_GEN:
            primary = "CURRICULUM_SCOPE_DRIFT"
            expl = "Endpoints look like extra curriculum content outside the intended required graph."
        elif ss == "GENUINE_HALLUCINATION" and ts == "GENUINE_HALLUCINATION":
            primary = "BOTH_ENDPOINTS_OUT_OF_SCOPE"
            expl = "Neither endpoint is an expected gold/optional concept."
        elif ss in {"OUT_OF_SCOPE", "GENUINE_HALLUCINATION", "RELATED_BUT_DISTINCT", "UNKNOWN"} and ts in {
            "OUT_OF_SCOPE",
            "GENUINE_HALLUCINATION",
            "RELATED_BUT_DISTINCT",
            "UNKNOWN",
        }:
            primary = "BOTH_ENDPOINTS_OUT_OF_SCOPE"
            expl = "Neither endpoint is confidently in the expected conceptual scope."
        else:
            primary = "UNKNOWN"
            expl = "Both endpoints out of matcher scope with mixed statuses."
    elif not _gen_in_scope(ss) and _gen_in_scope(ts):
        primary = "SOURCE_ENDPOINT_OUT_OF_SCOPE"
        expl = "Source endpoint is out of expected scope; target is in-scope."
    elif _gen_in_scope(ss) and not _gen_in_scope(ts):
        primary = "TARGET_ENDPOINT_OUT_OF_SCOPE"
        expl = "Target endpoint is out of expected scope; source is in-scope."
    else:
        primary = "UNKNOWN"
        expl = "Insufficient deterministic evidence for invalid-edge attribution."

    # Prefer curriculum drift when optional-only endpoints are involved and both in-scope
    if primary == "BOTH_ENDPOINTS_VALID_BUT_EDGE_INVALID" and (sb == "optional" and tb == "optional"):
        primary = "CURRICULUM_SCOPE_DRIFT"
        expl = "Both endpoints are optional curriculum topics; edge is outside the required core."

    assert primary in INVALID_EDGE_ATTRS
    return {
        "case_id": example.id,
        "generated_edge": [frm, to],
        "source_node_status": ss,
        "target_node_status": ts,
        "primary_attribution": primary,
        "secondary_observations": [],
        "explanation": expl,
    }


def _rate_map(counter: Counter[str], categories: tuple[str, ...], total: int) -> dict[str, Any]:
    rows = []
    for cat in categories:
        c = int(counter.get(cat, 0))
        rows.append({"attribution": cat, "count": c, "rate": (c / total) if total else 0.0})
    return {"total": total, "by_attribution": rows, "rates": {r["attribution"]: r["rate"] for r in rows}}


def run_node_edge_attribution(
    result_path: str | Path,
    *,
    dataset_path: str | Path | None = None,
    system: str = "synapse",
    output_dir: str | Path | None = None,
    topic_matching_mode: str = "curated_alias",
    edge_mode: str = "edge_calibrated",
    max_cases: int = 12,
) -> Path:
    """Attribute missing required / invalid extra edges; write JSON + Markdown (no rescore)."""
    target = Path(result_path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    examples = {e.id: e for e in load_dataset(dataset_path)}
    rep_map = load_node_representation_map()
    rows = ((payload.get("systems") or {}).get(system) or {}).get("example_results") or []

    missing_records: list[dict[str, Any]] = []
    invalid_records: list[dict[str, Any]] = []
    gold_node_rows: list[dict[str, Any]] = []
    gen_node_rows: list[dict[str, Any]] = []
    case_summaries: list[dict[str, Any]] = []

    for row in rows:
        eid = str(row.get("example_id") or "")
        base = examples.get(eid)
        if base is None:
            continue
        graph = _graph_from_row(row)
        if not graph.parse_ok:
            continue
        ex = adapt_example_for_edge_mode(base, edge_mode, topic_matching_mode=topic_matching_mode)
        comp = compare_graphs(ex, graph)

        for g in ex.required_topic_list():
            st = classify_gold_topic_representation(g, ex, graph, rep_map=rep_map)
            gold_node_rows.append({"case_id": eid, "gold_topic": g, "status": st["status"], "candidates": st["candidates"]})

        for t in graph.topics:
            st = classify_generated_topic(t, ex, rep_map=rep_map)
            gen_node_rows.append(
                {"case_id": eid, "generated_topic": t, "status": st["status"], "canonical": st.get("canonical")},
            )

        case_missing = []
        for edge in comp.get("missing_dependencies") or []:
            rec = attribute_missing_required_edge(str(edge[0]), str(edge[1]), ex, graph, rep_map=rep_map)
            missing_records.append(rec)
            case_missing.append(rec)

        case_invalid = []
        for edge in comp.get("extra_dependencies") or []:
            rec = attribute_invalid_extra_edge(str(edge[0]), str(edge[1]), ex, graph, rep_map=rep_map)
            invalid_records.append(rec)
            case_invalid.append(rec)

        # Case-level primary failure mode
        nodeish_m = {
            "SOURCE_ENDPOINT_MISSING",
            "TARGET_ENDPOINT_MISSING",
            "BOTH_ENDPOINTS_MISSING",
            "ENDPOINT_GRANULARITY_MISMATCH",
            "ENDPOINT_DECOMPOSITION",
            "ENDPOINT_ABSTRACTION_MISMATCH",
            "ENDPOINT_UNMATCHED",
        }
        rel_m = {"EDGE_OMISSION"}
        nodeish_i = {
            "SOURCE_ENDPOINT_OUT_OF_SCOPE",
            "TARGET_ENDPOINT_OUT_OF_SCOPE",
            "BOTH_ENDPOINTS_OUT_OF_SCOPE",
            "ENDPOINT_GRANULARITY_DRIFT",
            "ENDPOINT_DECOMPOSITION_DRIFT",
            "ENDPOINT_ABSTRACTION_DRIFT",
            "CURRICULUM_SCOPE_DRIFT",
        }
        rel_i = {"BOTH_ENDPOINTS_VALID_BUT_EDGE_INVALID"}

        n_node = sum(1 for r in case_missing if r["primary_attribution"] in nodeish_m) + sum(
            1 for r in case_invalid if r["primary_attribution"] in nodeish_i
        )
        n_rel = sum(1 for r in case_missing if r["primary_attribution"] in rel_m) + sum(
            1 for r in case_invalid if r["primary_attribution"] in rel_i
        )
        n_unk = sum(1 for r in case_missing if r["primary_attribution"] == "UNKNOWN") + sum(
            1 for r in case_invalid if r["primary_attribution"] == "UNKNOWN"
        )
        if n_node == 0 and n_rel == 0 and n_unk == 0:
            dominant = "NONE"
        elif n_node >= n_rel and n_node >= n_unk:
            dominant = "NODE"
        elif n_rel >= n_node and n_rel >= n_unk:
            dominant = "RELATIONSHIP"
        else:
            dominant = "UNRESOLVED"

        case_summaries.append(
            {
                "case_id": eid,
                "goal": ex.goal,
                "gold_topics": ex.required_topic_list(),
                "optional_topics": ex.optional_topic_list(),
                "generated_topics": list(graph.topics),
                "gold_dependencies": [list(d) for d in ex.required_dependency_list()],
                "generated_dependencies": [list(d) for d in graph.dependencies],
                "missing_edge_attributions": case_missing,
                "invalid_edge_attributions": case_invalid,
                "dominant_failure": dominant,
                "counts": {"node": n_node, "relationship": n_rel, "unresolved": n_unk},
            },
        )

    miss_ctr = Counter(r["primary_attribution"] for r in missing_records)
    inv_ctr = Counter(r["primary_attribution"] for r in invalid_records)
    gold_ctr = Counter(r["status"] for r in gold_node_rows)
    gen_ctr = Counter(r["status"] for r in gen_node_rows)

    miss_total = len(missing_records)
    inv_total = len(invalid_records)

    node_missing = {
        "SOURCE_ENDPOINT_MISSING",
        "TARGET_ENDPOINT_MISSING",
        "BOTH_ENDPOINTS_MISSING",
        "ENDPOINT_GRANULARITY_MISMATCH",
        "ENDPOINT_DECOMPOSITION",
        "ENDPOINT_ABSTRACTION_MISMATCH",
        "ENDPOINT_UNMATCHED",
    }
    rel_missing = {"EDGE_OMISSION"}
    node_invalid = {
        "SOURCE_ENDPOINT_OUT_OF_SCOPE",
        "TARGET_ENDPOINT_OUT_OF_SCOPE",
        "BOTH_ENDPOINTS_OUT_OF_SCOPE",
        "ENDPOINT_GRANULARITY_DRIFT",
        "ENDPOINT_DECOMPOSITION_DRIFT",
        "ENDPOINT_ABSTRACTION_DRIFT",
        "CURRICULUM_SCOPE_DRIFT",
    }
    rel_invalid = {"BOTH_ENDPOINTS_VALID_BUT_EDGE_INVALID"}

    n_node = sum(miss_ctr[c] for c in node_missing) + sum(inv_ctr[c] for c in node_invalid)
    n_rel = sum(miss_ctr[c] for c in rel_missing) + sum(inv_ctr[c] for c in rel_invalid)
    n_unk = miss_ctr["UNKNOWN"] + inv_ctr["UNKNOWN"]
    struct_total = miss_total + inv_total

    overall = {
        "NODE_SELECTION_OR_REPRESENTATION_ERROR": {
            "count": n_node,
            "rate": (n_node / struct_total) if struct_total else 0.0,
        },
        "RELATIONSHIP_GENERATION_ERROR": {
            "count": n_rel,
            "rate": (n_rel / struct_total) if struct_total else 0.0,
        },
        "UNRESOLVED_ERROR": {
            "count": n_unk,
            "rate": (n_unk / struct_total) if struct_total else 0.0,
        },
        "structural_disagreement_total": struct_total,
        "missing_required_edge_total": miss_total,
        "invalid_extra_edge_total": inv_total,
    }

    if n_node > n_rel and n_node > n_unk:
        diagnosis = "A. NODE_GENERATION is the primary bottleneck."
        diagnosis_code = "NODE_GENERATION"
    elif n_rel > n_node and n_rel > n_unk:
        diagnosis = "B. RELATIONSHIP_GENERATION is the primary bottleneck."
        diagnosis_code = "RELATIONSHIP_GENERATION"
    elif n_node == 0 and n_rel == 0:
        diagnosis = "D. Insufficient evidence."
        diagnosis_code = "INSUFFICIENT_EVIDENCE"
    else:
        diagnosis = "C. Mixed bottleneck."
        diagnosis_code = "MIXED"

    # Representative cases: diversify domains / dominant failures
    reps: list[dict[str, Any]] = []
    seen_dom: set[str] = set()
    for c in sorted(case_summaries, key=lambda x: -(x["counts"]["node"] + x["counts"]["relationship"] + x["counts"]["unresolved"])):
        if len(reps) >= max_cases:
            break
        if c["dominant_failure"] == "NONE" and (c["counts"]["node"] + c["counts"]["relationship"] + c["counts"]["unresolved"]) == 0:
            continue
        key = c["dominant_failure"] + ":" + c["case_id"].split("_")[0]
        if key in seen_dom and len(reps) >= 6:
            continue
        seen_dom.add(key)
        reps.append(c)
    # fill
    for c in case_summaries:
        if len(reps) >= max_cases:
            break
        if c not in reps and (c["counts"]["node"] + c["counts"]["relationship"] + c["counts"]["unresolved"]) > 0:
            reps.append(c)

    stamp = datetime.now(timezone.utc)
    artifact = {
        "timestamp": stamp.isoformat(),
        "source_benchmark": str(target),
        "system": system,
        "topic_matching_mode": topic_matching_mode,
        "edge_mode": edge_mode,
        "note": (
            "Diagnostic attribution only. Does not change Topic F1, Dependency F1, "
            "aliases, acceptable alternatives, or production behavior. No LLM judge."
        ),
        "missing_required_edge_attribution": _rate_map(miss_ctr, MISSING_EDGE_ATTRS, miss_total),
        "invalid_extra_edge_attribution": _rate_map(inv_ctr, INVALID_EDGE_ATTRS, inv_total),
        "gold_topic_coverage": {
            "total": len(gold_node_rows),
            "by_status": dict(gold_ctr),
            "rates": {k: (gold_ctr[k] / len(gold_node_rows) if gold_node_rows else 0.0) for k in GOLD_NODE_STATUSES},
        },
        "generated_topic_coverage": {
            "total": len(gen_node_rows),
            "by_status": dict(gen_ctr),
            "rates": {k: (gen_ctr[k] / len(gen_node_rows) if gen_node_rows else 0.0) for k in GEN_NODE_STATUSES},
        },
        "overall_error_split": overall,
        "diagnosis": diagnosis,
        "diagnosis_code": diagnosis_code,
        "missing_edge_records": missing_records,
        "invalid_edge_records": invalid_records,
        "representative_cases": reps,
        "metric_invariants": {
            "aliases_unchanged": True,
            "acceptable_edges_unchanged": True,
            "scores_not_rescored": True,
        },
    }

    out = Path(output_dir) if output_dir else _REPO_ROOT / "results" / "failure_analysis"
    out.mkdir(parents=True, exist_ok=True)
    stamp_s = stamp.strftime("%Y-%m-%d_%H%M%S")
    path = out / f"{stamp_s}_node_edge_attribution.json"
    path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md_path = out / f"{stamp_s}_node_vs_edge_error_analysis.md"
    md: list[str] = [
        f"# Node vs relationship error attribution — {artifact['timestamp']}",
        "",
        f"- Source: `{target}`",
        f"- Topic matching: `{topic_matching_mode}`",
        f"- Edge mode: `{edge_mode}`",
        f"- Diagnosis: **{diagnosis}**",
        "",
        "## 1. Overall summary",
        "",
        f"- Missing required edges: {miss_total}",
        f"- Invalid extra edges: {inv_total}",
        f"- Structural disagreements attributed: {struct_total}",
        "",
        "| Error source | Count | Rate |",
        "| --- | ---: | ---: |",
        f"| NODE_SELECTION_OR_REPRESENTATION_ERROR | {n_node} | {(n_node/struct_total if struct_total else 0):.3f} |",
        f"| RELATIONSHIP_GENERATION_ERROR | {n_rel} | {(n_rel/struct_total if struct_total else 0):.3f} |",
        f"| UNRESOLVED_ERROR | {n_unk} | {(n_unk/struct_total if struct_total else 0):.3f} |",
        "",
        "## 2. Missing required edge attribution",
        "",
        "| Attribution | Count | Rate |",
        "| --- | ---: | ---: |",
    ]
    for row in artifact["missing_required_edge_attribution"]["by_attribution"]:
        md.append(f"| {row['attribution']} | {row['count']} | {row['rate']:.3f} |")
    md.extend(
        [
            "",
            "## 3. Invalid extra edge attribution",
            "",
            "| Attribution | Count | Rate |",
            "| --- | ---: | ---: |",
        ],
    )
    for row in artifact["invalid_extra_edge_attribution"]["by_attribution"]:
        md.append(f"| {row['attribution']} | {row['count']} | {row['rate']:.3f} |")
    md.extend(
        [
            "",
            "## 4. Node coverage",
            "",
            "### Gold topics",
            "",
        ],
    )
    for k in GOLD_NODE_STATUSES:
        md.append(f"- {k}: {gold_ctr.get(k, 0)}")
    md.extend(["", "### Generated topics", ""])
    for k in GEN_NODE_STATUSES:
        md.append(f"- {k}: {gen_ctr.get(k, 0)}")
    md.extend(["", "## 5. Representative cases", ""])
    for c in reps:
        md.append(f"### `{c['case_id']}` — dominant: **{c['dominant_failure']}**")
        md.append(f"- Goal: {c['goal']}")
        md.append(f"- Gold topics: {c['gold_topics']}")
        md.append(f"- Generated topics: {c['generated_topics']}")
        md.append(f"- Gold deps: {c['gold_dependencies']}")
        md.append(f"- Generated deps: {c['generated_dependencies']}")
        md.append(
            "- Missing attributions: "
            + ", ".join(f"{r['required_edge']}:{r['primary_attribution']}" for r in c["missing_edge_attributions"][:6]),
        )
        md.append(
            "- Invalid attributions: "
            + ", ".join(f"{r['generated_edge']}:{r['primary_attribution']}" for r in c["invalid_edge_attributions"][:6]),
        )
        md.append("")
    md.extend(
        [
            "## Interpretation",
            "",
            "This analysis does not improve Synapse generation. It explains where measured "
            "structural errors originate under the current curated-alias + edge-calibrated baseline.",
            "",
        ],
    )
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    artifact["markdown_report"] = str(md_path)
    path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
