"""Deterministic unmatched-topic equivalence classification (no LLM).

Conservative by design: when rules disagree or evidence is weak, emit UNKNOWN.
Only EXACT_ALIAS / CLEAR_SYNONYM / TITLE_PARAPHRASE may be proposed as alias
candidates. Granularity / decomposition / abstraction / related / hallucination
must not become aliases.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.evaluation.dataset import load_dataset
from app.evaluation.inspect import _graph_from_row
from app.evaluation.matching_modes import (
    adapt_example_for_mode,
    approved_alias_map,
    load_curated_aliases,
)
from app.evaluation.metrics import (
    compare_graphs,
    normalize_topic,
    topic_similarity,
    topic_tokens,
)
from app.evaluation.schemas import EvalExample, GeneratedGraph

EQUIVALENCE_CLASSES = (
    "EXACT_ALIAS",
    "CLEAR_SYNONYM",
    "TITLE_PARAPHRASE",
    "GRANULARITY_VARIANT",
    "CONCEPT_DECOMPOSITION",
    "CONCEPT_ABSTRACTION",
    "SEMANTICALLY_RELATED_BUT_DISTINCT",
    "GENUINE_HALLUCINATION",
    "UNKNOWN",
)

_ALIAS_CANDIDATE_CLASSES = frozenset({"EXACT_ALIAS", "CLEAR_SYNONYM", "TITLE_PARAPHRASE"})

_BOILERPLATE_PREFIXES = (
    "introduction to ",
    "intro to ",
    "basics of ",
    "basics ",
    "overview of ",
    "overview ",
    "understanding ",
    "getting started with ",
    "getting started ",
    "advanced ",
    "fundamentals of ",
    "fundamentals ",
)

# Hand-maintained synonym surfaces (normalized). Classification proposals only.
_KNOWN_SYNONYM_NORMS: dict[str, str] = {
    "control structures": "control flow",
    "control structure": "control flow",
    "symmetric encryption": "symmetric crypto",
    "asymmetric encryption": "public key crypto",
    "public key encryption": "public key crypto",
    "hash functions": "hashing",
    "hash function": "hashing",
    "incident management": "incident response",
    "asynchronous processing": "async processing",
    "cross site scripting": "xss",
    "cross site scripting xss": "xss",
    "cross site request forgery": "csrf",
    "cross site request forgery csrf": "csrf",
    "data retrieval with select": "select queries",
    "sorting data with order by": "sorting results",
}

_DECOMPOSITION_PARTS: dict[str, frozenset[str]] = {
    # Keys/values are normalize_topic() forms.
    "linear algebra": frozenset(
        {
            normalize_topic("Vectors"),
            normalize_topic("Matrices"),
            normalize_topic("Matrix"),
            normalize_topic("Determinants"),
            normalize_topic("Determinant"),
            normalize_topic("Matrix Operations"),
        },
    ),
    "programming fundamental": frozenset(
        {
            normalize_topic("Variables"),
            normalize_topic("Data Types"),
            normalize_topic("Control Flow"),
            normalize_topic("Functions"),
            normalize_topic("Loops"),
            normalize_topic("Conditionals"),
        },
    ),
    "probability": frozenset(
        {
            normalize_topic("Statistics"),
            normalize_topic("Combinatorics"),
            normalize_topic("Independence of Events"),
        },
    ),
    "compiler": frozenset(
        {
            normalize_topic("Lexical Analysis"),
            normalize_topic("Syntax Analysis"),
            normalize_topic("Code Optimization"),
        },
    ),
    "real analysis": frozenset(
        {
            normalize_topic("Integration"),
            normalize_topic("Sets and Functions"),
        },
    ),
}

_ABSTRACTION_UMBRELLAS = frozenset(
    {
        normalize_topic("Programming Fundamentals"),
        normalize_topic("Computer Science"),
        normalize_topic("Mathematics"),
        normalize_topic("Machine Learning"),
        normalize_topic("Software Engineering"),
        normalize_topic("Cloud Computing Basics"),
        normalize_topic("Web Application Basics"),
        normalize_topic("Introduction to Databases"),
        normalize_topic("Distributed Systems Fundamentals"),
        normalize_topic("Distributed Systems Basics"),
    },
)

_PAREN_ACRONYM_RE = re.compile(r"\(([^)]+)\)\s*$")


def _strip_boilerplate(norm: str) -> str:
    s = norm
    changed = True
    while changed:
        changed = False
        for p in _BOILERPLATE_PREFIXES:
            if s.startswith(p):
                s = s[len(p) :].strip()
                changed = True
    return s


def _gold_pool(example: EvalExample) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for t in [*example.required_topic_list(), *example.optional_topic_list(), *example.gold_topics]:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _best_candidate(generated: str, example: EvalExample) -> tuple[str, float]:
    best_title = ""
    best = 0.0
    for gold in _gold_pool(example):
        score = topic_similarity(generated, gold)
        if score > best:
            best = score
            best_title = gold
        for alias in example.topic_aliases.get(gold, []):
            score = topic_similarity(generated, alias)
            if score > best:
                best = score
                best_title = gold
    return best_title, best


def _record(
    generated: str,
    gold: str,
    classification: str,
    reason: str,
    *,
    candidate_alias: bool,
    sim: float,
) -> dict[str, Any]:
    return {
        "generated_topic": generated,
        "candidate_gold_topic": gold,
        "proposed_classification": classification,
        "reason": reason,
        "similarity": round(float(sim), 3),
        "candidate_alias": bool(candidate_alias and classification in _ALIAS_CANDIDATE_CLASSES),
    }


def classify_unmatched_topic(
    generated: str,
    example: EvalExample,
    *,
    candidate_gold: str | None = None,
    similarity: float | None = None,
) -> dict[str, Any]:
    """Classify one unmatched generated title. Deterministic; prefers UNKNOWN over guessing."""
    gold = candidate_gold
    sim = similarity
    if gold is None or sim is None:
        gold, sim = _best_candidate(generated, example)

    gen_n = normalize_topic(generated)
    gold_n = normalize_topic(gold) if gold else ""
    gen_tokens = topic_tokens(generated)
    gold_tokens = topic_tokens(gold) if gold else set()
    stripped = _strip_boilerplate(gen_n)

    # Decomposition / abstraction against the full gold pool (even if similarity to closest is 0).
    for umbrella, parts in _DECOMPOSITION_PARTS.items():
        if gen_n in parts or any(gen_n == p or gen_n.startswith(p + " ") for p in parts):
            for g in _gold_pool(example):
                if normalize_topic(g) == umbrella:
                    return _record(
                        generated,
                        g,
                        "CONCEPT_DECOMPOSITION",
                        "Generated title decomposes a broader gold concept; not an alias.",
                        candidate_alias=False,
                        sim=topic_similarity(generated, g),
                    )

    if stripped in _ABSTRACTION_UMBRELLAS or gen_n in _ABSTRACTION_UMBRELLAS:
        specific = [g for g in _gold_pool(example) if normalize_topic(g) not in _ABSTRACTION_UMBRELLAS]
        if specific:
            target = gold if gold in specific else specific[0]
            return _record(
                generated,
                target,
                "CONCEPT_ABSTRACTION",
                "Generated title is a broader umbrella than the gold concept.",
                candidate_alias=False,
                sim=topic_similarity(generated, target),
            )

    m = _PAREN_ACRONYM_RE.search(generated.strip())
    if m and gold:
        acr = normalize_topic(m.group(1))
        if acr == gold_n and acr:
            return _record(
                generated,
                gold,
                "EXACT_ALIAS",
                "Generated title expands the gold acronym in parentheses.",
                candidate_alias=True,
                sim=sim,
            )

    if gold and _KNOWN_SYNONYM_NORMS.get(gen_n) == gold_n:
        return _record(
            generated,
            gold,
            "CLEAR_SYNONYM",
            "Hand-listed synonym pair for the same concept surface form.",
            candidate_alias=True,
            sim=sim,
        )
    if gold and _KNOWN_SYNONYM_NORMS.get(stripped) == gold_n:
        return _record(
            generated,
            gold,
            "CLEAR_SYNONYM",
            "Boilerplate-stripped title matches a known synonym of the gold concept.",
            candidate_alias=True,
            sim=sim,
        )

    if gold and stripped and stripped == gold_n:
        return _record(
            generated,
            gold,
            "TITLE_PARAPHRASE",
            "Same concept after removing tutorial boilerplate (Introduction/Basics/Overview).",
            candidate_alias=True,
            sim=sim,
        )

    if gold and gold_n in _DECOMPOSITION_PARTS:
        parts = _DECOMPOSITION_PARTS[gold_n]
        if gen_n in parts or any(p in gen_n for p in parts):
            return _record(
                generated,
                gold,
                "CONCEPT_DECOMPOSITION",
                "Generated title is a part/sub-concept of the gold umbrella, not an equivalent name.",
                candidate_alias=False,
                sim=sim,
            )

    if gold and gen_tokens and gold_tokens:
        if gen_tokens < gold_tokens or gold_tokens < gen_tokens:
            smaller, larger = (
                (gen_tokens, gold_tokens) if len(gen_tokens) < len(gold_tokens) else (gold_tokens, gen_tokens)
            )
            leftover = larger - smaller
            substantive = {t for t in leftover if t not in {"and", "or", "with", "of", "in", "for", "the", "a", "an"}}
            if substantive and sim < 0.75:
                return _record(
                    generated,
                    gold,
                    "GRANULARITY_VARIANT",
                    "Titles share tokens but differ in scope/granularity; not automatic aliases.",
                    candidate_alias=False,
                    sim=sim,
                )

    if gold and sim >= 0.15 and sim < 0.5 and (gen_tokens & gold_tokens):
        return _record(
            generated,
            gold,
            "SEMANTICALLY_RELATED_BUT_DISTINCT",
            "Shares vocabulary with gold but is a distinct concept under conservative rules.",
            candidate_alias=False,
            sim=sim,
        )

    if gold and sim >= 0.5:
        return _record(
            generated,
            gold,
            "UNKNOWN",
            "High token overlap but not an approved synonym/paraphrase rule; needs human review.",
            candidate_alias=False,
            sim=sim,
        )

    if not gold or sim <= 0.0:
        return _record(
            generated,
            gold or "",
            "GENUINE_HALLUCINATION",
            "No overlapping gold title under deterministic similarity; treat as out-of-reference.",
            candidate_alias=False,
            sim=sim or 0.0,
        )

    return _record(
        generated,
        gold,
        "UNKNOWN",
        "Insufficient deterministic evidence for a safer category.",
        candidate_alias=False,
        sim=sim,
    )


def extract_unmatched_topics(example: EvalExample, graph: GeneratedGraph) -> list[str]:
    if not graph.parse_ok:
        return []
    fair = adapt_example_for_mode(example, "fair")
    comparison = compare_graphs(fair, graph)
    return list(comparison.get("extra_topics") or [])


def build_topic_equivalence_review(
    result_path: str | Path,
    *,
    dataset_path: str | Path | None = None,
    system: str = "synapse",
    output_dir: str | Path | None = None,
    curated_path: str | Path | None = None,
) -> Path:
    """Write human-reviewable unmatched-topic classifications (approved_alias default false)."""
    target = Path(result_path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    examples = {e.id: e for e in load_dataset(dataset_path)}
    registry = load_curated_aliases(curated_path)
    approved = approved_alias_map(registry)
    approved_pairs = {
        (normalize_topic(canon), normalize_topic(alias))
        for canon, aliases in approved.items()
        for alias in aliases
    }

    rows = ((payload.get("systems") or {}).get(system) or {}).get("example_results") or []
    records: list[dict[str, Any]] = []
    for row in rows:
        eid = str(row.get("example_id") or "")
        example = examples.get(eid)
        if example is None:
            continue
        graph = _graph_from_row(row)
        for title in extract_unmatched_topics(example, graph):
            classified = classify_unmatched_topic(title, adapt_example_for_mode(example, "fair"))
            key = (normalize_topic(classified["candidate_gold_topic"]), normalize_topic(title))
            records.append(
                {
                    "case_id": eid,
                    "generated_topic": title,
                    "candidate_gold_topic": classified["candidate_gold_topic"],
                    "current_match_status": "UNMATCHED",
                    "proposed_classification": classified["proposed_classification"],
                    "reason": classified["reason"],
                    "similarity": classified["similarity"],
                    "candidate_alias": classified["candidate_alias"],
                    "approved_alias": key in approved_pairs,
                },
            )

    by_class = dict(Counter(r["proposed_classification"] for r in records))
    artifact = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_benchmark": str(target),
        "system": system,
        "matching_baseline": "fair",
        "note": (
            "Deterministic classification of topics unmatched under FAIR matching. "
            "approved_alias is true only when present in curated_aliases_v1 with approved=true. "
            "Do not bulk-approve candidates. No LLM classifier."
        ),
        "summary": {
            "unmatched_topics": len(records),
            "by_classification": dict(sorted(by_class.items(), key=lambda kv: (-kv[1], kv[0]))),
            "candidate_aliases": sum(1 for r in records if r["candidate_alias"]),
            "approved_aliases_applied": sum(1 for r in records if r["approved_alias"]),
        },
        "records": records,
    }

    out = Path(output_dir) if output_dir else Path(__file__).resolve().parents[3] / "results" / "failure_analysis"
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    path = out / f"{stamp}_topic_equivalence_review.json"
    path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md_lines = [
        f"# Topic equivalence review — {artifact['timestamp']}",
        "",
        f"- Source: `{target}`",
        f"- Unmatched topics: {len(records)}",
        f"- Candidate aliases (proposed): {artifact['summary']['candidate_aliases']}",
        f"- Already approved in curated registry: {artifact['summary']['approved_aliases_applied']}",
        "",
        "## Classification breakdown",
        "",
    ]
    for k, v in artifact["summary"]["by_classification"].items():
        md_lines.append(f"- {k}: {v}")
    md_lines.extend(["", "## Records", ""])
    for r in records:
        md_lines.append(
            f"- `{r['case_id']}`: {r['generated_topic']!r} → {r['candidate_gold_topic']!r} "
            f"[{r['proposed_classification']}] candidate={r['candidate_alias']} approved={r['approved_alias']} "
            f"— {r['reason']}",
        )
    path.with_suffix(".md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return path
