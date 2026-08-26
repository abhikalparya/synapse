"""Write structured benchmark JSON (+ optional markdown summary)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.evaluation.failure_analysis import format_failure_table


def default_results_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "results" / "benchmarks"


def result_filename(*, model: str, timestamp: datetime | None = None, benchmark_type: str | None = None, prompt_variant: str | None = None) -> str:
    ts = timestamp or datetime.now(timezone.utc)
    safe_model = "".join(ch if ch.isalnum() or ch in "-._" else "-" for ch in (model or "unknown"))
    stamp = ts.strftime("%Y-%m-%d_%H%M%S")
    parts = [stamp]
    if benchmark_type:
        safe_type = "".join(ch if ch.isalnum() or ch in "-._" else "-" for ch in benchmark_type)
        parts.append(safe_type)
    parts.append(safe_model)
    if prompt_variant:
        safe_pv = "".join(ch if ch.isalnum() or ch in "-._" else "-" for ch in prompt_variant)
        parts.append(safe_pv)
    return "_".join(parts) + ".json"


def write_benchmark_result(result: dict[str, Any], output_dir: str | Path | None = None) -> Path:
    out = Path(output_dir) if output_dir else default_results_dir()
    out.mkdir(parents=True, exist_ok=True)
    model = str(result.get("model") or "unknown")
    path = out / result_filename(
        model=model,
        benchmark_type=result.get("benchmark_type"),
        prompt_variant=(
            result.get("prompt_variant")
            if result.get("benchmark_type") in {"quality", "quality_stability"}
            else None
        ),
    )
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md_path = path.with_suffix(".md")
    md_path.write_text(render_markdown_report(result), encoding="utf-8")
    return path


def render_markdown_report(result: dict[str, Any]) -> str:
    btype = str(result.get("benchmark_type") or "quality")
    lines = [
        f"# Benchmark report — {result.get('timestamp', '')}",
        "",
        f"- Benchmark: `{btype}`",
        f"- Dataset: `{result.get('dataset')}`",
        f"- Dataset version: `{result.get('dataset_version', result.get('dataset'))}`",
        f"- Model: `{result.get('model')}`",
        f"- Provider: `{result.get('provider')}`",
        f"- Seed: `{result.get('seed')}`",
        f"- Examples: {result.get('example_count')}",
        f"- Repetitions: {result.get('repetitions')}",
    ]
    if result.get("prompt_variant"):
        lines.append(f"- Prompt variant: `{result.get('prompt_variant')}`")
        lines.append(f"- Prompt version: `{result.get('prompt_version')}`")
        lines.append(f"- Prompt hash: `{result.get('prompt_hash')}`")
    lines.append("")
    if btype == "reliability":
        m = result.get("metrics") or {}
        lines.extend(
            [
                "## Reliability",
                "",
                "| Metric | Rate |",
                "| --- | ---: |",
                f"| Validation catch rate | {float(m.get('validation_catch_rate') or 0):.3f} |",
                f"| Cycle prevention rate | {float(m.get('cycle_prevention_rate') or 0):.3f} |",
                f"| Invalid reference rejection rate | {float(m.get('invalid_reference_rejection_rate') or 0):.3f} |",
                f"| Transaction integrity rate | {float(m.get('transaction_integrity_rate') or 0):.3f} |",
                f"| Rollback correctness rate | {float(m.get('rollback_correctness_rate') or 0):.3f} |",
                "",
            ],
        )
        return "\n".join(lines)

    if btype == "audit":
        m = result.get("metrics") or {}
        lines.extend(
            [
                "## Audit detection",
                "",
                "| Split | Precision | Recall | False positive rate |",
                "| --- | ---: | ---: | ---: |",
                f"| Overall | {float(m.get('precision') or 0):.3f} | {float(m.get('recall') or 0):.3f} | {float(m.get('false_positive_rate') or 0):.3f} |",
                f"| Structural | {float(m.get('structural_precision') or 0):.3f} | {float(m.get('structural_recall') or 0):.3f} | {float(m.get('structural_false_positive_rate') or 0):.3f} |",
                f"| Semantic | {m.get('semantic_precision')} | {m.get('semantic_recall')} | {m.get('semantic_false_positive_rate')} |",
                "",
                f"Semantic mode: `{m.get('semantic_mode')}`",
                "",
                "## Before / after eval-only repair",
                "",
                f"- Dependency F1 delta: {float(m.get('dependency_f1_delta') or 0):+.3f}",
                f"- Dependency recall delta: {float(m.get('dependency_recall_delta') or 0):+.3f}",
                f"- Missing prerequisite rate delta: {float(m.get('missing_prerequisite_rate_delta') or 0):+.3f}",
                f"- Improved cases: {int(m.get('repair_improved') or 0)} / {int(m.get('repair_cases') or 0)}",
                f"- Regressed cases: {int(m.get('repair_regressed') or 0)}",
                "",
            ],
        )
        return "\n".join(lines)

    systems = result.get("systems") or {}
    lines.extend(
        [
            "## Graph quality (macro-average)",
            "",
            "| System | Topic F1 | Dependency F1 | Missing Prereq | Direction Error | Extra Dep | Redundant Transitive | Valid |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ],
    )
    for name, block in systems.items():
        m = block.get("metrics") or {}
        lines.append(
            "| {sys} | {tf1:.3f} | {df1:.3f} | {miss:.3f} | {dire:.3f} | {extra:.3f} | {red:.3f} | {valid:.3f} |".format(
                sys=name,
                tf1=float(m.get("topic_f1") or 0),
                df1=float(m.get("dependency_f1") or 0),
                miss=float(m.get("missing_prerequisite_rate") or 0),
                dire=float(m.get("dependency_direction_error_rate") or 0),
                extra=float(m.get("extra_dependency_rate") or 0),
                red=float(m.get("redundant_transitive_edge_rate") or 0),
                valid=float(m.get("graph_validity_rate") or 0),
            ),
        )

    if systems:
        lines.extend(["", "## Latency (graph-quality path)", ""])
        lines.append("| System | samples | p50_ms | p95_ms | mean_ms |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for name, block in systems.items():
            lat = block.get("latency") or {}
            lines.append(
                f"| {name} | {lat.get('samples', 0)} | {lat.get('p50_ms', 0)} | {lat.get('p95_ms', 0)} | {lat.get('mean_ms', 0)} |",
            )

        lines.extend(["", "## Estimated cost", ""])
        lines.append("| System | avg_cost_usd | total_cost_usd | note |")
        lines.append("| --- | ---: | ---: | --- |")
        for name, block in systems.items():
            cost = block.get("cost") or {}
            lines.append(
                f"| {name} | {cost.get('average_cost_usd')} | {cost.get('total_cost_usd')} | {cost.get('note', '')} |",
            )

    ops = result.get("operation_latency") or {}
    if ops:
        lines.extend(["", "## Operation latency", ""])
        lines.append("| Operation | samples | p50_ms | p95_ms | mean_ms |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for op, lat in ops.items():
            lines.append(
                f"| {op} | {lat.get('samples', 0)} | {lat.get('p50_ms', 0)} | {lat.get('p95_ms', 0)} | {lat.get('mean_ms', 0)} |",
            )

    failures = result.get("failures") or {}
    if isinstance(failures, dict) and failures and all(isinstance(v, int) for v in failures.values()):
        lines.extend(["", "## Failures", "", "```", format_failure_table(failures), "```", ""])

    models = result.get("model_comparison")
    if models:
        lines.extend(["", "## Multi-model comparison", ""])
        lines.append("| Model | Topic F1 | Dependency F1 | Missing Prereq | p50_ms | Avg Cost |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for row in models:
            lines.append(
                "| {m} | {tf1:.3f} | {df1:.3f} | {miss:.3f} | {p50} | {cost} |".format(
                    m=row.get("model"),
                    tf1=float(row.get("topic_f1") or 0),
                    df1=float(row.get("dependency_f1") or 0),
                    miss=float(row.get("missing_prerequisite_rate") or 0),
                    p50=row.get("p50_ms"),
                    cost=row.get("avg_cost_usd"),
                ),
            )

    prop = result.get("proposal_metrics") or {}
    if prop:
        lines.extend(
            [
                "## Proposal / human-feedback metrics",
                "",
                "```json",
                json.dumps(prop, indent=2),
                "```",
                "",
                "_Do not invent numbers. Empty/null rates mean insufficient recorded events._",
                "",
            ],
        )

    per_variant = result.get("per_variant")
    if isinstance(per_variant, dict) and per_variant:
        lines.extend(["", "## Prompt variant A/B summary", ""])
        lines.append("| Variant | Topic F1 | Dep F1 | Extra Dep | Redundant Transitive | Artifact |")
        lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
        for name, block in per_variant.items():
            sys_metrics = block.get("metrics") or {}
            if isinstance(sys_metrics, dict) and "synapse" in sys_metrics:
                m = sys_metrics.get("synapse") or {}
            elif isinstance(sys_metrics, dict) and "direct_llm_graph" in sys_metrics:
                m = sys_metrics.get("direct_llm_graph") or {}
            else:
                m = sys_metrics if isinstance(sys_metrics, dict) else {}
            lines.append(
                "| {v} | {tf1:.3f} | {df1:.3f} | {extra:.3f} | {red:.3f} | `{art}` |".format(
                    v=name,
                    tf1=float(m.get("topic_f1") or 0),
                    df1=float(m.get("dependency_f1") or 0),
                    extra=float(m.get("extra_dependency_rate") or 0),
                    red=float(m.get("redundant_transitive_edge_rate") or 0),
                    art=block.get("artifact") or "",
                ),
            )
        lines.append("")

    return "\n".join(lines)
