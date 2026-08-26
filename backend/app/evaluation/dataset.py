"""Load and validate the golden evaluation dataset (JSONL)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.evaluation.schemas import EvalExample

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_PATH = _REPO_ROOT / "data" / "eval" / "learning_graph_eval_v1.jsonl"

_REQUIRED = ("id", "category", "difficulty", "goal", "gold_topics", "gold_dependencies")
_DIFFICULTIES = {"beginner", "intermediate", "advanced"}


class DatasetError(ValueError):
    """Raised when the dataset file is malformed or fails structural checks."""


def _parse_dependencies(raw: Any, example_id: str) -> list[tuple[str, str]]:
    if not isinstance(raw, list):
        raise DatasetError(f"{example_id}: gold_dependencies must be a list")
    out: list[tuple[str, str]] = []
    for i, row in enumerate(raw):
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise DatasetError(f"{example_id}: gold_dependencies[{i}] must be a [from, to] pair")
        frm, to = str(row[0]).strip(), str(row[1]).strip()
        if not frm or not to:
            raise DatasetError(f"{example_id}: gold_dependencies[{i}] has empty endpoint")
        out.append((frm, to))
    return out


def parse_example(obj: dict[str, Any], *, line_no: int | None = None) -> EvalExample:
    loc = f"line {line_no}" if line_no is not None else obj.get("id", "<unknown>")
    if not isinstance(obj, dict):
        raise DatasetError(f"{loc}: example must be a JSON object")
    for key in _REQUIRED:
        if key not in obj:
            raise DatasetError(f"{loc}: missing required field {key!r}")

    example_id = str(obj["id"]).strip()
    if not example_id:
        raise DatasetError(f"{loc}: id must be non-empty")

    difficulty = str(obj["difficulty"]).strip()
    if difficulty not in _DIFFICULTIES:
        raise DatasetError(f"{example_id}: difficulty must be one of {_DIFFICULTIES}")

    gold_topics_raw = obj["gold_topics"]
    if not isinstance(gold_topics_raw, list) or not gold_topics_raw:
        raise DatasetError(f"{example_id}: gold_topics must be a non-empty list")
    gold_topics = [str(t).strip() for t in gold_topics_raw]
    if any(not t for t in gold_topics):
        raise DatasetError(f"{example_id}: gold_topics contains an empty title")

    gold_deps = _parse_dependencies(obj["gold_dependencies"], example_id)
    topic_set = set(gold_topics)
    for frm, to in gold_deps:
        if frm not in topic_set or to not in topic_set:
            raise DatasetError(
                f"{example_id}: dependency {frm!r} -> {to!r} references a topic not in gold_topics",
            )

    aliases_raw = obj.get("topic_aliases") or {}
    if not isinstance(aliases_raw, dict):
        raise DatasetError(f"{example_id}: topic_aliases must be an object")
    topic_aliases = {str(k): [str(a) for a in (v or [])] for k, v in aliases_raw.items()}

    extras_raw = obj.get("allowed_extra_topics") or []
    if not isinstance(extras_raw, list):
        raise DatasetError(f"{example_id}: allowed_extra_topics must be a list")

    summaries_raw = obj.get("gold_topic_summaries") or {}
    if not isinstance(summaries_raw, dict):
        raise DatasetError(f"{example_id}: gold_topic_summaries must be an object")

    notes_val = obj.get("input_notes")
    if notes_val is not None and not isinstance(notes_val, str):
        raise DatasetError(f"{example_id}: input_notes must be a string or null")

    def _opt_topic_list(key: str) -> list[str] | None:
        if key not in obj or obj[key] is None:
            return None
        raw = obj[key]
        if not isinstance(raw, list):
            raise DatasetError(f"{example_id}: {key} must be a list")
        return [str(t).strip() for t in raw if str(t).strip()]

    required_topics = _opt_topic_list("required_topics")
    optional_topics = _opt_topic_list("optional_topics") or []
    aliases_alt = obj.get("aliases") or {}
    if not isinstance(aliases_alt, dict):
        raise DatasetError(f"{example_id}: aliases must be an object")
    for k, v in aliases_alt.items():
        topic_aliases.setdefault(str(k), [])
        for a in v or []:
            if str(a) not in topic_aliases[str(k)]:
                topic_aliases[str(k)].append(str(a))

    required_dependencies = None
    if obj.get("required_dependencies") is not None:
        required_dependencies = _parse_dependencies(obj["required_dependencies"], example_id)
        check_set = set(required_topics or gold_topics)
        for frm, to in required_dependencies:
            if frm not in check_set or to not in check_set:
                raise DatasetError(
                    f"{example_id}: required_dependencies {frm!r} -> {to!r} not in required/gold topics",
                )

    acceptable: list[tuple[str, str]] = []
    if obj.get("acceptable_dependencies"):
        acceptable = _parse_dependencies(obj["acceptable_dependencies"], example_id)

    return EvalExample(
        id=example_id,
        category=str(obj["category"]).strip(),
        difficulty=difficulty,  # type: ignore[arg-type]
        goal=str(obj["goal"]).strip(),
        gold_topics=gold_topics,
        gold_dependencies=gold_deps,
        input_notes=notes_val.strip() if isinstance(notes_val, str) and notes_val.strip() else None,
        notes=str(obj.get("notes") or "").strip(),
        topic_aliases=topic_aliases,
        allowed_extra_topics=[str(t).strip() for t in extras_raw if str(t).strip()],
        gold_topic_summaries={str(k): str(v) for k, v in summaries_raw.items()},
        required_topics=required_topics,
        optional_topics=optional_topics,
        required_dependencies=required_dependencies,
        acceptable_dependencies=acceptable,
        dataset_version=str(obj.get("dataset_version") or "learning_graph_eval_v1"),
    )


def load_dataset(path: str | Path | None = None) -> list[EvalExample]:
    target = Path(path) if path else DEFAULT_DATASET_PATH
    if not target.is_file():
        raise DatasetError(f"Dataset not found: {target}")

    examples: list[EvalExample] = []
    seen_ids: set[str] = set()
    try:
        text = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise DatasetError(f"Failed to read dataset: {exc}") from exc

    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"line {line_no}: malformed JSON ({exc})") from exc
        example = parse_example(obj, line_no=line_no)
        if example.id in seen_ids:
            raise DatasetError(f"duplicate example id: {example.id!r}")
        seen_ids.add(example.id)
        examples.append(example)

    if not examples:
        raise DatasetError(f"Dataset is empty: {target}")
    return examples


def write_dataset(examples: list[EvalExample], path: str | Path) -> None:
    from app.evaluation.schemas import example_to_dict

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(example_to_dict(ex), ensure_ascii=False) for ex in examples]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def filter_examples(
    examples: list[EvalExample],
    *,
    ids: set[str] | None = None,
    limit: int | None = None,
    categories: set[str] | None = None,
) -> list[EvalExample]:
    out = examples
    if ids is not None:
        out = [e for e in out if e.id in ids]
    if categories is not None:
        out = [e for e in out if e.category in categories]
    if limit is not None:
        out = out[: max(0, limit)]
    return out
