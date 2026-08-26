"""Load and validate versioned domain curriculum inventories."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.curriculum.text import normalize_topic

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CURRICULUM_DIR = _REPO_ROOT / "data" / "curriculum"

ALLOWED_LEVELS = frozenset({"foundational", "intermediate", "advanced", "overview"})


@dataclass(frozen=True)
class CurriculumConcept:
    id: str
    title: str
    description: str
    aliases: tuple[str, ...] = ()
    level: str = "intermediate"
    prerequisite_ids: tuple[str, ...] = ()


@dataclass
class DomainInventory:
    domain: str
    version: str
    path: Path
    concepts: list[CurriculumConcept] = field(default_factory=list)
    provenance: str = ""
    source: str = ""
    created_at: str = ""
    reviewed_at: str = ""
    review_status: str = ""
    raw: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""

    def by_id(self) -> dict[str, CurriculumConcept]:
        return {c.id: c for c in self.concepts}

    def by_title_norm(self) -> dict[str, CurriculumConcept]:
        out: dict[str, CurriculumConcept] = {}
        for c in self.concepts:
            out[normalize_topic(c.title)] = c
            for a in c.aliases:
                out[normalize_topic(a)] = c
        return out

    def titles(self) -> list[str]:
        return [c.title for c in self.concepts]

    def size(self) -> int:
        return len(self.concepts)

    def prerequisite_edge_count(self) -> int:
        return sum(len(c.prerequisite_ids) for c in self.concepts)

    def alias_count(self) -> int:
        return sum(len(c.aliases) for c in self.concepts)

    def inventory_version_label(self) -> str:
        """Stable label such as compiler_construction_v1."""
        ver = self.version if self.version.startswith("v") else f"v{self.version}"
        return f"{self.domain}_{ver}"


class InventoryValidationError(ValueError):
    """Raised when an inventory fails structural validation."""


def inventory_file_hash(path: str | Path) -> str:
    """SHA-256 of file bytes (stable for unchanged files)."""
    data = Path(path).read_bytes()
    return hashlib.sha256(data).hexdigest()


def inventory_dict_hash(data: dict[str, Any]) -> str:
    """SHA-256 of canonical JSON (sorted keys) for in-memory payloads."""
    raw = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _has_cycle(edges: list[tuple[str, str]]) -> bool:
    adj: dict[str, set[str]] = {}
    nodes: set[str] = set()
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        nodes.add(a)
        nodes.add(b)
    visiting: set[str] = set()
    done: set[str] = set()

    def dfs(n: str) -> bool:
        if n in done:
            return False
        if n in visiting:
            return True
        visiting.add(n)
        for nxt in adj.get(n, ()):
            if dfs(nxt):
                return True
        visiting.remove(n)
        done.add(n)
        return False

    return any(dfs(n) for n in nodes)


def _find_cycle_example(edges: list[tuple[str, str]]) -> list[str] | None:
    adj: dict[str, list[str]] = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
    visiting: dict[str, int] = {}
    stack: list[str] = []

    def dfs(n: str) -> list[str] | None:
        visiting[n] = 1
        stack.append(n)
        for nxt in adj.get(n, ()):
            state = visiting.get(nxt, 0)
            if state == 1:
                if nxt in stack:
                    i = stack.index(nxt)
                    return stack[i:] + [nxt]
                return [n, nxt, n]
            if state == 0:
                found = dfs(nxt)
                if found:
                    return found
        visiting[n] = 2
        stack.pop()
        return None

    for node in list(adj):
        if visiting.get(node, 0) == 0:
            found = dfs(node)
            if found:
                return found
    return None


def validate_inventory_dict(data: dict[str, Any], *, path: str | Path | None = None) -> list[str]:
    """Return a list of validation errors (empty if valid)."""
    loc = str(path) if path else "<inventory>"
    errors: list[str] = []
    if not isinstance(data, dict):
        return [f"{loc}: root must be an object"]
    for key in ("domain", "version", "concepts"):
        if key not in data:
            errors.append(f"{loc}: missing required field {key!r}")
    concepts = data.get("concepts")
    if not isinstance(concepts, list) or not concepts:
        errors.append(f"{loc}: concepts must be a non-empty list")
        return errors

    ids: set[str] = set()
    title_norms: dict[str, str] = {}
    alias_norms: dict[str, str] = {}
    edges: list[tuple[str, str]] = []
    edge_set: set[tuple[str, str]] = set()
    duplicate_edge_count = 0

    for i, row in enumerate(concepts):
        if not isinstance(row, dict):
            errors.append(f"{loc}: concepts[{i}] must be an object")
            continue
        cid = str(row.get("id") or "").strip()
        title = str(row.get("title") or "").strip()
        desc = str(row.get("description") or "").strip()
        level = str(row.get("level") or "intermediate").strip() or "intermediate"
        if not cid:
            errors.append(f"{loc}: concepts[{i}] missing id")
        elif cid in ids:
            errors.append(f"{loc}: duplicate concept id {cid!r}")
        else:
            ids.add(cid)
        if not title:
            errors.append(f"{loc}: concepts[{i}] empty title")
        else:
            tn = normalize_topic(title)
            if not tn:
                errors.append(f"{loc}: concepts[{i}] ({cid}) empty title after normalize")
            elif tn in title_norms:
                errors.append(
                    f"{loc}: duplicate canonical title {title!r} collides with {title_norms[tn]!r}"
                )
            else:
                title_norms[tn] = cid
        if not desc:
            errors.append(f"{loc}: concepts[{i}] ({cid}) empty description")
        if level not in ALLOWED_LEVELS:
            errors.append(
                f"{loc}: concepts[{i}] ({cid}) invalid level {level!r}; "
                f"allowed={sorted(ALLOWED_LEVELS)}"
            )
        seen_aliases_for_concept: set[str] = set()
        for alias in row.get("aliases") or []:
            an = normalize_topic(str(alias))
            if not an:
                errors.append(f"{loc}: concepts[{i}] ({cid}) empty alias")
                continue
            if an in seen_aliases_for_concept:
                errors.append(f"{loc}: concepts[{i}] ({cid}) duplicate alias {alias!r}")
                continue
            seen_aliases_for_concept.add(an)
            if an in alias_norms and alias_norms[an] != cid:
                errors.append(
                    f"{loc}: ambiguous alias {alias!r} maps to both "
                    f"{alias_norms[an]!r} and {cid!r}"
                )
            elif an in title_norms and title_norms[an] != cid:
                errors.append(
                    f"{loc}: alias {alias!r} collides with title of {title_norms[an]!r}"
                )
            else:
                alias_norms[an] = cid
        for pre in row.get("prerequisite_ids") or []:
            pre_s = str(pre).strip()
            if not pre_s:
                continue
            if pre_s == cid:
                errors.append(f"{loc}: self-loop prerequisite on {cid!r}")
            edge = (cid, pre_s)
            if edge in edge_set:
                duplicate_edge_count += 1
                errors.append(
                    f"{loc}: duplicate prerequisite edge {cid!r} → {pre_s!r}"
                )
            else:
                edge_set.add(edge)
                edges.append(edge)

    for a, b in edges:
        if b not in ids:
            errors.append(f"{loc}: unknown prerequisite id {b!r} referenced by {a!r}")
    if _has_cycle(edges):
        cycle = _find_cycle_example(edges)
        if cycle and len(cycle) >= 2:
            path_str = " → ".join(cycle)
            errors.append(
                f"{loc}: cycle in curriculum inventory: {path_str} "
                f"(Reason: cycle in curriculum inventory)"
            )
        else:
            errors.append(f"{loc}: cycle in curriculum inventory")
    # duplicate_edge_count retained for diagnostics callers via error list only
    _ = duplicate_edge_count
    return errors


def find_redundant_transitive_edges(edges: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Return direct edges A→C that are implied by a longer path A↝C (A≠C).

    Inventory graphs should prefer direct prerequisites and avoid transitively
    dense shortcuts (if A→B and B→C, do not also assert A→C unless independently direct).
    """
    adj: dict[str, set[str]] = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
    direct = set(edges)
    redundant: list[tuple[str, str]] = []

    def reaches_without_edge(src: str, dst: str, forbidden: tuple[str, str]) -> bool:
        stack = [src]
        seen: set[str] = set()
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            for nxt in adj.get(n, ()):
                if (n, nxt) == forbidden:
                    continue
                if nxt == dst:
                    return True
                stack.append(nxt)
        return False

    for a, c in sorted(direct):
        if a == c:
            continue
        # longer path exists if we ignore the direct edge
        if reaches_without_edge(a, c, (a, c)):
            redundant.append((a, c))
    return redundant


def inventory_degree_stats(inventory: DomainInventory) -> dict[str, int]:
    in_deg: dict[str, int] = {c.id: 0 for c in inventory.concepts}
    out_deg: dict[str, int] = {c.id: 0 for c in inventory.concepts}
    for c in inventory.concepts:
        out_deg[c.id] = len(c.prerequisite_ids)
        for pre in c.prerequisite_ids:
            in_deg[pre] = in_deg.get(pre, 0) + 1
    return {
        "max_in_degree": max(in_deg.values()) if in_deg else 0,
        "max_out_degree": max(out_deg.values()) if out_deg else 0,
    }


def inventory_health_report(inventory: DomainInventory) -> dict[str, Any]:
    """Product-neutral inventory health (no gold / eval imports)."""
    raw_errors = validate_inventory_dict(inventory.raw, path=inventory.path)
    edges = [
        (c.id, pre)
        for c in inventory.concepts
        for pre in c.prerequisite_ids
    ]
    degrees = inventory_degree_stats(inventory)
    transitive = find_redundant_transitive_edges(edges)
    return {
        "domain": inventory.domain,
        "version": inventory.version,
        "parent_version": str(inventory.raw.get("parent_version") or ""),
        "inventory_version": inventory.inventory_version_label(),
        "concept_count": inventory.size(),
        "alias_count": inventory.alias_count(),
        "prerequisite_edge_count": inventory.prerequisite_edge_count(),
        "cycle_count": 1 if any("cycle" in e for e in raw_errors) else 0,
        "duplicate_count": sum(1 for e in raw_errors if "duplicate" in e),
        "transitive_shortcut_count": len(transitive),
        "transitive_shortcuts": [{"from": a, "to": b} for a, b in transitive],
        "max_in_degree": degrees["max_in_degree"],
        "max_out_degree": degrees["max_out_degree"],
        "validation_status": "valid" if not raw_errors else "invalid",
        "validation_errors": raw_errors,
        "review_status": inventory.review_status or "unknown",
        "inventory_hash": inventory.content_hash,
        "source": inventory.source or inventory.provenance,
        "created_at": inventory.created_at,
        "reviewed_at": inventory.reviewed_at,
        "change_log_count": len(inventory.raw.get("change_log") or []),
    }


_INVENTORY_CACHE: dict[tuple[str, float, int], DomainInventory] = {}


def load_inventory(path: str | Path) -> DomainInventory:
    target = Path(path)
    stat = target.stat()
    cache_key = (str(target.resolve()), float(stat.st_mtime), int(stat.st_size))
    cached = _INVENTORY_CACHE.get(cache_key)
    if cached is not None:
        return cached
    data = json.loads(target.read_text(encoding="utf-8"))
    errs = validate_inventory_dict(data, path=target)
    if errs:
        raise InventoryValidationError("; ".join(errs))
    concepts = [
        CurriculumConcept(
            id=str(row["id"]).strip(),
            title=str(row["title"]).strip(),
            description=str(row.get("description") or "").strip(),
            aliases=tuple(str(a).strip() for a in (row.get("aliases") or []) if str(a).strip()),
            level=str(row.get("level") or "intermediate").strip(),
            prerequisite_ids=tuple(
                str(p).strip() for p in (row.get("prerequisite_ids") or []) if str(p).strip()
            ),
        )
        for row in data["concepts"]
    ]
    content_hash = inventory_file_hash(target)
    inv = DomainInventory(
        domain=str(data["domain"]).strip(),
        version=str(data["version"]).strip(),
        path=target,
        concepts=concepts,
        provenance=str(data.get("provenance") or data.get("source") or ""),
        source=str(data.get("source") or data.get("provenance") or ""),
        created_at=str(data.get("created_at") or ""),
        reviewed_at=str(data.get("reviewed_at") or ""),
        review_status=str(data.get("review_status") or "reviewed"),
        raw=data,
        content_hash=content_hash,
    )
    _INVENTORY_CACHE[cache_key] = inv
    return inv


def clear_inventory_cache() -> None:
    """Test helper — drop cached inventory loads."""
    _INVENTORY_CACHE.clear()


def load_experiment_config(curriculum_dir: str | Path | None = None) -> dict[str, Any]:
    root = Path(curriculum_dir) if curriculum_dir else DEFAULT_CURRICULUM_DIR
    cfg_path = root / "experiment_config_v1.json"
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def load_case_domain_map(curriculum_dir: str | Path | None = None) -> dict[str, str]:
    root = Path(curriculum_dir) if curriculum_dir else DEFAULT_CURRICULUM_DIR
    data = json.loads((root / "case_domain_map_v1.json").read_text(encoding="utf-8"))
    mapping = data.get("case_to_domain") or {}
    return {str(k): str(v) for k, v in mapping.items()}


def resolve_domain_for_case(
    case_id: str,
    *,
    domain_override: str | None = None,
    curriculum_dir: str | Path | None = None,
) -> str:
    """Backward-compatible resolver used by evaluation adapters.

    Raises ``ValueError`` with ``DOMAIN_UNRESOLVED`` / ``DOMAIN_PRIOR_UNAVAILABLE``
    when the domain prior path cannot run.
    """
    from app.curriculum.resolution import resolve_domain

    resolution = resolve_domain(
        domain_override=domain_override,
        case_id=case_id,
        curriculum_dir=curriculum_dir,
        require_inventory=True,
        on_unresolved="error",
        on_unavailable="error",
    )
    if resolution.status == "RESOLVED" and resolution.domain:
        return resolution.domain
    reason = resolution.fallback_reason or resolution.status
    detail = f"{reason}: no domain mapping for case {case_id!r}"
    if resolution.status == "DOMAIN_PRIOR_UNAVAILABLE":
        detail = f"{reason}: inventory missing for domain {resolution.domain!r}"
    raise ValueError(detail)


def load_domain_inventory(
    domain: str,
    *,
    curriculum_dir: str | Path | None = None,
) -> DomainInventory:
    root = Path(curriculum_dir) if curriculum_dir else DEFAULT_CURRICULUM_DIR
    cfg = load_experiment_config(root)
    files = cfg.get("inventory_files") or {}
    if domain not in files:
        raise ValueError(f"DOMAIN_PRIOR_UNAVAILABLE: Unknown domain {domain!r}; known={sorted(files)}")
    path = root / files[domain]
    if not path.is_file():
        raise ValueError(f"DOMAIN_PRIOR_UNAVAILABLE: inventory file missing for {domain!r}: {path}")
    return load_inventory(path)


def list_inventory_paths(curriculum_dir: str | Path | None = None) -> list[Path]:
    root = Path(curriculum_dir) if curriculum_dir else DEFAULT_CURRICULUM_DIR
    return sorted(
        p
        for p in root.glob("*_v*.json")
        if p.name
        not in {
            "case_domain_map_v1.json",
            "experiment_config_v1.json",
            "domain_prioritization_v1.json",
        }
        and not p.name.startswith("case_domain_map")
        and not p.name.startswith("experiment_config")
        and not p.name.startswith("domain_prioritization")
    )


def inventory_matches_title(inventory: DomainInventory, title: str) -> CurriculumConcept | None:
    return inventory.by_title_norm().get(normalize_topic(title))
