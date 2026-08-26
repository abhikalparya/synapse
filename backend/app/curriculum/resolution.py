"""Domain resolution and domain-prior availability contract.

Resolution order:
1. Explicit ``domain`` / ``curriculum_domain`` argument
2. Case → domain map (``case_domain_map_v1.json``)
3. Optional deterministic category routing via map metadata (when provided)

States:
- ``RESOLVED`` — domain known and inventory file configured
- ``DOMAIN_UNRESOLVED`` — no domain could be determined
- ``DOMAIN_PRIOR_UNAVAILABLE`` — domain known but inventory missing/unconfigured
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from app.curriculum.inventory import (
    DEFAULT_CURRICULUM_DIR,
    load_case_domain_map,
    load_experiment_config,
)

DomainResolutionStatus = Literal[
    "RESOLVED",
    "DOMAIN_UNRESOLVED",
    "DOMAIN_PRIOR_UNAVAILABLE",
]

FallbackAction = Literal["baseline", "error"]


@dataclass(frozen=True)
class DomainResolution:
    status: DomainResolutionStatus
    domain: str | None = None
    inventory_file: str | None = None
    fallback_action: FallbackAction = "baseline"
    fallback_reason: str | None = None
    source: str = ""  # explicit | case_map | category | none

    @property
    def ok(self) -> bool:
        return self.status == "RESOLVED"


def _category_domain_hints(curriculum_dir: Path) -> dict[str, str]:
    """Optional deterministic category → domain routing from case map metadata."""
    path = curriculum_dir / "case_domain_map_v1.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    hints = data.get("category_to_domain") or {}
    return {str(k).strip(): str(v).strip() for k, v in hints.items() if str(k).strip() and str(v).strip()}


def inventory_configured(domain: str, *, curriculum_dir: str | Path | None = None) -> bool:
    root = Path(curriculum_dir) if curriculum_dir else DEFAULT_CURRICULUM_DIR
    cfg = load_experiment_config(root)
    files = cfg.get("inventory_files") or {}
    rel = files.get(domain)
    if not rel:
        return False
    return (root / rel).is_file()


def resolve_domain(
    *,
    domain_override: str | None = None,
    case_id: str | None = None,
    category: str | None = None,
    curriculum_dir: str | Path | None = None,
    require_inventory: bool = True,
    on_unresolved: FallbackAction | None = None,
    on_unavailable: FallbackAction | None = None,
) -> DomainResolution:
    """Resolve curriculum domain for an operation.

    ``require_inventory``: when True, a mapped domain without an inventory file is
    ``DOMAIN_PRIOR_UNAVAILABLE`` rather than ``RESOLVED``.
    """
    root = Path(curriculum_dir) if curriculum_dir else DEFAULT_CURRICULUM_DIR
    cfg = load_experiment_config(root)
    fb = cfg.get("fallback") or {}
    unresolved_action: FallbackAction = on_unresolved or fb.get("on_unresolved_domain") or "baseline"
    unavailable_action: FallbackAction = on_unavailable or fb.get("on_missing_inventory") or "baseline"
    if unresolved_action not in ("baseline", "error"):
        unresolved_action = "baseline"
    if unavailable_action not in ("baseline", "error"):
        unavailable_action = "baseline"

    domain: str | None = None
    source = "none"
    if domain_override and str(domain_override).strip():
        domain = str(domain_override).strip()
        source = "explicit"
    elif case_id:
        mapping = load_case_domain_map(root)
        mapped = mapping.get(case_id)
        if mapped:
            domain = mapped
            source = "case_map"
    if domain is None and category and str(category).strip():
        hints = _category_domain_hints(root)
        hinted = hints.get(str(category).strip())
        if hinted:
            domain = hinted
            source = "category"

    if not domain:
        return DomainResolution(
            status="DOMAIN_UNRESOLVED",
            fallback_action=unresolved_action,
            fallback_reason="DOMAIN_UNRESOLVED",
            source=source,
        )

    files = (cfg.get("inventory_files") or {})
    inv_file = files.get(domain)
    path_ok = bool(inv_file) and (root / str(inv_file)).is_file()
    if require_inventory and not path_ok:
        return DomainResolution(
            status="DOMAIN_PRIOR_UNAVAILABLE",
            domain=domain,
            inventory_file=str(inv_file) if inv_file else None,
            fallback_action=unavailable_action,
            fallback_reason="DOMAIN_PRIOR_UNAVAILABLE",
            source=source,
        )

    return DomainResolution(
        status="RESOLVED",
        domain=domain,
        inventory_file=str(inv_file) if inv_file else None,
        source=source,
    )


def resolution_to_meta(resolution: DomainResolution) -> dict[str, Any]:
    return {
        "domain_resolution_status": resolution.status,
        "domain": resolution.domain,
        "inventory_file": resolution.inventory_file,
        "fallback_action": resolution.fallback_action,
        "fallback_reason": resolution.fallback_reason,
        "domain_resolution_source": resolution.source,
    }
