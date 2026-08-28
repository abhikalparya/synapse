"""Phase 11: Obsidian vault import/export bridge.

Import parses a vault into a reviewable Proposal via the same LLM-review pipeline as
ingest mode (Phase 8) -- nothing is written to the graph until that proposal is applied.
Export walks the live graph (or a prerequisite subgraph) back out to an Obsidian-ready
folder of ``.md`` files with wikilinks reconstructed from Dependency edges.
"""

import io
import logging
import re
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from app.models.proposal import Proposal
from app.prompts.obsidian import build_obsidian_import_prompt
from app.services.graph import compute_prerequisite_chain
from app.services.llm import call_llm_detailed, llm_operation
from app.services.obsidian_vault import load_vault
from app.services.operation_context import finalize_generation_meta, synapse_operation
from app.services.proposal_common import build_topics_and_dependencies, parse_llm_json_object, review_confidence_threshold
from app.services.proposal_events import log_proposal_created
from app.services.proposals import save_proposal
from app.services.topics import load_all_topics, load_dependencies

logger = logging.getLogger(__name__)


class VaultNotFoundError(ValueError):
    """Raised when ``vault_path`` doesn't resolve to an existing directory."""


def _resolve_vault_dir(vault_path: str) -> Path:
    p = Path(vault_path).expanduser().resolve()
    if not p.is_dir():
        raise VaultNotFoundError(f"No such directory: {vault_path!r}")
    return p


async def import_vault(vault_path: str) -> Proposal:
    """Parse the vault at ``vault_path``, run it through an ingest-style LLM review call,
    and persist (but do not apply) the resulting Proposal. Raises VaultNotFoundError if
    the path isn't a directory, or ValueError if it contains no markdown notes."""
    vault_dir = _resolve_vault_dir(vault_path)
    notes = load_vault(vault_dir)
    if not notes:
        raise ValueError(f"No .md files found under {vault_path!r}")

    existing_topics = load_all_topics()
    known_titles = sorted({str(r.get("title", "")).strip() for r in existing_topics if r.get("title")})
    prompt = build_obsidian_import_prompt(
        [(n.title, n.body, n.links) for n in notes],
        known_topic_titles=known_titles,
    )

    with synapse_operation():
        with llm_operation("obsidian_import"):
            record = await call_llm_detailed(prompt)
        data = parse_llm_json_object(record.text)

        raw_topics = data.get("topics")
        raw_deps = data.get("dependencies")
        if not isinstance(raw_topics, list):
            raw_topics = []
        if not isinstance(raw_deps, list):
            raw_deps = []

        proposed_topics, proposed_dependencies, skipped_dependencies = build_topics_and_dependencies(
            raw_topics,
            raw_deps,
            confidence_threshold=review_confidence_threshold(),
            existing_topics=existing_topics,
        )

        # Best-effort: trace each kept topic back to its source note when the LLM used the
        # note's exact title (the prompt asks for this on 1:1 mappings) -- topics that were
        # merged from multiple notes, or renamed, simply get no source_note_path, which is a
        # harmless degradation (no Resource gets attached for those on apply).
        notes_by_title = {n.title.casefold(): n for n in notes}
        for pt in proposed_topics:
            note = notes_by_title.get(pt.title.casefold())
            if note is not None:
                pt.source_note_path = note.relative_path

        meta = finalize_generation_meta({"generation_strategy": "obsidian_import"})
        proposal = Proposal(
            id=uuid.uuid4().hex,
            status="pending",
            mode="ingest",
            source=f"Obsidian vault import: {vault_dir}",
            topics=proposed_topics,
            dependencies=proposed_dependencies,
            skipped_dependencies=skipped_dependencies,
            generation_meta=meta,
            created_at=datetime.now(timezone.utc),
        )
        save_proposal(proposal)
        log_proposal_created(proposal)

    logger.info(
        "Obsidian import proposal %s built from %s notes: topics=%s dependencies=%s skipped=%s",
        proposal.id,
        len(notes),
        len(proposed_topics),
        len(proposed_dependencies),
        len(skipped_dependencies),
    )
    return proposal


_ILLEGAL_FILENAME_CHARS = re.compile(r'[\\/:*?"<>|]')


def _safe_note_filename(title: str, used: set[str]) -> str:
    base = _ILLEGAL_FILENAME_CHARS.sub("", title).strip() or "untitled"
    candidate = f"{base}.md"
    n = 2
    while candidate.casefold() in used:
        candidate = f"{base} ({n}).md"
        n += 1
    used.add(candidate.casefold())
    return candidate


def export_vault_zip(scope: str | None = None) -> bytes:
    """Build an Obsidian-ready vault (as an in-memory zip) from the live graph, or from
    ``scope``'s prerequisite subgraph if a topic id is given. Each topic becomes one
    ``.md`` file; dependency edges become ``[[wikilink]]`` bullets under "Prerequisites"
    so the exported folder is directly usable as a vault. Raises LookupError if ``scope``
    doesn't match a known topic."""
    all_topics = load_all_topics()
    all_deps = load_dependencies()

    if scope:
        subgraph = compute_prerequisite_chain(scope, all_topics, all_deps)
        if subgraph is None:
            raise LookupError(f"No topic with id {scope!r}")
        topic_ids = {c["id"] for c in subgraph["chain"]}
        topics = [t for t in all_topics if t["id"] in topic_ids]
    else:
        topics = all_topics

    topic_by_id = {t["id"]: t for t in topics}
    prereqs_by_topic: dict[str, list[str]] = {t["id"]: [] for t in topics}
    for d in all_deps:
        f, to = d["from_topic_id"], d["to_topic_id"]
        if f in topic_by_id and to in topic_by_id:
            prereqs_by_topic[f].append(to)

    used_filenames: set[str] = set()
    filename_by_id = {t["id"]: _safe_note_filename(t["title"] or t["id"], used_filenames) for t in topics}

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for t in topics:
            lines = [f"---\nstatus: {t['status']}\n---\n", t["summary"] or "", ""]
            if t["resources"]:
                lines.append("## Resources")
                for r in t["resources"]:
                    lines.append(f"- [{r['type']}] {r['title'] or r['source_ref']} ({r['source_ref']})")
                lines.append("")
            prereq_ids = prereqs_by_topic.get(t["id"], [])
            if prereq_ids:
                lines.append("## Prerequisites")
                for pid in prereq_ids:
                    prereq_title = topic_by_id[pid]["title"] or pid
                    lines.append(f"- [[{prereq_title}]]")
                lines.append("")
            zf.writestr(filename_by_id[t["id"]], "\n".join(lines))
    return buf.getvalue()
