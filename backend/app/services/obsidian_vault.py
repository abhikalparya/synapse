"""Parsing for Obsidian vaults: a folder of ``.md`` files with optional YAML-ish
frontmatter and ``[[wikilink]]``-style cross-references. No YAML dependency -- vault
frontmatter here is treated as flat ``key: value`` lines, which covers the common case
(tags, title, aliases as a simple list) without pulling in a parser for a format we only
ever read a couple of fields from.
"""

import re
from dataclasses import dataclass
from pathlib import Path

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]")


@dataclass
class VaultNote:
    relative_path: str
    title: str
    frontmatter: dict[str, str]
    body: str
    links: list[str]


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    fields: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().strip("\"'")
        value = value.strip().strip("\"'")
        if key:
            fields[key] = value
    return fields, text[match.end() :]


def extract_wikilinks(body: str) -> list[str]:
    seen: dict[str, None] = {}
    for m in _WIKILINK_RE.finditer(body):
        target = m.group(1).strip()
        if target:
            seen.setdefault(target, None)
    return list(seen.keys())


def load_vault(vault_path: Path) -> list[VaultNote]:
    """Recursively load every ``.md`` file under ``vault_path``. Note title is the
    frontmatter ``title`` field if present, otherwise the filename stem."""
    notes: list[VaultNote] = []
    for path in sorted(vault_path.rglob("*.md")):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        frontmatter, body = _parse_frontmatter(text)
        title = frontmatter.get("title", "").strip() or path.stem
        notes.append(
            VaultNote(
                relative_path=str(path.relative_to(vault_path)),
                title=title,
                frontmatter=frontmatter,
                body=body.strip(),
                links=extract_wikilinks(body),
            ),
        )
    return notes
