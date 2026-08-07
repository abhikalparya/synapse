---
name: obsidian-to-synapse-import
description: Audits an Obsidian vault for importability into Synapse's dependency graph and runs the import via POST /obsidian/import. Use when the user wants to bring an Obsidian vault into Synapse, or asks whether a vault is ready to import.
---

# Obsidian → Synapse import

Synapse turns a folder of Obsidian notes into a prerequisite dependency graph: each kept
note becomes a `Topic`, and `[[wikilinks]]` between notes are judged by an LLM call and
turned into `Dependency` edges where they represent a genuine prerequisite relationship
(not just a loose association). The import always produces a reviewable `Proposal` --
nothing is written to the graph until that proposal is applied.

Work through these steps in order.

## 1. Audit the vault before importing

Given a vault path, inspect it and report back to the user before calling the import
endpoint:

- Count `.md` files (`find <vault_path> -name '*.md' | wc -l`). Zero means there's
  nothing to import -- stop and tell the user.
- Sample a handful of notes and check wikilink density: `grep -l '\[\[' <vault_path>/**/*.md`
  vs. the total file count. A vault where most notes have zero `[[wikilinks]]` will still
  import (each note can still become a standalone topic), but the resulting graph will
  have few/no dependency edges -- flag this to the user as an expectation-setter, not a
  blocker.
- Look for very short or empty notes (pure stubs, daily-journal entries, or
  table-of-contents/MOC notes that just list links). These aren't a hard blocker --
  the import prompt already asks the LLM to skip non-substantive notes -- but call out
  roughly how many you see so the user isn't surprised if the resulting topic count is
  much smaller than the file count.
- Check for non-markdown content (images, PDFs, `.canvas` files) the user might expect
  to carry over. The importer only reads `.md` files; anything else is silently ignored.

## 2. What makes a vault "importable"

- At least one `.md` file with real prose content (not just frontmatter/links).
- The vault path must be a real, readable directory on the machine running the backend
  (the endpoint takes a filesystem path, not an upload).

There's no minimum wikilink density or vault size requirement -- a vault with zero
internal links still imports fine, it just won't produce any dependency edges on its own.

## 3. Flag blockers before handing off

Stop and tell the user (don't call the import endpoint) if:

- The path doesn't exist or isn't a directory.
- There are zero `.md` files under it.

Otherwise, summarize what you found (note count, rough wikilink density, any stub/MOC
notes you're expecting to get skipped) and confirm with the user before importing,
since the import call makes a real LLM request.

## 4. Run the import

```bash
curl -s -X POST http://127.0.0.1:8000/obsidian/import \
  -H "Content-Type: application/json" \
  -d '{"vault_path": "/absolute/path/to/vault"}'
```

This returns a `Proposal` (`status: "pending"`) with `topics`, `dependencies`, and
`skipped_dependencies` (wikilinks the model judged as not genuine prerequisites, with a
reason). Topics that map 1:1 to a source note carry that note's path forward, so once
applied they'll have a `Resource` pointing back to the original file.

## 5. Review and apply

Do not apply automatically. Show the user the proposed topics/dependencies (and what got
skipped and why), and let them either apply it themselves in the app's AI operations /
proposals review UI, or explicitly ask you to apply it:

```bash
curl -s -X POST http://127.0.0.1:8000/apply \
  -H "Content-Type: application/json" \
  -d '{"proposal_id": "<id from the import response>"}'
```

If the user instead wants to discard it, `POST /discard` with the same body shape.
