# Synapse — Feature Documentation

This document covers every feature currently shipped in Synapse: what it is, why it exists
(the actual reasoning from the build, not a generic justification), and how to use it —
both via the API and through the UI. It describes the app as built, not the original plan;
anything that didn't ship isn't in here.

For setup instructions, see [README.md](README.md).

---

## 1. Topic/dependency graph model

**What it is.** The core data model: a `Topic` (a title, a summary, a status, attached
resources) and a `Dependency` — a *directed* edge meaning "this topic requires that one
first." The graph they form must always be a DAG (directed acyclic graph): no topic may,
even transitively, depend on itself.

**Why it was added.** This is the foundational pivot of the whole project — Synapse
started as a tag-linked wiki where a "graph" was just an undirected projection of shared
tags and loose `related_topics` references. That kind of link only captures *association*
("these two pages mention similar things"), not *learning order*. A directed prerequisite
edge captures something a tag never could: "you cannot understand B without A first." That
distinction is what makes click-to-path navigation, closure quizzes, and status tracking
meaningful at all — none of them make sense on an undirected association graph.

The DAG-cycle check exists because a prerequisite graph with a cycle is a contradiction: if
A requires B and B (transitively) requires A, neither can ever be validly "the first thing
to learn." Every dependency-adding code path — the manual `POST /dependencies` endpoint,
every AI operation mode, and even topic **merges** — is validated against this invariant
before it's allowed to write. Merges get a mathematical shortcut rather than a redundant
runtime check: collapsing two nodes of an already-acyclic graph can only create a cycle if
a path already existed in *both* directions between them, which a DAG cannot have — so a
merge is provably always safe once you know the input was acyclic.

**How to use it.**

- `POST /topics` — create a topic (`title`, optional `summary`, `status`). Returns 201 with
  the created `Topic`.
- `POST /dependencies` — `{"from_topic_id": "...", "to_topic_id": "..."}`, meaning
  `from` requires `to`. Returns 201 with the created edge, or **409** if it would close a
  cycle (including a direct self-loop).
- `GET /topics`, `GET /topics/{id}`, `GET /dependencies` — read back the raw graph.
- `PATCH /topics/{id}` — partial update (`status`, `zone_id`); see the Study loop and
  Zones sections below for what those fields mean.
- **UI:** the main canvas (`KnowledgeGraph.tsx`) renders every topic as a node and every
  dependency as a directed edge, force-laid-out with `react-force-graph-2d`. Node color
  reflects status (see Study loop). There's no manual "add dependency" button in the UI
  yet — edges are created by the AI operation modes or directly via the API.

---

## 2. Click-to-path navigation

**What it is.** Click any topic and the panel shows — and the graph visually highlights —
the full ordered chain of prerequisites leading to it: every topic you'd need to complete
first, in dependency order, plus the edges connecting them.

**Why it was added.** A directed graph is only useful for *navigation* if you can actually
ask "what do I need before this?" and get a real answer, not just see a tangle of arrows.
This is the feature that turns "directed edges exist" into "the tool tells you your study
order." It's computed as a DFS post-order traversal over the "from requires to" edges,
walked backward from the target — post-order naturally yields prerequisites first and the
target last, so the response is already in the order you'd actually study them.

**How to use it.**

- `GET /graph/path?target={topic_id}` → `{"target", "chain": [{"id","title","status"}, ...],
  "edges": [{"source","target"}, ...]}`. `chain` is a topic-picked-first-to-last-in-that-order
  list; a root topic with no prerequisites returns a chain of just itself.
- **UI:** click any node on the graph, or pick a topic from the "Jump to concept…" search bar
  or the sidebar's "Recent topics" list. The selected topic's full prerequisite chain lights
  up on the canvas (a distinct highlight color for the path's nodes and edges), and the
  right-hand "Connections" panel opens showing that topic's detail.

---

## 3. Generative roadmap creation (ingest)

**What it is.** Turn a learning goal, a flat list of topic names, or previously-uploaded raw
notes into a proposed set of topics and prerequisite edges, via an LLM call. This is one of
the four AI operation modes (see the next section) — it's the only one that starts from
outside the graph rather than modifying something already in it.

**Why it was added.** Building a prerequisite graph by hand, topic by topic, doesn't scale
past a handful of nodes — this is the on-ramp that lets a whole subject area (e.g. "learn
how transformer-based language models work, from the ground up") turn into a structured,
review-ready graph in one call, instead of dozens of manual `POST /topics` +
`POST /dependencies` calls.

**How to use it.**

- Optionally upload source material first: `POST /ingest/upload` (single file) or
  `POST /ingest/upload/batch` (up to 30) — accepts `.txt`, `.md`, `.pdf`, `.docx`; each is
  parsed to plain text and saved as a raw note. `POST /ingest` also accepts pasted raw text
  directly. All three return the note's basename, which `filenames` (below) references.
- `POST /ai/ingest` — body is `{"goal": "...", "topics": ["...", ...], "filenames": ["..."]}`
  (any combination; at least one non-empty). Optional product fields:
  `generation_strategy` (`baseline` default, or opt-in `domain_curriculum_prior` /
  experimental `domain_prior_edge_classifier`), `curriculum_domain`, `require_domain_prior`.
  Returns a **pending Proposal** — see the next section for what happens next. Nothing is
  written to the graph by this call alone. Closed experiments (Concept-First, coverage
  recovery) are evaluation-only and are rejected by this endpoint.
- **UI:** the sidebar's "+ AI operations" button opens a modal with mode tabs; the "Ingest"
  tab has a single free-text box for the goal (file/topic-dump inputs are reachable via the
  same endpoint but the current UI's ingest tab is goal-text-only).

---

## 4. Reviewable AI changes: ingest / expand / audit / reshape, and apply/rollback

**What it is.** All four AI operation modes, and the mechanism that governs every one of
them: an AI call never mutates the graph directly. It always produces a `Proposal` —
topics/dependencies/removals/merges/edits, each with a confidence score — that a human
reviews and explicitly applies. Applying is one atomic transaction; a snapshot is taken
first so the whole apply can be rolled back as a single unit.

**Why it was added.** This is the single most load-bearing design decision in the app. An
LLM will occasionally propose a wrong topic, a backwards edge, or a hallucinated merge —
letting that write directly to a shared graph with no review step means one bad call can
silently corrupt structure that took real effort to build. So every AI call is split into
two phases that can never be collapsed into one: *propose* (an LLM call plus validation,
producing a `Proposal` row) and *apply* (a plain database transaction with no LLM
involvement, triggered only by an explicit human action). Apply is atomic — new topics,
new dependencies, removals, edits, and merges all happen in one transaction — so a partial
failure can't leave the graph half-mutated. And because an apply can create many topics and
edges at once, "undo" needs to restore the *entire* database to a point in time, not just
the last write; that's why rollback works off a whole-database snapshot (via SQLite's own
online backup API) taken immediately before each apply, rather than tracking undo at the
level of individual operations.

The four modes exist as **separate** operations, not one generic "improve the graph"
button, because each has a genuinely different risk profile:

- **Ingest** — new content in, new topics + dependencies out. The only mode that starts
  from raw external input.
- **Expand** — deepens *one* existing topic with new sub-topics/prerequisites beneath it,
  without touching the rest of the graph. Scoped by construction so a request about one
  corner of the graph can't accidentally restructure an unrelated part of it.
- **Audit** — read-only. Structural checks (orphaned topics, duplicate titles, thin
  summaries) run for free with no LLM call; a second pass asks an LLM to judge missing
  prerequisites and cycle-risk relationships. It returns a diagnostic `AuditReport`
  directly, never a `Proposal` — there is nothing to apply, by design, so audit is safe to
  run at any time with zero chance of it mutating anything, even indirectly. If the LLM
  pass fails, the API does not disguise a structural-only result as a full semantic audit:
  `status` becomes `"partial"` and `semantic_analysis` is `"unavailable"`.
- **Reshape** — the most invasive mode: split/merge/reorder an existing subgraph you
  select. It's the only mode that can propose *removing* an edge, *merging* two topics, or
  *editing* an existing summary — capabilities the other three deliberately don't have,
  because ingest/expand only ever add.

**How to use it.**

- `POST /ai/ingest` — see section 3.
- `POST /ai/expand` — `{"topic_id": "...", "instructions": "optional free text"}`.
- `POST /ai/reshape` — `{"topic_ids": ["...", ...], "instructions": "optional free text"}`
  (the selected subgraph to restructure).
- `POST /ai/audit` — no body. Returns `AuditReport` directly (`total_topics`, `findings`:
  each `{"type", "topic_ids", "detail"}` where `type` is one of `orphaned_topic`,
  `duplicate_title`, `thin_topic`, `missing_prerequisite`, `cycle_risk`). Structural
  checks always run. If the LLM semantic pass fails, the report is explicit about
  degraded mode: `status` is `"partial"`, `semantic_analysis` is `"unavailable"`,
  `structural_findings` lists the deterministic issues, and `findings` does **not**
  pretend to include semantic analysis. A fully successful pass uses `status: "ok"`
  and `semantic_analysis: "available"` (including when the model reports no semantic issues).
- For ingest/expand/reshape, the response is a `Proposal`: `id`, `mode`, `source`, `topics`
  (each with a `confidence` score and `needs_review` flag for anything at or below the
  configured threshold), `dependencies`, and for reshape also `removed_dependencies`,
  `merges`, `edits` — plus `skipped_dependencies` for any proposed edge that would have
  closed a cycle or referenced an unknown topic.
- `POST /apply` — `{"proposal_id": "..."}`. Commits the proposal atomically; returns
  `ApplyResponse` with everything actually created/removed/merged/edited, `snapshot_id`,
  and any dependency adds skipped at commit time (a real per-edge cycle/uniqueness failure,
  not a bug — the rest of the apply still commits).
- `POST /discard` — `{"proposal_id": "..."}`. Marks it discarded; nothing is written.
- `POST /rollback` — `{"snapshot_id": "..."}` (optional; defaults to the most recent).
  Restores the whole database to that snapshot.
- **UI:** "+ AI operations" opens the modal with five tabs (Ingest / Expand / Audit /
  Reshape / Obsidian — the last covered in section 9). Expand lets you pick a topic from a
  dropdown; Reshape lets you check off multiple topics. Audit's tab runs immediately and
  renders findings inline (no apply/discard — there's nothing to commit). Ingest/Expand
  /Reshape show the resulting proposal — topics with confidence badges, dependencies,
  removals, merges, edits, skipped items — with **Apply** and **Discard** buttons. The
  sidebar's "Undo last change" button calls `POST /rollback` with no target (most recent).

---

## 5. Study loop: status, resources, quizzes

**What it is.** Per-topic status tracking (`not_started` / `in_progress` / `complete`),
attached resources (links, documents, or previously-ingested notes — things you studied
*from*), and generated closure quizzes that can gate marking a topic complete.

**Why it was added.** A dependency graph on its own is just a map — this is what turns it
into something you actually *use* to study: a visible record of where you are, material to
learn from per topic, and a check that you actually understood a topic rather than just
clicking a button. The quiz gate specifically exists so "complete" can mean something more
than a self-report: if a topic has a generated quiz, `PATCH .../status: complete` is
refused (409) until `quiz_passed` is true for that quiz. The gate is scoped per-topic (only
kicks in if a quiz already exists for it) and is a deliberate config toggle, not a hard
requirement — some topics genuinely don't need a quiz to be "done."

**How to use it.**

- `PATCH /topics/{id}` — `{"status": "in_progress"}` etc. Returns 409 with a clear message
  if the quiz gate blocks a `complete` transition.
- `POST /topics/{id}/resources` — `{"type": "link"|"document"|"note", "source_ref": "...",
  "title": "optional"}`. `type: "note"` must reference an already-ingested raw note
  basename (422 otherwise) — resources aren't free-text, they trace back to real material.
- `POST /topics/{id}/quiz` — generates a multiple-choice quiz from the topic's summary +
  resources; returns `QuizPublic` (questions + choices, **no** correct answers).
- `POST /topics/{id}/quiz/submit` — `{"answers": {"question_id": choice_index, ...}}`.
  Returns `QuizResult` (per-question correctness, `score`, `passed`) and sets
  `quiz_passed` on the topic if passed.
- **UI:** the topic detail panel ("Connections") has a **Status** section (three toggle
  buttons), a **Resources** section (list + an attach form with type/source/title), and a
  **Closure quiz** section — "Generate quiz" builds one, then renders each question as a
  radio-button list with a Submit button; results highlight correct/incorrect choices
  inline.

---

## 6. Zones and artifacts

**What it is.** Two independent additions to a topic:

- **Zones** — an optional, non-overlapping visual/logical grouping region. A topic belongs
  to at most one zone (`Topic.zone_id`) at a time.
- **Artifacts** — something a learner *produced* while studying a topic (a note, code
  snippet, summary, or generated output) — explicitly distinct from a Resource, which is
  something they studied *from* (an input).

**Why it was added.** As graphs grow past a couple dozen nodes, an unstructured
force-directed layout stops being legible — zones give a way to visually cluster related
topics (e.g. "Math foundations" vs. "Model architecture") without changing the underlying
dependency structure at all. Artifacts exist because Resources and "things you made" are
genuinely different kinds of data with different shapes and different growth patterns
(artifact content can be arbitrarily long, e.g. a full note), so they're kept out of a
`Topic`'s own serialization (unlike Resources) and fetched via dedicated endpoints instead
— that separation is also why zone membership is exclusive rather than many-to-many: it's
what makes rendering an unambiguous convex-hull region around a zone's members possible in
the first place.

**How to use it.**

- `POST /zones` — `{"label": "...", "color": "optional #hex"}`. `GET /zones`,
  `PATCH /zones/{id}`, `DELETE /zones/{id}` (deleting a zone unassigns, never deletes, its
  member topics).
- Assign/unassign a topic to a zone via `PATCH /topics/{id}` with `{"zone_id": "..."}` (or
  `null` to unassign) — there's no separate zone-side assignment endpoint.
- `GET /topics/{id}/artifacts`, `POST /topics/{id}/artifacts` —
  `{"type": "note"|"code_snippet"|"summary"|"generated_output", "title": "optional",
  "content": "..."}`.
- **UI:** the sidebar has a **Zones** section (list with color swatches, plus an inline
  create form). The topic detail panel has a **Zone** dropdown (includes "No zone") and an
  **Artifacts** section (list + create form) below Resources. On the graph canvas, each
  zone renders as a soft translucent region behind its member nodes — a circle for one
  node, a rounded capsule for two, a padded and smoothed convex hull for three or more —
  drawn on a canvas pre-render hook so it sits behind the nodes/edges themselves.

---

## 7. In-session assistant (ask)

**What it is.** A free-text Q&A assistant scoped to whichever topic you're currently
viewing. Answers are grounded only in that topic's own summary, attached resources, and
previously-produced artifacts — never the rest of the graph.

**Why it was added.** This is deliberately a *different* kind of AI feature from the four
operation modes: read/explain-only, with no path to a graph mutation at all, even if asked.
It exists for the moments where you want to ask "wait, why does this need that?" without
leaving the topic you're looking at, and without that question turning into (or being
confused with) a request to change the graph — if you ask it to add a topic, it explicitly
declines and points you at the AI operations panel instead. Each exchange is persisted as a
`qa_log`-type Artifact (reusing the Artifacts machinery from section 6) so the
conversation survives a page reload as part of that topic's study log, rather than being
purely ephemeral.

**How to use it.**

- `POST /topics/{id}/ask` — `{"question": "..."}`. Returns `{"answer": "...",
  "artifact_id": "..."}`. Whether prior turns for that topic are included as context
  depends on the **memory** setting (section 10) — off by default means each question is
  answered fresh.
- **UI:** the topic detail panel's **"Ask about this topic"** section shows prior turns
  (reconstructed from the persisted `qa_log` artifacts) as a simple chat history, with a
  textarea and Ask button below.

---

## 8. MCP bridge

**What it is.** A read-only [MCP](https://modelcontextprotocol.io) server exposing the
dependency graph and study progress to external agents (Claude Desktop, Claude Code,
Cursor), separate from the HTTP API.

**Why it was added.** So an AI coding/chat agent working outside the app entirely can still
answer questions like "what am I supposed to study next" or "what does this topic depend
on" by querying live state directly, without needing to go through (or even know about) the
FastAPI server. It's deliberately **read-only** — it exposes graph and progress queries,
never a mutation path — for the same reason ingest/expand/reshape are gated behind explicit
apply: an external agent should never be able to silently change your graph. It reads the
same SQLite file the main backend writes to, so it reflects live state whether or not
`uvicorn` is running.

**How to use it.**

- Run it directly: `cd backend && python -m app.mcp_server` (stdio transport).
- Point Claude Desktop/Code or Cursor at it with `cwd` set to `backend/`:
  ```json
  { "mcpServers": { "synapse": {
      "command": "python", "args": ["-m", "app.mcp_server"],
      "cwd": "/absolute/path/to/wiki-llm/backend"
  } } }
  ```
- Exposes tools built on the same graph/topic services the HTTP API uses — fetching the
  full dependency graph, resolving a topic's prerequisite chain, and summarizing study
  progress (topic/status counts, percent complete) — all read-only.

---

## 9. Obsidian import/export bridge, and the packaged import skill

**What it is.** A two-way bridge to [Obsidian](https://obsidian.md) vaults: **import**
parses a folder of `.md` notes (with `[[wikilinks]]`) into a reviewable proposal of topics
and dependencies; **export** walks the live graph (or one topic's prerequisite subgraph)
back out to a wikilinked `.md` folder. A packaged Claude Code skill
(`.claude/skills/obsidian-to-synapse-import`) documents how to audit a vault before
importing it.

**Why it was added.** Vaults are a common place people already keep notes, and Obsidian
wikilinks are a natural, low-effort *hint* of a relationship between two notes — but they're
undirected and untyped, so "note A links to note B" doesn't by itself tell you whether B is
a genuine prerequisite of A or just a loose cross-reference. That's a judgment call, so
import runs through the same LLM-review pipeline as ingest mode (a dedicated prompt asks
the model to keep or skip each note as a topic, and to judge each wikilink as a real
prerequisite or not) rather than blindly turning every link into an edge — and it produces
a `Proposal` like any other ingest call, so nothing is written until it's applied, same as
everywhere else in the app. Export exists as the natural inverse: reconstructing wikilinks
from real `Dependency` edges means the exported folder is immediately usable as a vault
again, not just a one-way dump.

**How to use it.**

- `POST /obsidian/import` — `{"vault_path": "/absolute/path/to/vault"}`. Returns a
  `Proposal` (mode `ingest`) — review and `POST /apply` it like any other. 404 if the path
  isn't a directory, 422 if it has no `.md` files. Topics that map 1:1 to a source note get
  a `Resource` pointing back to that file automatically once applied.
- `GET /obsidian/export?scope={topic_id}` — omit `scope` for the whole graph, or pass a
  topic id to export just its prerequisite subgraph. Returns a downloadable `.zip` of
  `.md` files (`application/zip`).
- The skill (`obsidian-to-synapse-import`) is invoked by name in Claude Code; it walks
  through auditing a vault (note count, wikilink density, likely-skipped stub/journal
  notes), decides what's a hard blocker vs. just worth flagging, then hands off to the
  import endpoint and the same review-before-apply flow above.
- **UI:** the AI operations modal's **Obsidian** tab takes a vault path and calls import,
  reusing the exact same proposal-review-and-apply UI as ingest/expand/reshape rather than
  a separate screen. The sidebar's **"Export to Obsidian vault"** link downloads the
  whole-graph export directly.

---

## 10. Multi-provider LLM support, and memory/persona/thinking settings

**What it is.** Two related pieces:

- **Provider abstraction** — every LLM call in the app goes through one function
  (`call_llm`), which delegates to whichever provider is configured: OpenAI, Google
  Gemini, or a generic OpenAI-compatible endpoint (self-hosted or third-party), selected
  via one `.env` variable.
- **Workspace settings** — a single-row settings object (persona, memory, thinking) that's
  applied to LLM calls without any per-call-site changes.

**Why they were added.** The provider abstraction exists so switching models — for cost,
quality, or self-hosting reasons — is a config change, not a code change: every ingest
/expand/audit/reshape/quiz/ask call already went through one function before this, so
provider selection could be added at that single choke point instead of touching ten call
sites individually. Settings piggyback on exactly the same choke point:

- **Persona** is a free-text instruction appended (not prepended) to every prompt, so it
  layers a tone/style preference on top of a call's own output-format rules without
  overriding them — an ingest call's "respond with ONLY valid JSON" instruction stays first
  and freshest in context either way.
- **Memory** controls whether the in-session assistant (section 7) includes prior Q&A turns
  for a topic as context on the next question, or answers each question fresh. It's a
  toggle rather than always-on because "does the assistant remember what I just asked" is a
  real behavioral difference worth being able to turn off, not just an implementation
  detail.
- **Thinking** ("extended") appends an instruction asking the model to reason step-by-step
  before answering. It's implemented as a prompt-level nudge, applied uniformly across
  every provider and model, rather than a provider-specific reasoning-effort parameter —
  the default model (`gpt-4o-mini`) has no native reasoning-effort API, so a fragile
  per-model special case would only work for some configurations; a plain prompt
  instruction works the same way regardless of which provider is active.

**Operation correlation (`operation_id`).** Each user-facing AI action (ingest, expand,
audit, reshape, Obsidian import, ask, quiz generation) begins a logical operation with a
single `operation_id` (UUID hex). That id is attached to:

- every LLM usage line in `backend/data/llm_usage.jsonl` (when logging is enabled)
- `Proposal.generation_meta` (`operation_id` plus `llm_calls[]` summaries)
- proposal lifecycle events (`proposal_created`, `proposal_applied`, `proposal_discarded`, and rollback when deterministically known)

`operation_id` is **not** a distributed trace id, HTTP request id, or graph-row provenance
marker — it only correlates one logical AI workflow with its LLM calls, resulting proposal,
and apply/discard/rollback outcomes where those relationships can be established honestly.

**How to use it.**

- Set `LLM_PROVIDER` in `.env` to `openai` (default), `gemini`, or `openai_compatible`,
  plus that provider's key/model vars (see `.env.example`) — restart the backend to pick up
  the change. No code changes needed for any of the three.
- `GET /settings` / `PATCH /settings` — `{"persona": "...", "memory_enabled": bool,
  "thinking_level": "standard"|"extended"}` (partial updates supported).
- **UI:** the sidebar's **Settings** button opens a panel with a persona textarea, a
  memory checkbox, and a thinking dropdown, with Save/Close actions.

---

## 11. SQLite persistence (implementation detail)

The backend persists everything — topics, dependencies, resources, artifacts, zones,
proposals, quizzes, settings — in a single SQLite database (`backend/data/synapse.db`) via
SQLAlchemy, rather than flat JSON files. This exists mainly so `POST /apply` can be a real
atomic database transaction and `POST /rollback` can be a real point-in-time snapshot
(via SQLite's own online backup API) instead of a file-copy scheme. It's not a feature you
interact with directly — every endpoint above already covers everything it backs.
