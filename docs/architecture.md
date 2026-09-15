# Synapse architecture map

Learning artifact, not permanent documentation. Written at Step 0 of the working brief,
before any code changes, so I have a concrete mental model of the two paths that matter
most: how an AI-generated proposal comes into existence, and how a frontend action gets
data onto the screen.

## 1. End-to-end trace: `POST /ai/ingest`

Goal / topic-dump / raw notes in, a pending `Proposal` out. Nothing is written to the
graph on this path -- that only happens later, via `POST /apply`.

1. **Route** -- [`backend/app/routes/ai.py:16`](../backend/app/routes/ai.py#L16)
   `ingest()` takes the request body (`IngestRequest`: goal, topics, filenames,
   generation_strategy, curriculum_domain, require_domain_prior) and calls
   `run_ingest(...)`, translating `ValueError` -> 422 and `RuntimeError` -> 502.

2. **Service entry** -- `run_ingest()` in
   [`backend/app/services/ingest.py:70`](../backend/app/services/ingest.py#L70)
   - Builds one combined prompt-source string from goal + topic dump + resolved raw
     notes (`_build_source_text`).
   - Opens `synapse_operation()` (see step 3), which mints an `operation_id` for
     everything downstream to correlate against.
   - Resolves the generation strategy via `resolve_runtime_generation_strategy()` in
     [`generation_strategy.py`](../backend/app/services/generation_strategy.py) --
     production default is `"baseline"`; `domain_curriculum_prior` and
     `domain_prior_edge_classifier` are opt-in and fall back to baseline if their
     required domain inventory isn't available. Everything below follows the
     `baseline` branch, `_run_ingest_baseline()`, since that's what a plain ingest
     call takes and what Step 1-7 work will actually exercise.

3. **Operation correlation** --
   [`backend/app/services/operation_context.py`](../backend/app/services/operation_context.py)
   `synapse_operation()` is a context manager that sets a `ContextVar` operation id and
   an empty list to collect LLM call summaries into. `finalize_generation_meta()` later
   reads both back out and merges them into the proposal's `generation_meta`. This is
   how Step 4 (run metadata) will join a proposal back to its LLM usage log lines.

4. **Prompt construction** --
   [`backend/app/prompts/ingest.py`](../backend/app/prompts/ingest.py)
   `build_ingest_prompt(source_text, known_topic_titles)` assembles the fixed
   `INGEST_JSON_SCHEMA` instruction body (there's also an experimental
   `concept_direct_prerequisite` variant, opt-in only) plus a list of existing topic
   titles the model should avoid duplicating, plus the actual source text.

5. **LLM call** -- `call_llm_detailed()` in
   [`backend/app/services/llm.py:229`](../backend/app/services/llm.py#L229), wrapped in
   `llm_operation("ingest")` which just labels the call for the usage log. This is the
   single choke point every LLM call in the app goes through (ingest / expand / audit /
   reshape / quiz generation / ask). It resolves the configured provider
   (`openai` / `gemini` / `openai_compatible`, see `_build_provider()`), applies
   persona/thinking settings, times the call, and returns an `LLMCallRecord` (text,
   latency, token counts, estimated cost, provider/model, success/error). Every call is
   appended to `llm_usage.jsonl` and, if inside a `synapse_operation()`, attached to that
   operation's running summary list -- this is the data Step 4 needs.

6. **Response parsing** -- `parse_llm_json_object()` in
   [`backend/app/services/proposal_common.py:40`](../backend/app/services/proposal_common.py#L40)
   strips markdown code fences if present and requires the result to be a JSON object.
   `run_ingest` then requires a non-empty `topics` list; an empty/malformed `dependencies`
   list is tolerated (defaults to `[]`), a malformed `topics` list is not (raises
   `ValueError` -> 422).

7. **Deterministic validation / graph construction** --
   `build_topics_and_dependencies()`, same file,
   [line 48](../backend/app/services/proposal_common.py#L48). This is the authoritative
   boundary (see section 4): it never trusts the LLM's structure at face value.
   - Every topic title is canonicalized (`canonical_topic_title()` in
     [`topic_identity.py`](../backend/app/services/topic_identity.py)) and matched
     against existing graph topics -- a title that already exists resolves to the real
     topic id instead of becoming a new proposed topic.
   - Confidence is clamped to `[0, 1]`; anything at or below
     `review_confidence_threshold()` (env `ROADMAP_REVIEW_CONFIDENCE_THRESHOLD`, default
     0.6) is flagged `needs_review` rather than rejected.
   - Dependencies referencing an unknown title, a self-loop, an exact duplicate, or an
     edge that would create a cycle (checked against every other edge accepted so far in
     this same call, via `would_create_cycle()` in
     [`topics.py`](../backend/app/services/topics.py)) are diverted into
     `skipped_dependencies` with a reason string instead of silently dropped or
     force-applied.

8. **Proposal assembly + persistence** -- back in `_run_ingest_baseline()`
   ([ingest.py:122](../backend/app/services/ingest.py#L122)):
   `finalize_generation_meta()` attaches `operation_id` and the collected `llm_calls`
   summaries to `generation_meta`; a `Proposal` (status `"pending"`, mode `"ingest"`) is
   built and handed to `save_proposal()` in
   [`backend/app/services/proposals.py:62`](../backend/app/services/proposals.py#L62),
   which upserts a `ProposalRow` -- topics/dependencies/etc. are stored as JSON columns,
   not normalized tables (see `db/models.py`'s module docstring: a proposal is only ever
   read/written as a whole object, keyed by ephemeral temp ids meaningful only within
   that one proposal).

9. **Event log** -- `log_proposal_created()` in
   [`proposal_events.py`](../backend/app/services/proposal_events.py) records the
   lifecycle event, then `run_ingest` returns the `Proposal` back through the route as
   the JSON response.

**Apply, separately** -- `POST /apply` (`routes/proposals.py`) -> `apply_proposal()` in
[`services/proposals.py:132`](../backend/app/services/proposals.py#L132) is the *only*
place a graph mutation can originate from. It snapshots the whole DB first
(`snapshot_graph()` in [`snapshots.py`](../backend/app/services/snapshots.py), a live
SQLite backup-API copy, not a raw file copy), then inside one transaction: creates
topics, resolves temp-ids to real ids, adds dependencies (a per-edge cycle/uniqueness
failure here is caught and reported, not fatal), then removals/edits/merges in that
order. If anything in the whole-graph-invariant path (removals/edits/merges) fails, the
whole apply rolls back -- those operations only ever reference topics the proposal
author already confirmed existed at build time, so a failure there means real
inconsistency, not an expected outcome. `POST /rollback` (`restore_snapshot()`) reverts
to a snapshot wholesale, which is coarser than a single apply's transaction rollback and
is the only way to undo an apply that already committed.

## 2. End-to-end trace: a frontend action

Traced the ingest flow in `AiOperationsModal`, since it's the one that actually calls a
route that exists (unlike `IngestModal`, see Step 1 write-up below) and mirrors the
backend trace above exactly.

1. **Component state** --
   [`frontend/src/components/AiOperationsModal.tsx`](../frontend/src/components/AiOperationsModal.tsx)
   holds `goal` (textarea), `busy`, `error`, and `proposal` as local `useState`. Submitting
   the ingest form calls `handleIngest()` ([line 88](../frontend/src/components/AiOperationsModal.tsx#L88)).

2. **`fetchJson`** -- a small local helper (re-implemented per-component; see section 3
   for why that's worth noticing) at
   [line 25](../frontend/src/components/AiOperationsModal.tsx#L25): plain `fetch()`,
   reads the body as text first so a non-OK response's JSON `detail` field (FastAPI's
   `HTTPException` shape) can be surfaced as the thrown `Error`'s message, otherwise
   falls back to raw text or `res.statusText`.

3. **The call** -- `fetchJson<Proposal>("/ai/ingest", { method: "POST", body:
   JSON.stringify({ goal: trimmed }) })`. This lands on the exact route traced in
   section 1. The frontend's `Proposal` type
   ([`frontend/src/types.ts`](../frontend/src/types.ts)) is expected to mirror the
   backend's Pydantic `Proposal` model field-for-field -- there's no schema generation
   tying them together, so they can drift silently.

4. **State update** -- on success, `setProposal(result)` swaps the modal from the
   ingest form into `<ProposalDetails proposal={proposal} ... onApply={...}
   onDiscard={...} />` ([line 470](../frontend/src/components/AiOperationsModal.tsx#L470)).
   `ProposalDetails` is the one shared diff renderer for every proposal-producing mode
   (ingest/expand/reshape/obsidian) -- this is the component Step 3 (Activity view) is
   told to reuse rather than building a second one.

5. **Apply, and the bubble-up to page state** -- clicking Apply in `ProposalDetails`
   calls `handleApply()` ([line 183](../frontend/src/components/AiOperationsModal.tsx#L183)),
   which POSTs `/apply` and then calls the `onApplied` prop passed down from
   [`App.tsx`](../frontend/src/App.tsx). `App.tsx` wires that to `handleApplied()`
   ([line 261](../frontend/src/App.tsx#L261)), which calls `refreshPersistedState(true)`
   -- a `Promise.all` over `refreshGraph`, `refreshStats`, `refreshDependencies`,
   `refreshProposals`, each hitting its own `GET` route through the same
   per-component `fetchJson` pattern and landing in its own top-level `useState`
   (`graphData`, `stats`, `dependencies`, `pendingProposals`). A toast is set from the
   `ApplyResponse` counts. This is the general shape every mutating action in the app
   follows: local component state for the in-flight operation, then a refetch of the
   relevant top-level state in `App.tsx` rather than optimistic local patching.

## 3. Files that matter most

| File | Why it matters |
|---|---|
| [`backend/app/services/llm.py`](../backend/app/services/llm.py) | Single choke point for every LLM call in the app; owns provider selection, usage logging, and cost/latency instrumentation. |
| [`backend/app/services/proposal_common.py`](../backend/app/services/proposal_common.py) | The deterministic validation layer between "what the model said" and "what becomes a Proposal" -- confidence gating, cycle checks, identity resolution. |
| [`backend/app/services/proposals.py`](../backend/app/services/proposals.py) | Owns proposal persistence and is the *only* code path that can turn a proposal into real graph writes (`apply_proposal`). |
| [`backend/app/services/ingest.py`](../backend/app/services/ingest.py) | The one AI mode that starts from raw external input rather than existing graph state; also the clearest example of the strategy-routing pattern shared with expand/reshape. |
| [`backend/app/services/operation_context.py`](../backend/app/services/operation_context.py) | Defines `operation_id` correlation, the mechanism Step 4 depends on to join usage data back to a proposal. |
| [`backend/app/services/generation_strategy.py`](../backend/app/services/generation_strategy.py) | Central strategy allow-list; explicitly fences off closed/evaluation-only experiments from the product API. |
| [`backend/app/services/topics.py`](../backend/app/services/topics.py) | Owns the actual graph mutation primitives (`_create_topic_in_session`, cycle detection) that every apply path calls into. |
| [`backend/app/services/snapshots.py`](../backend/app/services/snapshots.py) | Whole-DB backup/restore backing rollback; distinct mechanism from apply's own transaction. |
| [`backend/app/db/models.py`](../backend/app/db/models.py) | Full schema in one file; the module docstring explains the JSON-column-vs-normalized-table tradeoff that shows up repeatedly. |
| [`backend/app/models/proposal.py`](../backend/app/models/proposal.py) | The Proposal shape itself -- what's reviewable, what's authoritative, what's still just observability (`generation_meta`). |
| [`frontend/src/App.tsx`](../frontend/src/App.tsx) | Owns all top-level persisted state and the refetch-after-mutation pattern every other component relies on. |
| [`frontend/src/components/AiOperationsModal.tsx`](../frontend/src/components/AiOperationsModal.tsx) | The actual, working proposal-producing UI (ingest/expand/reshape/obsidian/audit) -- as opposed to `IngestModal`. |
| [`frontend/src/components/ProposalDetails.tsx`](../frontend/src/components/ProposalDetails.tsx) | The one shared diff renderer for proposals; Step 3 must reuse this, not rebuild it. |
| [`frontend/src/components/IngestModal.tsx`](../frontend/src/components/IngestModal.tsx) | Calls `/generate` and `/generate/from-raw`, which do not exist on the backend -- dead path, subject of Step 1. |
| [`frontend/src/types.ts`](../frontend/src/types.ts) | Hand-maintained mirror of the backend Pydantic models; no generation step keeps them in sync. |

## 4. Where the AI boundary sits

**Can call a model:** `call_llm` / `call_llm_detailed`
([`services/llm.py`](../backend/app/services/llm.py)) is the only function that talks to
a provider, and it's reached exclusively from the four AI operation services
(`ingest.py`, `expand.py`, `audit.py`, `reshape.py`) plus quiz generation and `ask`. All
of those are read/propose-only or, for quiz/ask, don't mutate the graph at all.

**Authoritative regardless of what the model returns:**
- `build_topics_and_dependencies()` / `proposal_common.py` -- confidence clamping, DAG
  cycle checks, and canonical-title deduplication happen in plain Python and cannot be
  overridden by anything the model outputs. An LLM claiming a topic has confidence 5.0
  gets clamped to 1.0; an LLM proposing a cycle gets that edge silently diverted to
  `skipped_dependencies`, never inserted.
- `apply_proposal()` -- the model's output never reaches the graph directly. It only
  ever produces a `Proposal` row; a human (or an explicit `POST /apply` call) is a
  separate, later action, and `apply_proposal` re-derives real ids and re-runs its own
  cycle/uniqueness checks at apply time rather than trusting what was true when the
  proposal was built.
- Everything in `generation_meta` (strategy, domain, `llm_calls` summaries, prompt
  version) is explicitly documented as **not used for graph mutation** -- see the
  `Proposal.generation_meta` field description in
  [`models/proposal.py`](../backend/app/models/proposal.py). It's observability data
  only.

**Not authoritative / advisory only:** `confidence` and `needs_review` are the model's
own self-assessment, surfaced to a human reviewer -- nothing in the pipeline auto-rejects
a low-confidence topic; it's still proposed, just flagged.
