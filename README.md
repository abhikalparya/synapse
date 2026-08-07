# Synapse

A directed dependency-graph learning tool: break a subject into topics, connect them
with prerequisite edges, and study through the graph instead of a linear course.

## What it does

- **Topic/dependency graph** — topics are nodes; a dependency edge means "this topic
  requires that one first." The graph enforces a DAG (no cycles) and renders as a
  force-directed layout with click-to-path prerequisite highlighting.
- **AI operations, always reviewable** — four modes (`ingest` a goal/notes into new
  topics, `expand` one topic into sub-topics, `audit` the graph for structural issues,
  `reshape` a subgraph via merge/split/reorder) all produce a pending Proposal. Nothing
  touches the graph until you explicitly apply it; every apply can be rolled back.
- **Study loop** — per-topic status tracking, attached resources, and closure quizzes
  that gate marking a topic complete.
- **Zones & artifacts** — optional visual grouping regions for topics, and a place to
  keep what you *produce* while studying (notes, snippets, summaries) separate from what
  you *studied from*.
- **In-session assistant** — ask a question scoped to the topic you're viewing; answers
  are grounded in that topic's own summary/resources, never the whole graph, and never
  propose a graph change themselves.
- **Obsidian bridge** — import a vault (`.md` notes + `[[wikilinks]]`) into a reviewable
  proposal, or export the graph (or a subgraph) back out as wikilinked `.md` files.
- **Multi-provider LLM** — OpenAI, Gemini, or any OpenAI-compatible endpoint, selected
  via one `.env` variable, no code changes.
- **Workspace settings** — a persona prefix, a memory toggle for the assistant's
  conversation history, and an extended-thinking nudge, applied to every LLM call.
- **MCP bridge** — a read-only MCP server so external agents (Claude Desktop/Code,
  Cursor) can query the graph and study progress directly.

## Stack

- **Backend:** FastAPI, SQLAlchemy + SQLite, Pydantic
- **Frontend:** React, TypeScript, Vite, react-force-graph-2d

## Running it

```bash
# Backend
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp ../.env.example .env   # set OPENAI_API_KEY (or another provider, see .env.example)
uvicorn app.main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend
npm install
npm run dev   # http://localhost:5173, proxies API calls to :8000
```

### MCP bridge (optional)

```bash
cd backend && python -m app.mcp_server
```

Point Claude Desktop/Code or Cursor at it with `cwd` set to `backend/`:

```json
{
  "mcpServers": {
    "synapse": {
      "command": "python",
      "args": ["-m", "app.mcp_server"],
      "cwd": "/absolute/path/to/wiki-llm/backend"
    }
  }
}
```

## Layout

```
backend/app/
  routes/     FastAPI endpoints
  services/   business logic (one per concern: topics, proposals, obsidian, llm, ...)
  models/     Pydantic request/response schemas
  db/         SQLAlchemy models + session
  prompts/    LLM prompt builders
frontend/src/
  components/ KnowledgeGraph, NodeDetailsPanel, AiOperationsModal, SettingsPanel, ...
```
