"""
Read-only MCP bridge over Synapse's dependency graph -- lets external agents (Claude
Desktop/Code, Cursor) query the current graph and study progress without going through
the HTTP API. Reads the same topics/ + topics/_dependencies.json files the FastAPI
backend uses, so it reflects live state whether or not `uvicorn app.main:app` is running.

Run directly (stdio transport, what Claude Desktop/Cursor expect):
    cd backend && python -m app.mcp_server

Claude Desktop / Cursor config (point "cwd" at this repo's backend/ directory):
    {
      "mcpServers": {
        "synapse": {
          "command": "python",
          "args": ["-m", "app.mcp_server"],
          "cwd": "/absolute/path/to/wiki-llm/backend"
        }
      }
    }
"""

from mcp.server import MCPServer

from app.services.graph import build_dependency_graph, compute_prerequisite_chain
from app.services.topics import get_topic_by_id, load_all_topics, load_dependencies

mcp = MCPServer("synapse", version="0.1.0")


def _topic_public(row: dict) -> dict:
    return {k: v for k, v in row.items() if k != "path"}


@mcp.tool()
def get_graph() -> dict:
    """Return the full dependency graph: one node per topic, directed edges for
    "from requires to" prerequisite relationships."""
    topics = load_all_topics()
    dependencies = load_dependencies()
    return build_dependency_graph(topics, dependencies)


@mcp.tool()
def get_path(target: str) -> dict:
    """Return the ordered prerequisite chain leading to the topic id `target`
    (root topics first, `target` itself last). A root topic with no prerequisites
    returns a chain containing only itself."""
    topics = load_all_topics()
    dependencies = load_dependencies()
    result = compute_prerequisite_chain(target, topics, dependencies)
    if result is None:
        return {"error": f"No topic with id {target!r}"}
    return {"target": target, **result}


@mcp.tool()
def get_topic(topic_id: str) -> dict:
    """Return full detail for one topic by id: title, summary, status, resources,
    and whether its current closure quiz has been passed."""
    row = get_topic_by_id(topic_id)
    if row is None:
        return {"error": f"No topic with id {topic_id!r}"}
    return _topic_public(row)


@mcp.tool()
def get_progress() -> dict:
    """Return a study-progress summary: topic count by status, overall percent
    complete, and a compact per-topic status list."""
    topics = load_all_topics()
    counts = {"not_started": 0, "in_progress": 0, "complete": 0}
    items = []
    for row in topics:
        status = str(row.get("status", "not_started"))
        counts[status] = counts.get(status, 0) + 1
        items.append(
            {
                "id": row.get("id"),
                "title": row.get("title"),
                "status": status,
                "quiz_passed": bool(row.get("quiz_passed", False)),
            },
        )
    total = len(topics)
    percent_complete = round((counts["complete"] / total * 100), 1) if total else 0.0
    return {
        "total_topics": total,
        "status_counts": counts,
        "percent_complete": percent_complete,
        "topics": items,
    }


if __name__ == "__main__":
    mcp.run()
