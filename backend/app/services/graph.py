from typing import Any


def build_dependency_graph(
    topics: list[dict[str, Any]],
    dependencies: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """
    Build graph-ready structures from Topic + Dependency records.

    Nodes: one per topic; ``group`` mirrors ``status`` for node coloring.

    Links: directed edges from ``Dependency.from_topic_id`` to ``Dependency.to_topic_id``
    ("from requires to"). Edges referencing an unknown topic id are dropped.
    """
    nodes: list[dict[str, Any]] = []
    node_ids: set[str] = set()
    for row in topics:
        tid = str(row.get("id", "")).strip()
        if not tid or tid in node_ids:
            continue
        node_ids.add(tid)
        nodes.append(
            {
                "id": tid,
                "group": str(row.get("status", "not_started")),
                "title": str(row.get("title", "")).strip() or tid,
                "summary": str(row.get("summary", "")).strip(),
                "status": str(row.get("status", "not_started")),
                "resources": row.get("resources") or [],
            },
        )

    links: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str]] = set()
    for dep in dependencies:
        s = str(dep.get("from_topic_id", "")).strip()
        t = str(dep.get("to_topic_id", "")).strip()
        if not s or not t or s == t:
            continue
        if s not in node_ids or t not in node_ids:
            continue
        key = (s, t)
        if key in seen_edges:
            continue
        seen_edges.add(key)
        links.append({"source": s, "target": t})

    return {"nodes": nodes, "links": links}
