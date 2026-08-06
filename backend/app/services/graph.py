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
                "quiz_passed": bool(row.get("quiz_passed", False)),
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


def _build_forward_adjacency(dependencies: list[dict[str, Any]]) -> dict[str, list[str]]:
    """``from_topic_id -> [to_topic_id, ...]`` ("from requires to")."""
    adj: dict[str, list[str]] = {}
    for d in dependencies:
        f = str(d.get("from_topic_id", "")).strip()
        t = str(d.get("to_topic_id", "")).strip()
        if f and t:
            adj.setdefault(f, []).append(t)
    return adj


def compute_prerequisite_chain(
    target_id: str,
    topics: list[dict[str, Any]],
    dependencies: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """
    Ordered prerequisite chain ending at ``target_id``: DFS post-order over the
    "from requires to" edges reachable backward from the target. Because edges point
    dependent -> prerequisite, post-order naturally yields prerequisites first and the
    target last (a root topic with no prerequisites yields a chain of just itself).
    Returns None if ``target_id`` is not a known topic.
    """
    topic_by_id = {str(t.get("id", "")): t for t in topics if t.get("id")}
    if target_id not in topic_by_id:
        return None

    adj = _build_forward_adjacency(dependencies)
    visited: set[str] = set()
    order: list[str] = []

    def dfs(node: str) -> None:
        if node in visited or node not in topic_by_id:
            return
        visited.add(node)
        for nxt in adj.get(node, []):
            dfs(nxt)
        order.append(node)

    dfs(target_id)
    ancestor_ids = set(order)

    edges = [
        {"source": s, "target": t}
        for s, tos in adj.items()
        if s in ancestor_ids
        for t in tos
        if t in ancestor_ids
    ]

    chain = [
        {
            "id": tid,
            "title": str(topic_by_id[tid].get("title", "")).strip() or tid,
            "status": str(topic_by_id[tid].get("status", "not_started")),
        }
        for tid in order
    ]
    return {"chain": chain, "edges": edges}
