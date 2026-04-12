from collections import defaultdict
from pathlib import Path
from typing import Any

from app.services.tags import normalize_tags_list


def _norm_str_list(raw: Any) -> list[str]:
    if not raw:
        return []
    out: list[str] = []
    for x in raw:
        s = str(x).strip()
        if s:
            out.append(s)
    return out


def _link_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _cluster_group_labels(
    titles: list[str],
    merged: dict[str, dict[str, Any]],
) -> dict[str, str]:
    """
    Group pages by shared-tag connectivity (transitive). Each component gets a
    short ``group`` label: smallest normalized tag in the union, else smallest title.
    """
    parent = {t: t for t in titles}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    n = len(titles)
    for i in range(n):
        ta = titles[i]
        sa = merged[ta]["tag_set"]
        if not sa:
            continue
        for j in range(i + 1, n):
            tb = titles[j]
            if not sa.isdisjoint(merged[tb]["tag_set"]):
                union(ta, tb)

    members: dict[str, list[str]] = defaultdict(list)
    for t in titles:
        members[find(t)].append(t)

    labels: dict[str, str] = {}
    for _root, group in members.items():
        tag_union: set[str] = set()
        for m in group:
            tag_union |= merged[m]["tag_set"]
        if tag_union:
            label = min(tag_union)
        else:
            label = min(group)
        for m in group:
            labels[m] = label
    return labels


def build_knowledge_graph(pages: list[dict[str, Any]]) -> dict[str, list[dict[str, str]]]:
    """
    Build graph-ready structures from wiki page dicts (each may include ``path``).

    Nodes: one per unique page title. ``group`` is a shared-tag cluster label
    (smallest tag in the connected component, or the page title if untagged).

    Links: undirected unique pairs — from ``related_topics`` when the target is
    another node title, and between pages that share at least one tag.
    """
    merged: dict[str, dict[str, Any]] = {}
    for page in sorted(pages, key=lambda p: str(p.get("path", ""))):
        path = page.get("path")
        stem = Path(path).stem if isinstance(path, Path) else ""
        title = str(page.get("title", "")).strip() or stem or "untitled"
        tags = normalize_tags_list(page.get("tags"))
        related = _norm_str_list(page.get("related_topics"))

        summary = str(page.get("summary", "")).strip()
        key_points = _norm_str_list(page.get("key_points"))
        source_notes = _norm_str_list(page.get("source_notes"))

        mf = _norm_str_list(page.get("merged_from"))

        if title not in merged:
            merged[title] = {
                "tags": list(tags),
                "tag_set": set(tags),
                "related": list(related),
                "summary": summary,
                "key_points": list(key_points),
                "source_notes": list(source_notes),
                "merged_from": list(mf),
            }
            continue

        m = merged[title]
        for t in tags:
            if t not in m["tag_set"]:
                m["tag_set"].add(t)
                m["tags"].append(t)
        seen_r = set(m["related"])
        for r in related:
            if r not in seen_r:
                seen_r.add(r)
                m["related"].append(r)
        if len(summary) > len(str(m.get("summary", ""))):
            m["summary"] = summary
        seen_kp = set(m["key_points"])
        for kp in key_points:
            if kp not in seen_kp:
                seen_kp.add(kp)
                m["key_points"].append(kp)
        seen_sn = set(m["source_notes"])
        for sn in source_notes:
            if sn not in seen_sn:
                seen_sn.add(sn)
                m["source_notes"].append(sn)
        seen_mf = {x.casefold() for x in m["merged_from"]}
        for x in mf:
            if x.casefold() not in seen_mf:
                seen_mf.add(x.casefold())
                m["merged_from"].append(x)

    node_ids = set(merged.keys())
    titles = sorted(node_ids)
    cluster_labels = _cluster_group_labels(titles, merged)
    nodes = [
        {
            "id": tid,
            "group": cluster_labels[tid],
            "title": tid,
            "summary": str(merged[tid].get("summary", "")),
            "key_points": list(merged[tid].get("key_points", [])),
            "tags": list(merged[tid].get("tags", [])),
            "source_notes": list(merged[tid].get("source_notes", [])),
            "merged_from": list(merged[tid].get("merged_from", [])),
        }
        for tid in titles
    ]

    seen_edges: set[tuple[str, str]] = set()
    links: list[dict[str, str]] = []

    def add_edge(s: str, t: str) -> None:
        if s == t:
            return
        if s not in node_ids or t not in node_ids:
            return
        k = _link_key(s, t)
        if k in seen_edges:
            return
        seen_edges.add(k)
        links.append({"source": k[0], "target": k[1]})

    for title, m in merged.items():
        for rt in m["related"]:
            add_edge(title, rt)

    for i, ta in enumerate(titles):
        sa = merged[ta]["tag_set"]
        if not sa:
            continue
        for tb in titles[i + 1 :]:
            if sa.isdisjoint(merged[tb]["tag_set"]):
                continue
            add_edge(ta, tb)

    return {"nodes": nodes, "links": links}
