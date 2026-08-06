import logging

from fastapi import APIRouter, HTTPException, Query

from app.services.graph import build_dependency_graph, compute_prerequisite_chain
from app.services.topics import load_all_topics, load_dependencies

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/graph")
async def get_knowledge_graph():
    topics = load_all_topics()
    dependencies = load_dependencies()
    graph = build_dependency_graph(topics, dependencies)
    logger.info(
        "GET /graph served nodes=%s links=%s",
        len(graph["nodes"]),
        len(graph["links"]),
    )
    return graph


@router.get("/graph/path")
async def get_prerequisite_path(
    target: str = Query(..., description="Topic id to compute the prerequisite chain for"),
):
    """Ordered prerequisite chain leading to ``target`` (root topics first, target last)."""
    topics = load_all_topics()
    dependencies = load_dependencies()
    result = compute_prerequisite_chain(target, topics, dependencies)
    if result is None:
        raise HTTPException(status_code=404, detail=f"No topic with id {target!r}")
    logger.info(
        "GET /graph/path target=%s chain_len=%s edges=%s",
        target,
        len(result["chain"]),
        len(result["edges"]),
    )
    return {"target": target, **result}
