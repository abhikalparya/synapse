import logging

from fastapi import APIRouter

from app.services.graph import build_dependency_graph
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
