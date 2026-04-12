import logging

from fastapi import APIRouter

from app.services.graph import build_knowledge_graph
from app.services.wiki import load_all_wiki_pages

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/graph")
async def get_knowledge_graph():
    pages = load_all_wiki_pages()
    graph = build_knowledge_graph(pages)
    logger.info(
        "GET /graph served nodes=%s links=%s",
        len(graph["nodes"]),
        len(graph["links"]),
    )
    return graph
