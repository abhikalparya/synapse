import logging

from fastapi import APIRouter

from app.models.stats import KnowledgeStatsResponse
from app.services.stats import compute_knowledge_stats
from app.services.wiki import load_all_wiki_pages

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/stats", response_model=KnowledgeStatsResponse)
async def get_knowledge_stats():
    pages = load_all_wiki_pages()
    stats = compute_knowledge_stats(pages)
    logger.info(
        "GET /stats total_nodes=%s total_edges=%s recent=%s top_tags=%s",
        stats["total_nodes"],
        stats["total_edges"],
        len(stats["recent_nodes"]),
        len(stats["top_tags"]),
    )
    return stats
