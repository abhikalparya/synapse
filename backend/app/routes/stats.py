import logging

from fastapi import APIRouter

from app.models.stats import KnowledgeStatsResponse
from app.services.stats import compute_knowledge_stats
from app.services.topics import load_all_topics, load_dependencies

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/stats", response_model=KnowledgeStatsResponse)
async def get_knowledge_stats():
    topics = load_all_topics()
    dependencies = load_dependencies()
    stats = compute_knowledge_stats(topics, dependencies)
    logger.info(
        "GET /stats total_nodes=%s total_edges=%s recent=%s",
        stats["total_nodes"],
        stats["total_edges"],
        len(stats["recent_nodes"]),
    )
    return stats
