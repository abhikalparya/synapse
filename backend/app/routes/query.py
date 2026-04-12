import json
import logging
import os

from fastapi import APIRouter, HTTPException
from openai import APIError

from app.models.query import QueryRequest, QueryResponse
from app.prompts.query import build_query_answer_prompt
from app.services.llm import call_llm_query_answer_with_confidence
from app.services.search import find_relevant_pages
from app.services.wiki import (
    create_new_page_from_query,
    load_all_wiki_pages,
    update_wiki_page,
)

router = APIRouter()
logger = logging.getLogger(__name__)


def _writeback_confidence_threshold() -> float:
    raw = os.environ.get("WIKI_WRITEBACK_CONFIDENCE_THRESHOLD", "0.6").strip()
    try:
        t = float(raw)
    except ValueError:
        return 0.6
    return max(0.0, min(1.0, t))


def _page_display_title(page: dict) -> str:
    return str(page.get("title", "")).strip() or page["path"].name


def _build_wiki_context(relevant: list[dict], max_pages: int = 5) -> tuple[str, list[str]]:
    titles: list[str] = []
    blocks: list[str] = []
    for page in relevant[:max_pages]:
        title = _page_display_title(page)
        titles.append(title)
        body = {k: v for k, v in page.items() if k != "path"}
        blocks.append(f"## {title}\n{json.dumps(body, indent=2, ensure_ascii=False)}")
    return "\n\n".join(blocks), titles


@router.post("/query", response_model=QueryResponse)
async def query_wiki(body: QueryRequest):
    pages = load_all_wiki_pages()
    relevant = find_relevant_pages(body.query, pages)
    context_text, used_nodes = _build_wiki_context(relevant)

    try:
        answer, confidence = await call_llm_query_answer_with_confidence(
            build_query_answer_prompt(body.query, context_text)
        )
    except (RuntimeError, APIError) as exc:
        logger.warning("Query LLM answer failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    threshold = _writeback_confidence_threshold()

    if confidence <= threshold:
        logger.info(
            "Query graph impact: used_nodes=%s wiki_action=skipped updated_node=None "
            "(low confidence confidence=%.4f threshold=%.4f)",
            used_nodes,
            confidence,
            threshold,
        )
        return QueryResponse(
            answer=answer,
            used_nodes=used_nodes,
            updated_node=None,
            confidence_score=confidence,
            wiki_action="skipped",
            wiki_file=None,
        )

    updated_node: str | None = None
    try:
        if relevant:
            target = relevant[0]["path"]
            await update_wiki_page(target, body.query, answer, confidence_score=confidence)
            wiki_action = "updated"
            wiki_file = str(target)
            updated_node = _page_display_title(relevant[0])
        else:
            out_path, new_title = await create_new_page_from_query(
                body.query, answer, confidence_score=confidence
            )
            wiki_action = "created"
            wiki_file = str(out_path)
            updated_node = new_title
    except (OSError, ValueError, RuntimeError, APIError) as exc:
        logger.warning("Query wiki write-back failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    logger.info(
        "Query completed wiki_action=%s confidence=%.4f wiki_file=%s used_nodes=%s updated_node=%s",
        wiki_action,
        confidence,
        wiki_file,
        used_nodes,
        updated_node,
    )
    logger.info(
        "Query graph impact: used_nodes=%s wiki_action=%s updated_node=%s",
        used_nodes,
        wiki_action,
        updated_node,
    )
    return QueryResponse(
        answer=answer,
        used_nodes=used_nodes,
        updated_node=updated_node,
        confidence_score=confidence,
        wiki_action=wiki_action,
        wiki_file=wiki_file,
    )
