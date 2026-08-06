import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.generate import router as generate_router
from app.routes.graph import router as graph_router
from app.routes.ingest import router as ingest_router
from app.routes.lint import router as lint_router
from app.routes.proposals import router as proposals_router
from app.routes.quiz import router as quiz_router
from app.routes.stats import router as stats_router
from app.routes.topics import router as topics_router
from app.services.llm import close_async_openai_client

_backend_dir = Path(__file__).resolve().parent.parent
for _env_path in (_backend_dir.parent / ".env", _backend_dir / ".env"):
    with suppress(OSError):
        load_dotenv(_env_path)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    yield
    await close_async_openai_client()


app = FastAPI(title="Synapse", version="0.1.0", lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:4173",
        "http://localhost:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest_router)
app.include_router(generate_router)
app.include_router(topics_router)
app.include_router(quiz_router)
app.include_router(proposals_router)
app.include_router(graph_router)
app.include_router(stats_router)
app.include_router(lint_router)
