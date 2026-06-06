"""FastAPI application entrypoint.

Run locally:
    uvicorn app.main:app --reload
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.api import chat, documents, health, metrics

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Take-home scope: create_all instead of Alembic migrations (documented
    # assumption — schema is append-only over a weekend).
    from app import models  # noqa: F401  (register tables on Base)
    from app.db import Base, engine

    Base.metadata.create_all(engine)
    yield


app = FastAPI(
    lifespan=lifespan,
    title="Vault Docs",
    description="AI-powered document vault: upload, async processing, insights, "
    "and chat with citations.",
    version="0.1.0",
)

app.include_router(health.router)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(metrics.router)


@app.get("/", include_in_schema=False)
def ui() -> FileResponse:
    """Single-page UI — no build step, served straight from the repo."""
    return FileResponse(_STATIC_DIR / "index.html")
