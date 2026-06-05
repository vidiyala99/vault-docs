"""FastAPI application entrypoint.

Run locally:
    uvicorn app.main:app --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import documents, health


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
