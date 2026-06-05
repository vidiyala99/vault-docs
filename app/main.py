"""FastAPI application entrypoint.

Run locally:
    uvicorn app.main:app --reload
"""

from fastapi import FastAPI

from app.api import health

app = FastAPI(
    title="Vault Docs",
    description="AI-powered document vault: upload, async processing, insights, "
    "and chat with citations.",
    version="0.1.0",
)

app.include_router(health.router)
