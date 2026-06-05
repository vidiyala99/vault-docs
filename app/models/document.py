"""Document, chunk, and processing-event models.

Design notes:
- `content_hash` is unique: byte-identical uploads are detected at the API
  layer and short-circuit to the existing document (dedupe).
- `embedding` is an untyped pgvector column (no fixed dimension) so the
  OpenAI path (text-embedding-3-small, 1536-d) and the keyless local path
  (all-MiniLM-L6-v2, 384-d) coexist. At corpus scale you would pin the
  dimension and add an HNSW index; at take-home scale exact scan is both
  honest and fast.
- `ProcessingEvent` is an append-only log of every status transition.
  Processing metrics (throughput, durations, failure rates) are derived
  from it instead of being tracked in separate mutable counters.
"""

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base
from app.lifecycle import DocumentStatus


def _uuid() -> str:
    return str(uuid.uuid4())


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    filename: Mapped[str] = mapped_column(String(512))
    content_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    size_bytes: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(
        String(20), default=DocumentStatus.UPLOADED.value, index=True
    )
    error_message: Mapped[str | None] = mapped_column(Text, default=None)
    page_count: Mapped[int | None] = mapped_column(Integer, default=None)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )
    events: Mapped[list["ProcessingEvent"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="ProcessingEvent.id",
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer)
    page_number: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    start_offset: Mapped[int] = mapped_column(Integer)
    end_offset: Mapped[int] = mapped_column(Integer)
    embedding = mapped_column(Vector(), nullable=True)

    document: Mapped[Document] = relationship(back_populates="chunks")


class ProcessingEvent(Base):
    __tablename__ = "processing_events"

    # Monotonic integer PK: Postgres `now()` is frozen per transaction, so
    # two events written in one transaction share a timestamp — the id is
    # the reliable ordering.
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    from_status: Mapped[str | None] = mapped_column(String(20), default=None)
    to_status: Mapped[str] = mapped_column(String(20))
    detail: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )

    document: Mapped[Document] = relationship(back_populates="events")
