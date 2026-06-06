"""Document stats and processing metrics.

Processing metrics are derived from the append-only ProcessingEvent log
rather than mutable counters: the event log is already the source of truth
for every status transition, so durations and failure rates fall out of a
single query. At take-home scale the pairing of processing→terminal events
happens in Python; at real scale this becomes a windowed SQL aggregate over
the same table — the data model doesn't change.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.lifecycle import DocumentStatus
from app.models import Document, DocumentChunk, ProcessingEvent

router = APIRouter(prefix="/metrics", tags=["metrics"])

_IN_FLIGHT = (DocumentStatus.QUEUED.value, DocumentStatus.PROCESSING.value)
_TERMINAL = (DocumentStatus.READY.value, DocumentStatus.FAILED.value)


class DocumentStatsOut(BaseModel):
    total: int
    by_status: dict[str, int]
    total_size_bytes: int
    total_pages: int
    total_chunks: int


class ProcessingMetricsOut(BaseModel):
    completed: int
    failed: int
    in_flight: int
    failure_rate: float
    avg_processing_seconds: float | None
    max_processing_seconds: float | None


@router.get("/documents", response_model=DocumentStatsOut)
def document_stats(db: Session = Depends(get_db)):
    by_status = dict(
        db.query(Document.status, func.count()).group_by(Document.status).all()
    )
    total_size, total_pages = db.query(
        func.coalesce(func.sum(Document.size_bytes), 0),
        func.coalesce(func.sum(Document.page_count), 0),
    ).one()
    total_chunks = db.query(func.count(DocumentChunk.id)).scalar()
    return DocumentStatsOut(
        total=sum(by_status.values()),
        by_status=by_status,
        total_size_bytes=total_size,
        total_pages=total_pages,
        total_chunks=total_chunks,
    )


@router.get("/processing", response_model=ProcessingMetricsOut)
def processing_metrics(db: Session = Depends(get_db)):
    completed = _status_count(db, DocumentStatus.READY.value)
    failed = _status_count(db, DocumentStatus.FAILED.value)
    in_flight = (
        db.query(func.count()).filter(Document.status.in_(_IN_FLIGHT)).scalar()
    )

    durations = _processing_durations(db)
    terminal_total = completed + failed
    return ProcessingMetricsOut(
        completed=completed,
        failed=failed,
        in_flight=in_flight,
        failure_rate=failed / terminal_total if terminal_total else 0.0,
        avg_processing_seconds=sum(durations) / len(durations) if durations else None,
        max_processing_seconds=max(durations) if durations else None,
    )


def _status_count(db: Session, status: str) -> int:
    return db.query(func.count()).filter(Document.status == status).scalar()


def _processing_durations(db: Session) -> list[float]:
    """Seconds from each document's `processing` event to its terminal event.

    Documents still in flight (no terminal event yet) are excluded.
    """
    events = (
        db.query(ProcessingEvent)
        .filter(
            ProcessingEvent.to_status.in_(
                (DocumentStatus.PROCESSING.value, *_TERMINAL)
            )
        )
        .order_by(ProcessingEvent.id)
        .all()
    )
    started: dict[str, ProcessingEvent] = {}
    durations: list[float] = []
    for event in events:
        if event.to_status == DocumentStatus.PROCESSING.value:
            started[event.document_id] = event
        elif event.document_id in started:
            start = started.pop(event.document_id)
            durations.append((event.created_at - start.created_at).total_seconds())
    return durations
