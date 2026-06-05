"""Document state changes, with the event log kept in lockstep.

All status writes go through `apply_transition` — it enforces the
lifecycle guard AND appends the ProcessingEvent row, so the status column
and the event log cannot drift apart.
"""

from sqlalchemy.orm import Session

from app.lifecycle import DocumentStatus, transition
from app.models import Document, ProcessingEvent


def record_created(session: Session, document: Document) -> None:
    """Log the birth event (no prior status)."""
    session.add(
        ProcessingEvent(
            document_id=document.id,
            from_status=None,
            to_status=document.status,
        )
    )


def apply_transition(
    session: Session,
    document: Document,
    target: DocumentStatus,
    detail: str | None = None,
) -> None:
    """Move `document` to `target`, raising on illegal edges."""
    current = DocumentStatus(document.status)
    document.status = transition(current, target).value
    session.add(
        ProcessingEvent(
            document_id=document.id,
            from_status=current.value,
            to_status=target.value,
            detail=detail,
        )
    )
