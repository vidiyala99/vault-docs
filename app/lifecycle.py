"""Document processing lifecycle: typed statuses with guarded transitions.

    uploaded → queued → processing → ready
                  ↑          ↓
                  └──────  failed

`ready` is terminal. `failed` may be re-queued (retry). Workers must go
through `transition()` so an illegal move is an exception, not silent
state corruption.
"""

from enum import StrEnum


class DocumentStatus(StrEnum):
    UPLOADED = "uploaded"
    QUEUED = "queued"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


_ALLOWED: dict[DocumentStatus, frozenset[DocumentStatus]] = {
    DocumentStatus.UPLOADED: frozenset({DocumentStatus.QUEUED}),
    DocumentStatus.QUEUED: frozenset({DocumentStatus.PROCESSING}),
    DocumentStatus.PROCESSING: frozenset({DocumentStatus.READY, DocumentStatus.FAILED}),
    DocumentStatus.READY: frozenset(),
    DocumentStatus.FAILED: frozenset({DocumentStatus.QUEUED}),
}


class InvalidTransitionError(Exception):
    def __init__(self, current: DocumentStatus, target: DocumentStatus) -> None:
        super().__init__(f"cannot transition from '{current.value}' to '{target.value}'")
        self.current = current
        self.target = target


def can_transition(current: DocumentStatus, target: DocumentStatus) -> bool:
    return target in _ALLOWED[current]


def transition(current: DocumentStatus, target: DocumentStatus) -> DocumentStatus:
    """Return the new status, or raise InvalidTransitionError."""
    if not can_transition(current, target):
        raise InvalidTransitionError(current, target)
    return target
