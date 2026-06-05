"""Document lifecycle: uploaded → queued → processing → ready | failed.

The transition guard is the backbone of the async pipeline — a worker must
never move a document along an illegal edge (e.g. straight to ready, or
out of a terminal state).
"""

import pytest

from app.lifecycle import (
    DocumentStatus,
    InvalidTransitionError,
    can_transition,
    transition,
)


class TestHappyPath:
    def test_full_pipeline_path(self):
        """A document walks uploaded → queued → processing → ready."""
        status = DocumentStatus.UPLOADED
        for nxt in (
            DocumentStatus.QUEUED,
            DocumentStatus.PROCESSING,
            DocumentStatus.READY,
        ):
            status = transition(status, nxt)
        assert status is DocumentStatus.READY

    def test_processing_can_fail(self):
        assert (
            transition(DocumentStatus.PROCESSING, DocumentStatus.FAILED)
            is DocumentStatus.FAILED
        )

    def test_failed_can_be_requeued_for_retry(self):
        assert (
            transition(DocumentStatus.FAILED, DocumentStatus.QUEUED)
            is DocumentStatus.QUEUED
        )


class TestGuards:
    def test_cannot_skip_pipeline_stages(self):
        with pytest.raises(InvalidTransitionError):
            transition(DocumentStatus.UPLOADED, DocumentStatus.READY)

    def test_ready_is_terminal(self):
        for target in DocumentStatus:
            assert not can_transition(DocumentStatus.READY, target)

    def test_self_transition_is_invalid(self):
        with pytest.raises(InvalidTransitionError):
            transition(DocumentStatus.PROCESSING, DocumentStatus.PROCESSING)

    def test_error_names_both_states(self):
        with pytest.raises(InvalidTransitionError, match="uploaded.*ready"):
            transition(DocumentStatus.UPLOADED, DocumentStatus.READY)


class TestEnumShape:
    def test_statuses_are_strings_for_db_storage(self):
        assert DocumentStatus.UPLOADED == "uploaded"
        assert DocumentStatus.READY == "ready"
