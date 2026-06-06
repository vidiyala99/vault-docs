"""API contract: document stats and processing metrics.

Processing metrics are derived from the append-only ProcessingEvent log,
not from separate counters — these tests pin that contract by inserting
events with explicit timestamps and asserting exact derived numbers.
"""

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from tests.conftest import requires_db
from tests.test_documents_api import FakeQueue, upload

pytestmark = requires_db


@pytest.fixture
def client(db_session_factory, tmp_path):
    from app.api import deps
    from app.main import app
    from app.storage.local import LocalStorage

    storage = LocalStorage(root=tmp_path / "blobs")

    def override_get_db():
        session = db_session_factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[deps.get_db] = override_get_db
    app.dependency_overrides[deps.get_storage] = lambda: storage
    app.dependency_overrides[deps.get_queue] = lambda: FakeQueue()
    with TestClient(app) as c:
        c.storage = storage
        yield c
    app.dependency_overrides.clear()


def run_pipeline(client, db_session_factory, doc_id):
    from app.tasks.process import process_document

    process_document(
        doc_id,
        session_factory=db_session_factory,
        storage=client.storage,
        embedder=None,
    )


class TestDocumentStats:
    def test_empty_vault_returns_zeroes(self, client):
        resp = client.get("/metrics/documents")
        assert resp.status_code == 200
        assert resp.json() == {
            "total": 0,
            "by_status": {},
            "total_size_bytes": 0,
            "total_pages": 0,
            "total_chunks": 0,
        }

    def test_counts_reflect_uploads_and_processing(self, client, db_session_factory):
        ready_id = upload(client, content=b"Vault stores documents safely.").json()["id"]
        run_pipeline(client, db_session_factory, ready_id)
        upload(client, content=b"still waiting in the queue")

        body = client.get("/metrics/documents").json()
        assert body["total"] == 2
        assert body["by_status"] == {"ready": 1, "queued": 1}
        assert body["total_size_bytes"] == 30 + 26
        assert body["total_pages"] == 1  # queued doc has no pages yet
        assert body["total_chunks"] == 1

    def test_failed_documents_are_counted(self, client, db_session_factory):
        doc_id = upload(client, content=b"%PDF-1.4 garbage", name="broken.pdf").json()["id"]
        run_pipeline(client, db_session_factory, doc_id)

        body = client.get("/metrics/documents").json()
        assert body["by_status"] == {"failed": 1}


def seed_processed_document(
    session_factory,
    *,
    filename: str,
    outcome: str,
    started: datetime,
    finished: datetime,
):
    """Insert a document plus a processing→terminal event pair with
    explicit timestamps, so duration assertions are exact."""
    from app.models import Document, ProcessingEvent

    with session_factory() as s:
        doc = Document(
            filename=filename,
            content_hash=f"hash-{filename}",
            size_bytes=10,
            status=outcome,
        )
        s.add(doc)
        s.flush()
        s.add(
            ProcessingEvent(
                document_id=doc.id,
                from_status="queued",
                to_status="processing",
                created_at=started,
            )
        )
        s.add(
            ProcessingEvent(
                document_id=doc.id,
                from_status="processing",
                to_status=outcome,
                created_at=finished,
            )
        )
        s.commit()


class TestProcessingMetrics:
    def test_no_processing_yet(self, client):
        resp = client.get("/metrics/processing")
        assert resp.status_code == 200
        assert resp.json() == {
            "completed": 0,
            "failed": 0,
            "in_flight": 0,
            "failure_rate": 0.0,
            "avg_processing_seconds": None,
            "max_processing_seconds": None,
        }

    def test_durations_and_failure_rate_derive_from_event_log(
        self, client, db_session_factory
    ):
        t0 = datetime(2026, 6, 6, 12, 0, 0, tzinfo=UTC)
        seed_processed_document(
            db_session_factory,
            filename="fast.txt",
            outcome="ready",
            started=t0,
            finished=t0 + timedelta(seconds=2),
        )
        seed_processed_document(
            db_session_factory,
            filename="slow.txt",
            outcome="ready",
            started=t0,
            finished=t0 + timedelta(seconds=6),
        )
        seed_processed_document(
            db_session_factory,
            filename="broken.pdf",
            outcome="failed",
            started=t0,
            finished=t0 + timedelta(seconds=1),
        )

        body = client.get("/metrics/processing").json()
        assert body["completed"] == 2
        assert body["failed"] == 1
        assert body["failure_rate"] == pytest.approx(1 / 3)
        assert body["avg_processing_seconds"] == pytest.approx(3.0)
        assert body["max_processing_seconds"] == pytest.approx(6.0)

    def test_in_flight_counts_queued_and_processing_documents(self, client):
        upload(client, content=b"queued document one")
        upload(client, content=b"queued document two")

        body = client.get("/metrics/processing").json()
        assert body["in_flight"] == 2
        assert body["completed"] == 0

    def test_end_to_end_pipeline_shows_up_in_metrics(self, client, db_session_factory):
        doc_id = upload(client, content=b"measure this document").json()["id"]
        run_pipeline(client, db_session_factory, doc_id)

        body = client.get("/metrics/processing").json()
        assert body["completed"] == 1
        assert body["in_flight"] == 0
        assert body["avg_processing_seconds"] is not None
        assert body["avg_processing_seconds"] >= 0.0
