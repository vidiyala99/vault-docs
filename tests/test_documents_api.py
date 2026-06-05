"""API contract: upload, dedupe, retrieval, and the processing pipeline.

These run against the real Postgres (skipped when it's down) with storage
pointed at a tmp dir and the queue replaced by a recorder — so the contract
under test is ours, not Redis's.
"""

import pytest
from fastapi.testclient import TestClient

from tests.conftest import requires_db
from tests.test_extraction import make_pdf

pytestmark = requires_db


class FakeQueue:
    def __init__(self):
        self.jobs = []

    def enqueue(self, fn, *args, **kwargs):
        self.jobs.append((fn, args, kwargs))


@pytest.fixture
def fake_queue():
    return FakeQueue()


@pytest.fixture
def client(db_session_factory, tmp_path, fake_queue):
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
    app.dependency_overrides[deps.get_queue] = lambda: fake_queue
    with TestClient(app) as c:
        c.storage = storage
        yield c
    app.dependency_overrides.clear()


def upload(client, content: bytes = b"hello vault document", name: str = "notes.txt"):
    return client.post("/documents", files={"file": (name, content)})


class TestUpload:
    def test_upload_returns_201_with_queued_document(self, client):
        resp = upload(client)
        assert resp.status_code == 201
        body = resp.json()
        assert body["filename"] == "notes.txt"
        assert body["status"] == "queued"
        assert body["size_bytes"] == 20
        assert body["deduplicated"] is False
        assert len(body["content_hash"]) == 64

    def test_upload_enqueues_processing_job(self, client, fake_queue):
        doc_id = upload(client).json()["id"]
        assert len(fake_queue.jobs) == 1
        _, args, _ = fake_queue.jobs[0]
        assert args == (doc_id,)

    def test_duplicate_bytes_return_existing_document(self, client, fake_queue):
        first = upload(client).json()
        resp = upload(client, name="renamed-copy.txt")  # same bytes, new name
        assert resp.status_code == 200
        assert resp.json()["id"] == first["id"]
        assert resp.json()["deduplicated"] is True
        assert len(fake_queue.jobs) == 1  # no second processing job

    def test_unsupported_file_type_is_415(self, client):
        resp = upload(client, name="virus.exe")
        assert resp.status_code == 415
        assert ".exe" in resp.json()["detail"]

    def test_empty_file_is_400(self, client):
        assert upload(client, content=b"").status_code == 400


class TestRetrieval:
    def test_list_documents(self, client):
        upload(client)
        upload(client, content=b"a different file")
        resp = client.get("/documents")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_get_document_by_id(self, client):
        doc_id = upload(client).json()["id"]
        resp = client.get(f"/documents/{doc_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == doc_id

    def test_unknown_document_is_404(self, client):
        assert client.get("/documents/no-such-id").status_code == 404


class TestProcessingPipeline:
    """Run the worker task inline: upload → extract → chunk, status ready."""

    def _run(self, client, db_session_factory, doc_id):
        from app.tasks.process import process_document

        process_document(
            doc_id, session_factory=db_session_factory, storage=client.storage
        )

    def test_txt_document_reaches_ready_with_chunks(self, client, db_session_factory):
        doc_id = upload(client, content=b"Vault stores documents safely.").json()["id"]
        self._run(client, db_session_factory, doc_id)

        body = client.get(f"/documents/{doc_id}").json()
        assert body["status"] == "ready"
        assert body["page_count"] == 1
        assert body["chunk_count"] == 1

    def test_pdf_chunks_carry_page_numbers(self, client, db_session_factory):
        pdf = make_pdf("First page content here.", "Second page content here.")
        doc_id = upload(client, content=pdf, name="policy.pdf").json()["id"]
        self._run(client, db_session_factory, doc_id)

        body = client.get(f"/documents/{doc_id}").json()
        assert body["status"] == "ready"
        assert body["page_count"] == 2

    def test_failure_lands_in_failed_with_message(self, client, db_session_factory):
        doc_id = upload(client, content=b"%PDF-1.4 garbage", name="broken.pdf").json()["id"]
        self._run(client, db_session_factory, doc_id)

        body = client.get(f"/documents/{doc_id}").json()
        assert body["status"] == "failed"
        assert body["error_message"]

    def test_every_transition_is_event_logged(self, client, db_session_factory):
        doc_id = upload(client, content=b"event log test").json()["id"]
        self._run(client, db_session_factory, doc_id)

        from app.models import ProcessingEvent

        with db_session_factory() as s:
            events = (
                s.query(ProcessingEvent)
                .filter_by(document_id=doc_id)
                .order_by(ProcessingEvent.id)
                .all()
            )
        assert [e.to_status for e in events] == [
            "uploaded", "queued", "processing", "ready",
        ]
