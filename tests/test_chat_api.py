"""Chat API contract for cited answers.

The first version intentionally exercises deterministic mode: no API key, no
network, but the endpoint must still retrieve from ready document chunks and
return source citations.
"""

import pytest
from conftest import requires_db
from fastapi.testclient import TestClient

pytestmark = requires_db


class FakeQueue:
    def __init__(self):
        self.jobs = []

    def enqueue(self, fn, *args, **kwargs):
        self.jobs.append((fn, args, kwargs))


class FakeGenerator:
    mode = "ai"

    def generate(self, question, chunks):
        return f"AI answer for: {question}"


class FakeVectorEmbedder:
    def embed_documents(self, texts):
        vectors = []
        for text in texts:
            text = text.lower()
            if "deductible" in text:
                vectors.append([1.0, 0.0, 0.0])
            elif "payroll" in text:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return vectors


@pytest.fixture
def client(db_session_factory, tmp_path):
    from app.api import deps
    from app.main import app
    from app.providers import DeterministicGenerator
    from app.storage.local import LocalStorage

    storage = LocalStorage(root=tmp_path / "blobs")
    queue = FakeQueue()

    def override_get_db():
        session = db_session_factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[deps.get_db] = override_get_db
    app.dependency_overrides[deps.get_storage] = lambda: storage
    app.dependency_overrides[deps.get_queue] = lambda: queue
    app.dependency_overrides[deps.get_embedder] = lambda: None
    app.dependency_overrides[deps.get_generator] = lambda: DeterministicGenerator()
    with TestClient(app) as c:
        c.storage = storage
        yield c
    app.dependency_overrides.clear()


def _upload(client, text: str, name: str = "property-policy.txt") -> str:
    resp = client.post("/documents", files={"file": (name, text.encode("utf-8"))})
    assert resp.status_code == 201
    return resp.json()["id"]


def _process(client, db_session_factory, document_id: str) -> None:
    from app.tasks.process import process_document

    process_document(
        document_id,
        session_factory=db_session_factory,
        storage=client.storage,
        embedder=None,
    )


def test_ask_returns_deterministic_answer_with_citations(client, db_session_factory):
    doc_id = _upload(
        client,
        "Commercial property policy. The property deductible is $10,000. "
        "Water damage is excluded when caused by flood.",
    )
    _process(client, db_session_factory, doc_id)

    session_resp = client.post("/chat/sessions", json={})
    assert session_resp.status_code == 201

    ask_resp = client.post(
        f"/chat/sessions/{session_resp.json()['id']}/ask",
        json={"question": "What is the property deductible?"},
    )

    assert ask_resp.status_code == 200
    body = ask_resp.json()
    assert body["mode"] == "deterministic"
    assert "$10,000" in body["answer"]
    assert body["citations"] == [
        {
            "document_id": doc_id,
            "filename": "property-policy.txt",
            "page_number": 1,
            "snippet": (
                "Commercial property policy. The property deductible is $10,000. "
                "Water damage is excluded when caused by flood."
            ),
        }
    ]


def test_ask_refuses_when_documents_do_not_answer_question(client, db_session_factory):
    doc_id = _upload(
        client,
        "Commercial property policy. The property deductible is $10,000.",
    )
    _process(client, db_session_factory, doc_id)
    session_id = client.post("/chat/sessions", json={}).json()["id"]

    resp = client.post(
        f"/chat/sessions/{session_id}/ask",
        json={"question": "Does the deductible include earthquake coverage?"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "deterministic"
    assert body["answer"] == "I could not find that in your documents."
    assert body["citations"] == []


def test_session_messages_include_question_and_answer(client, db_session_factory):
    doc_id = _upload(
        client,
        "Commercial property policy. The property deductible is $10,000.",
    )
    _process(client, db_session_factory, doc_id)
    session_id = client.post("/chat/sessions", json={}).json()["id"]
    client.post(
        f"/chat/sessions/{session_id}/ask",
        json={"question": "What is the property deductible?"},
    )

    resp = client.get(f"/chat/sessions/{session_id}/messages")

    assert resp.status_code == 200
    assert [
        {"role": item["role"], "content": item["content"]} for item in resp.json()
    ] == [
        {"role": "user", "content": "What is the property deductible?"},
        {"role": "assistant", "content": "The property deductible is $10,000."},
    ]


def test_ask_uses_ai_generator_when_configured(client, db_session_factory):
    from app.api import deps

    doc_id = _upload(
        client,
        "Commercial property policy. The property deductible is $10,000.",
    )
    _process(client, db_session_factory, doc_id)
    deps.get_generator.cache_clear()
    client.app.dependency_overrides[deps.get_generator] = lambda: FakeGenerator()
    session_id = client.post("/chat/sessions", json={}).json()["id"]

    resp = client.post(
        f"/chat/sessions/{session_id}/ask",
        json={"question": "What is the property deductible?"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "ai"
    assert body["answer"] == "AI answer for: What is the property deductible?"
    assert body["citations"][0]["document_id"] == doc_id


def test_ask_uses_vector_retrieval_when_embeddings_are_available(
    client, db_session_factory
):
    from app.api import deps
    from app.tasks.process import process_document

    payroll_id = _upload(
        client,
        "Payroll records show the employee payroll total is $900,000.",
        name="payroll.txt",
    )
    deductible_id = _upload(
        client,
        "The property deductible is $10,000.",
        name="policy.txt",
    )
    process_document(
        payroll_id,
        session_factory=db_session_factory,
        storage=client.storage,
        embedder=FakeVectorEmbedder(),
    )
    process_document(
        deductible_id,
        session_factory=db_session_factory,
        storage=client.storage,
        embedder=FakeVectorEmbedder(),
    )
    client.app.dependency_overrides[deps.get_embedder] = lambda: FakeVectorEmbedder()
    session_id = client.post("/chat/sessions", json={}).json()["id"]

    resp = client.post(
        f"/chat/sessions/{session_id}/ask",
        json={"question": "What is the property deductible?"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert "$10,000" in body["answer"]
    assert body["citations"][0]["document_id"] == deductible_id
