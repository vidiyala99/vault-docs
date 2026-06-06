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

    def __init__(self):
        self.seen_history = None
        self.seen_chunks = None

    def generate(self, question, chunks, history=()):
        self.seen_history = list(history)
        self.seen_chunks = list(chunks)
        return f"AI answer for: {question}"


class RefusingGenerator:
    mode = "ai"

    def generate(self, question, chunks, history=()):
        return "I could not find that in your documents."


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


def test_anaphoric_followup_retrieves_via_previous_turn(client, db_session_factory):
    """Multi-turn: 'How much is it?' has nothing to retrieve on by itself —
    the condensed query must borrow the previous user turn's terms."""
    doc_id = _upload(
        client,
        "Commercial property policy. The property deductible is $10,000. "
        "Water damage is excluded when caused by flood.",
    )
    _process(client, db_session_factory, doc_id)
    session_id = client.post("/chat/sessions", json={}).json()["id"]

    first = client.post(
        f"/chat/sessions/{session_id}/ask",
        json={"question": "Tell me about the property deductible."},
    )
    assert "$10,000" in first.json()["answer"]

    followup = client.post(
        f"/chat/sessions/{session_id}/ask",
        json={"question": "How much is it?"},
    )

    assert followup.status_code == 200
    body = followup.json()
    assert "$10,000" in body["answer"]
    assert body["citations"][0]["document_id"] == doc_id


def test_followup_without_prior_turns_still_refuses(client, db_session_factory):
    """An anaphoric question on a fresh session has no context to borrow —
    it must refuse, not guess."""
    doc_id = _upload(
        client,
        "Commercial property policy. The property deductible is $10,000.",
    )
    _process(client, db_session_factory, doc_id)
    session_id = client.post("/chat/sessions", json={}).json()["id"]

    resp = client.post(
        f"/chat/sessions/{session_id}/ask",
        json={"question": "How much is it?"},
    )

    assert resp.status_code == 200
    assert resp.json()["answer"] == "I could not find that in your documents."
    assert resp.json()["citations"] == []


def test_ai_generator_receives_session_history(client, db_session_factory):
    from app.api import deps

    doc_id = _upload(
        client,
        "Commercial property policy. The property deductible is $10,000.",
    )
    _process(client, db_session_factory, doc_id)
    generator = FakeGenerator()
    client.app.dependency_overrides[deps.get_generator] = lambda: generator
    session_id = client.post("/chat/sessions", json={}).json()["id"]

    client.post(
        f"/chat/sessions/{session_id}/ask",
        json={"question": "What is the property deductible?"},
    )
    client.post(
        f"/chat/sessions/{session_id}/ask",
        json={"question": "Is that per occurrence?"},
    )

    assert generator.seen_history == [
        ("user", "What is the property deductible?"),
        ("assistant", "AI answer for: What is the property deductible?"),
    ]


def test_ai_refusal_carries_no_citations(client, db_session_factory):
    """When the model refuses, the near-miss chunks that were retrieved are
    not sources of the answer — citing them misleads."""
    from app.api import deps

    doc_id = _upload(
        client,
        "Commercial property policy. The property deductible is $10,000.",
    )
    _process(client, db_session_factory, doc_id)
    client.app.dependency_overrides[deps.get_generator] = lambda: RefusingGenerator()
    session_id = client.post("/chat/sessions", json={}).json()["id"]

    resp = client.post(
        f"/chat/sessions/{session_id}/ask",
        json={"question": "What is the property deductible?"},
    )

    assert resp.status_code == 200
    assert resp.json()["answer"] == "I could not find that in your documents."
    assert resp.json()["citations"] == []


def test_generator_receives_multiple_chunks_of_context(client, db_session_factory):
    """limit=1 starves the model: a question whose answer spans the corpus
    needs more than one retrieved chunk as context."""
    from app.api import deps

    _upload(
        client,
        "Commercial property policy. The property deductible is $10,000.",
        name="policy-a.txt",
    )
    _upload(
        client,
        "Umbrella policy. The property deductible is waived above $1,000,000.",
        name="policy-b.txt",
    )
    for doc in client.get("/documents").json():
        _process(client, db_session_factory, doc["id"])
    generator = FakeGenerator()
    client.app.dependency_overrides[deps.get_generator] = lambda: generator
    session_id = client.post("/chat/sessions", json={}).json()["id"]

    resp = client.post(
        f"/chat/sessions/{session_id}/ask",
        json={"question": "What is the property deductible?"},
    )

    assert resp.status_code == 200
    assert len(generator.seen_chunks) == 2
    assert len(resp.json()["citations"]) == 2


def test_ask_scoped_to_document_only_cites_that_document(client, db_session_factory):
    """Two docs answer the same question differently — scoping to one must
    answer from it, not from the global best match."""
    _upload(
        client,
        "Commercial property policy. The property deductible is $10,000.",
        name="policy-a.txt",
    )
    doc_b = _upload(
        client,
        "Commercial property policy. The property deductible is $25,000.",
        name="policy-b.txt",
    )
    for doc in client.get("/documents").json():
        _process(client, db_session_factory, doc["id"])
    session_id = client.post("/chat/sessions", json={}).json()["id"]

    resp = client.post(
        f"/chat/sessions/{session_id}/ask",
        json={"question": "What is the property deductible?", "document_id": doc_b},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert "$25,000" in body["answer"]
    assert {c["document_id"] for c in body["citations"]} == {doc_b}


def test_ask_scoped_to_document_refuses_when_it_cannot_answer(
    client, db_session_factory
):
    """The answer exists in another document — a scoped ask must refuse
    rather than silently widen the search."""
    _upload(
        client,
        "Payroll records show the employee payroll total is $900,000.",
        name="payroll.txt",
    )
    policy_id = _upload(
        client,
        "Commercial property policy. The property deductible is $10,000.",
        name="policy.txt",
    )
    for doc in client.get("/documents").json():
        _process(client, db_session_factory, doc["id"])
    session_id = client.post("/chat/sessions", json={}).json()["id"]

    resp = client.post(
        f"/chat/sessions/{session_id}/ask",
        json={
            "question": "What is the employee payroll total?",
            "document_id": policy_id,
        },
    )

    assert resp.status_code == 200
    assert resp.json()["answer"] == "I could not find that in your documents."
    assert resp.json()["citations"] == []


def test_ask_scoped_to_unknown_document_is_404(client, db_session_factory):
    session_id = client.post("/chat/sessions", json={}).json()["id"]

    resp = client.post(
        f"/chat/sessions/{session_id}/ask",
        json={"question": "What is the deductible?", "document_id": "nope"},
    )

    assert resp.status_code == 404


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
