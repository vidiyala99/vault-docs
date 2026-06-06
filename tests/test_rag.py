"""Retrieval ranking behavior."""

from conftest import requires_db

pytestmark = requires_db


def test_vector_retrieval_reranks_candidates_with_keyword_overlap(db_session_factory):
    from app.models import Document, DocumentChunk
    from app.rag import retrieve_chunks

    with db_session_factory() as session:
        vector_only = Document(
            filename="vector-only.txt",
            content_hash="a" * 64,
            size_bytes=10,
            status="ready",
        )
        hybrid = Document(
            filename="hybrid.txt",
            content_hash="b" * 64,
            size_bytes=10,
            status="ready",
        )
        session.add_all([vector_only, hybrid])
        session.flush()
        session.add_all(
            [
                DocumentChunk(
                    document_id=vector_only.id,
                    chunk_index=0,
                    page_number=1,
                    text="General policy information.",
                    start_offset=0,
                    end_offset=27,
                    embedding=[1.0, 0.0, 0.0],
                ),
                DocumentChunk(
                    document_id=hybrid.id,
                    chunk_index=0,
                    page_number=1,
                    text="The property deductible is $10,000.",
                    start_offset=0,
                    end_offset=36,
                    embedding=[0.95, 0.05, 0.0],
                ),
            ]
        )
        session.commit()

        results = retrieve_chunks(
            session,
            "What is the property deductible?",
            query_embedding=[1.0, 0.0, 0.0],
            limit=2,
        )

    assert [item.document.filename for item in results] == [
        "hybrid.txt",
        "vector-only.txt",
    ]
