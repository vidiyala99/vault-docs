"""The processing pipeline task: queued → processing → ready | failed.

Runs on an RQ worker in production; tests call it inline with an injected
session factory and storage. Any exception lands the document in `failed`
with the error message preserved — a bad file must never wedge a document
in `processing` or kill the worker.
"""

from app.ingestion.chunk import chunk_text
from app.ingestion.extract import extract_pages
from app.lifecycle import DocumentStatus
from app.models import Document, DocumentChunk
from app.services import apply_transition

_DEFAULT_EMBEDDER = object()
_DEFAULT_INSIGHTS_PROVIDER = object()

def process_document(
    document_id: str,
    *,
    session_factory=None,
    storage=None,
    embedder=_DEFAULT_EMBEDDER,
    insights_provider=_DEFAULT_INSIGHTS_PROVIDER,
) -> None:
    if session_factory is None:
        from app.db import SessionLocal

        session_factory = SessionLocal
    if storage is None:
        from app.config import get_settings
        from app.storage.local import LocalStorage

        storage = LocalStorage(root=get_settings().storage_dir)
    if embedder is _DEFAULT_EMBEDDER:
        from app.providers import get_embedder

        embedder = get_embedder()
    if insights_provider is _DEFAULT_INSIGHTS_PROVIDER:
        from app.providers import get_insights_provider

        insights_provider = get_insights_provider()

    with session_factory() as session:
        doc = session.get(Document, document_id)
        if doc is None:  # row vanished; nothing to do
            return

        apply_transition(session, doc, DocumentStatus.PROCESSING)
        session.commit()

        try:
            data = storage.load(doc.content_hash)
            pages = extract_pages(data, doc.filename)

            chunk_index = 0
            chunks_to_embed = []
            for page in pages:
                for chunk in chunk_text(page.text):
                    db_chunk = DocumentChunk(
                        document_id=doc.id,
                        chunk_index=chunk_index,
                        page_number=page.page_number,
                        text=chunk.text,
                        start_offset=chunk.start,
                        end_offset=chunk.end,
                    )
                    session.add(db_chunk)
                    chunks_to_embed.append(db_chunk)
                    chunk_index += 1

            if embedder is not None and chunks_to_embed:
                try:
                    embeddings = embedder.embed_documents(
                        [chunk.text for chunk in chunks_to_embed]
                    )
                    for chunk, embedding in zip(chunks_to_embed, embeddings, strict=True):
                        chunk.embedding = embedding
                except Exception:
                    # Embeddings improve retrieval, but extraction/chunking should still
                    # succeed if the provider is unavailable or the key is invalid.
                    pass

            if insights_provider is not None:
                try:
                    insights = insights_provider.generate(doc.filename, pages)
                except Exception:
                    from app.providers import DeterministicInsightsProvider

                    insights = DeterministicInsightsProvider().generate(doc.filename, pages)
                doc.ai_summary = insights.summary
                doc.ai_key_points = insights.key_points
                doc.ai_document_type = insights.document_type

            doc.page_count = len(pages)
            apply_transition(
                session, doc, DocumentStatus.READY, detail=f"{chunk_index} chunks"
            )
            session.commit()
        except Exception as exc:
            session.rollback()
            doc = session.get(Document, document_id)
            doc.error_message = str(exc)[:2000]
            apply_transition(session, doc, DocumentStatus.FAILED, detail=str(exc)[:500])
            session.commit()
