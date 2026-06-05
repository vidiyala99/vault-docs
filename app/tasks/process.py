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


def process_document(document_id: str, *, session_factory=None, storage=None) -> None:
    if session_factory is None:
        from app.db import SessionLocal

        session_factory = SessionLocal
    if storage is None:
        from app.config import get_settings
        from app.storage.local import LocalStorage

        storage = LocalStorage(root=get_settings().storage_dir)

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
            for page in pages:
                for chunk in chunk_text(page.text):
                    session.add(
                        DocumentChunk(
                            document_id=doc.id,
                            chunk_index=chunk_index,
                            page_number=page.page_number,
                            text=chunk.text,
                            start_offset=chunk.start,
                            end_offset=chunk.end,
                        )
                    )
                    chunk_index += 1

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
