"""Document upload and retrieval endpoints."""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Response, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_queue, get_storage
from app.ingestion.extract import SUPPORTED_EXTENSIONS
from app.lifecycle import DocumentStatus
from app.models import Document
from app.services import apply_transition, record_created
from app.storage.local import content_hash
from app.tasks.process import process_document

router = APIRouter(prefix="/documents", tags=["documents"])


class DocumentOut(BaseModel):
    id: str
    filename: str
    status: str
    content_hash: str
    size_bytes: int
    page_count: int | None
    chunk_count: int
    error_message: str | None
    created_at: datetime


class UploadOut(DocumentOut):
    deduplicated: bool


class InsightsOut(BaseModel):
    document_id: str
    summary: str | None
    key_points: list[str]
    document_type: str | None


def _out(doc: Document, **extra) -> dict:
    return {
        "id": doc.id,
        "filename": doc.filename,
        "status": doc.status,
        "content_hash": doc.content_hash,
        "size_bytes": doc.size_bytes,
        "page_count": doc.page_count,
        "chunk_count": len(doc.chunks),
        "error_message": doc.error_message,
        "created_at": doc.created_at,
        **extra,
    }


@router.post("", status_code=201, response_model=UploadOut)
def upload_document(
    file: UploadFile,
    response: Response,
    db: Session = Depends(get_db),
    storage=Depends(get_storage),
    queue=Depends(get_queue),
):
    data = file.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="file is empty")

    filename = file.filename or "upload"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"unsupported file type '{ext}'; "
            f"supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    key = content_hash(data)
    existing = db.query(Document).filter_by(content_hash=key).first()
    if existing is not None:
        # Byte-identical upload: no new blob, no new processing job.
        response.status_code = 200
        return _out(existing, deduplicated=True)

    storage.save(data)
    doc = Document(
        filename=filename,
        content_hash=key,
        size_bytes=len(data),
        status=DocumentStatus.UPLOADED.value,
    )
    db.add(doc)
    db.flush()  # assign id before logging events against it
    record_created(db, doc)
    apply_transition(db, doc, DocumentStatus.QUEUED, detail="enqueued for processing")
    db.commit()

    # Enqueue only after commit: the worker must be able to see the row.
    queue.enqueue(process_document, doc.id)
    return _out(doc, deduplicated=False)


@router.get("", response_model=list[DocumentOut])
def list_documents(db: Session = Depends(get_db)):
    docs = db.query(Document).order_by(Document.created_at.desc()).all()
    return [_out(d) for d in docs]


@router.get("/{document_id}", response_model=DocumentOut)
def get_document(document_id: str, db: Session = Depends(get_db)):
    doc = db.get(Document, document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="document not found")
    return _out(doc)


@router.get("/{document_id}/insights", response_model=InsightsOut)
def get_document_insights(document_id: str, db: Session = Depends(get_db)):
    doc = db.get(Document, document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="document not found")
    return {
        "document_id": doc.id,
        "summary": doc.ai_summary,
        "key_points": doc.ai_key_points or [],
        "document_type": doc.ai_document_type,
    }
