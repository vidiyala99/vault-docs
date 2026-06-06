"""Chat endpoints with deterministic cited answers."""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_embedder, get_generator
from app.models import ChatMessage, ChatSession
from app.providers import DeterministicGenerator, Embedder, Generator
from app.rag import retrieve_chunks

router = APIRouter(prefix="/chat", tags=["chat"])


class SessionOut(BaseModel):
    id: str
    created_at: datetime


class AskIn(BaseModel):
    question: str


class CitationOut(BaseModel):
    document_id: str
    filename: str
    page_number: int
    snippet: str


class AskOut(BaseModel):
    answer: str
    mode: str
    citations: list[CitationOut]


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    created_at: datetime


@router.post("/sessions", status_code=201, response_model=SessionOut)
def create_session(db: Session = Depends(get_db)):
    session = ChatSession()
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@router.post("/sessions/{session_id}/ask", response_model=AskOut)
def ask(
    session_id: str,
    payload: AskIn,
    db: Session = Depends(get_db),
    generator: Generator = Depends(get_generator),
    embedder: Embedder | None = Depends(get_embedder),
):
    session = db.get(ChatSession, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="chat session not found")

    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    query_embedding = None
    if embedder is not None:
        try:
            query_embedding = embedder.embed_documents([question])[0]
        except Exception:
            query_embedding = None

    retrieved = retrieve_chunks(db, question, limit=1, query_embedding=query_embedding)
    try:
        answer = generator.generate(question, retrieved)
        mode = generator.mode
    except Exception:
        fallback = DeterministicGenerator()
        answer = fallback.generate(question, retrieved)
        mode = fallback.mode
    citations = [
        CitationOut(
            document_id=item.document.id,
            filename=item.document.filename,
            page_number=item.chunk.page_number,
            snippet=item.chunk.text,
        )
        for item in retrieved
    ]

    db.add(ChatMessage(session_id=session.id, role="user", content=question))
    db.add(ChatMessage(session_id=session.id, role="assistant", content=answer))
    db.commit()

    return AskOut(answer=answer, mode=mode, citations=citations)


@router.get("/sessions/{session_id}/messages", response_model=list[MessageOut])
def list_messages(session_id: str, db: Session = Depends(get_db)):
    session = db.get(ChatSession, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="chat session not found")
    return session.messages
