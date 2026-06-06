"""Chat endpoints with deterministic cited answers."""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_embedder, get_generator
from app.models import ChatMessage, ChatSession, Document
from app.providers import DeterministicGenerator, Embedder, Generator
from app.rag import REFUSAL_TEXT, condense_question, retrieve_chunks

router = APIRouter(prefix="/chat", tags=["chat"])

# Earlier turns passed to the generator — enough for follow-up resolution
# without letting long sessions inflate the prompt.
_HISTORY_LIMIT = 6


class SessionOut(BaseModel):
    id: str
    created_at: datetime


class AskIn(BaseModel):
    question: str
    # Optional scope: restrict retrieval to one document ("chat with this
    # document"). Unset = search the whole vault.
    document_id: str | None = None


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

    if payload.document_id is not None:
        if db.get(Document, payload.document_id) is None:
            raise HTTPException(status_code=404, detail="document not found")

    # Prior turns, captured before this question is written to the session.
    history = tuple(
        (message.role, message.content)
        for message in session.messages[-_HISTORY_LIMIT:]
    )
    prior_user = [content for role, content in history if role == "user"]
    retrieval_question = condense_question(question, prior_user)

    query_embedding = None
    if embedder is not None:
        try:
            query_embedding = embedder.embed_documents([retrieval_question])[0]
        except Exception:
            query_embedding = None

    # Three chunks of context: one starves the generator on questions whose
    # support spans a document, and the prompt labels each source anyway.
    retrieved = retrieve_chunks(
        db,
        retrieval_question,
        limit=3,
        query_embedding=query_embedding,
        document_id=payload.document_id,
    )
    try:
        answer = generator.generate(question, retrieved, history=history)
        mode = generator.mode
    except Exception:
        fallback = DeterministicGenerator()
        answer = fallback.generate(question, retrieved, history=history)
        mode = fallback.mode

    # A refusal has no sources: the near-miss chunks that were retrieved did
    # not support the answer, and citing them would dress up "I don't know".
    refused = answer.strip() == REFUSAL_TEXT
    citations = (
        []
        if refused
        else [
            CitationOut(
                document_id=item.document.id,
                filename=item.document.filename,
                page_number=item.chunk.page_number,
                snippet=item.chunk.text,
            )
            for item in retrieved
        ]
    )

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
