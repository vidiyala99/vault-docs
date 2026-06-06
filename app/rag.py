"""Small deterministic retrieval/generation path for keyless chat."""

import re
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.lifecycle import DocumentStatus
from app.models import Document, DocumentChunk

_STOP_WORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "is",
    "of",
    "the",
    "to",
    "what",
    # Question scaffolding carries no retrievable content — "When does the
    # lease expire?" must not count "when"/"does" as evidence of support.
    "can",
    "could",
    "did",
    "do",
    "does",
    "how",
    "many",
    "much",
    "should",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "why",
    "would",
}

# Multi-turn condensation: a follow-up that leans on the previous turn
# ("How much is it?") carries nothing retrievable by itself. The rule is
# deterministic and inspectable — pronoun present, or fewer than two
# content-bearing terms once question scaffolding is stripped.
_PRONOUNS = {"it", "that", "this", "they", "them", "those", "these", "its"}
_QUESTION_WORDS = {"how", "much", "many", "when", "where", "which", "who", "why", "does", "do", "can"}


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: DocumentChunk
    document: Document
    score: float


def retrieve_chunks(
    db: Session,
    question: str,
    *,
    limit: int = 3,
    min_score: int = 2,
    query_embedding: list[float] | None = None,
) -> list[RetrievedChunk]:
    terms = _terms(question)
    if not terms:
        return []

    if query_embedding is not None:
        vector_ranked = _retrieve_vector_candidates(db, terms, query_embedding, limit=limit)
        if vector_ranked:
            return vector_ranked

    rows = (
        db.query(DocumentChunk, Document)
        .join(Document, Document.id == DocumentChunk.document_id)
        .filter(Document.status == DocumentStatus.READY.value)
        .all()
    )
    ranked = []
    for chunk, document in rows:
        text_terms = set(_terms(chunk.text))
        score = len(terms & text_terms)
        if score >= min_score:
            ranked.append(RetrievedChunk(chunk=chunk, document=document, score=score))

    ranked.sort(key=lambda item: (-item.score, item.document.filename, item.chunk.chunk_index))
    return ranked[:limit]


def _retrieve_vector_candidates(
    db: Session,
    question_terms: set[str],
    query_embedding: list[float],
    *,
    limit: int,
    candidate_limit: int = 20,
) -> list[RetrievedChunk]:
    distance = DocumentChunk.embedding.cosine_distance(query_embedding)
    rows = (
        db.query(DocumentChunk, Document, distance.label("distance"))
        .join(Document, Document.id == DocumentChunk.document_id)
        .filter(Document.status == DocumentStatus.READY.value)
        .filter(DocumentChunk.embedding.is_not(None))
        .order_by(distance)
        .limit(candidate_limit)
        .all()
    )
    ranked = []
    for chunk, document, chunk_distance in rows:
        vector_score = 1.0 - float(chunk_distance or 0.0)
        keyword_score = _keyword_score(question_terms, set(_terms(chunk.text)))
        final_score = (0.75 * vector_score) + (0.25 * keyword_score)
        ranked.append(
            RetrievedChunk(chunk=chunk, document=document, score=final_score)
        )

    ranked.sort(key=lambda item: (-item.score, item.document.filename, item.chunk.chunk_index))
    return ranked[:limit]


def condense_question(question: str, prior_user_messages: list[str]) -> str:
    """Fold the previous user turn into an anaphoric follow-up so retrieval
    has terms to work with. Standalone questions pass through unchanged."""
    if not prior_user_messages or not _is_anaphoric(question):
        return question
    return f"{prior_user_messages[-1]} {question}"


def _is_anaphoric(question: str) -> bool:
    words = set(re.findall(r"[a-z]+", question.lower()))
    if words & _PRONOUNS:
        return True
    content_terms = _terms(question) - _QUESTION_WORDS - _PRONOUNS
    return len(content_terms) < 2


def answer_from_chunks(question: str, chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "I could not find that in your documents."

    question_terms = _terms(question)
    sentences = _sentences(chunks[0].chunk.text)
    # Tie-break equal term overlap toward the shortest sentence: form-style
    # text yields giant period-less "sentences" that match on incidental
    # terms, and the precise short sentence should beat them.
    best = max(
        sentences,
        key=lambda sentence: (len(question_terms & _terms(sentence)), -len(sentence)),
        default=chunks[0].chunk.text,
    )
    return best.strip()


def _terms(text: str) -> set[str]:
    # "." and "," stay inside terms ("$10,000") but are stripped at the
    # edges so "deductible." and "deductible" are the same term.
    terms = (term.strip(".,") for term in re.findall(r"[a-z0-9$.,]+", text.lower()))
    return {term for term in terms if len(term) > 1 and term not in _STOP_WORDS}


def _keyword_score(question_terms: set[str], text_terms: set[str]) -> float:
    if not question_terms:
        return 0.0
    return len(question_terms & text_terms) / len(question_terms)


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
