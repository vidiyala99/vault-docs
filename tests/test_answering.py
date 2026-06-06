"""Extractive answer selection (pure functions — no DB).

Both cases here were caught by the eval suite running against the corpus,
then reduced to unit tests: the eval finds the weakness, the unit test
pins the fix.
"""

from app.models import Document, DocumentChunk
from app.rag import RetrievedChunk, _terms, answer_from_chunks


def make_retrieved(text: str) -> list[RetrievedChunk]:
    doc = Document(filename="doc.txt", content_hash="x" * 64, size_bytes=1)
    chunk = DocumentChunk(
        document_id="d1", chunk_index=0, page_number=1, text=text,
        start_offset=0, end_offset=len(text),
    )
    return [RetrievedChunk(chunk=chunk, document=doc, score=1.0)]


def test_tie_breaks_prefer_the_shortest_supported_sentence():
    """Form-style text produces giant period-less 'sentences' that tie on
    term overlap with the precise target sentence — the short one must win."""
    chunk_text = (
        "APPLICANT INFORMATION\n\n"
        "Applicant Name: Harbor Light Brewing LLC\n"
        "Business Type: Limited Liability Company\n"
        "Nature of Business: Craft brewery with attached taproom and limited "
        "food service.\n\n"
        "Estimated annual payroll is $1,380,000 across 24 employees."
    )
    answer = answer_from_chunks(
        "What is the estimated annual payroll for Harbor Light Brewing?",
        make_retrieved(chunk_text),
    )
    assert answer == "Estimated annual payroll is $1,380,000 across 24 employees."


def test_question_scaffolding_words_are_not_evidence_terms():
    """'When does ... expire?' must not retrieve on the strength of 'when'
    and 'does' — only content-bearing terms count toward support."""
    assert _terms("When does the building lease expire?") == {
        "building",
        "lease",
        "expire",
    }
