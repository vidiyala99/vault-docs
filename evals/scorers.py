"""Scoring math for the RAG eval suite.

All scorers are pure and deterministic. They use their own tokenizer on
purpose: if eval math shared app.rag's tokenizer, a retrieval change could
silently move the yardstick that is supposed to measure it.
"""

import math
import re
from collections.abc import Sequence

# A sentence counts as supported when at least this fraction of its content
# terms appear in a single retrieved chunk. 1.0 would demand verbatim
# extraction; 0.6 tolerates AI-mode paraphrase while still failing
# fabrications, which by definition introduce terms the chunk lacks.
_SUPPORT_THRESHOLD = 0.6

_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "in", "is",
    "it", "of", "on", "or", "that", "the", "this", "to", "under", "when",
    "with",
}


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int = 5) -> float:
    """Binary-relevance NDCG over the top-k retrieved ids."""
    if not relevant:
        return 0.0
    dcg = sum(
        1 / math.log2(rank + 2)
        for rank, item in enumerate(retrieved[:k])
        if item in relevant
    )
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0


def mrr(retrieved: Sequence[str], relevant: set[str]) -> float:
    """Reciprocal rank of the first relevant item (0.0 if none retrieved)."""
    for rank, item in enumerate(retrieved):
        if item in relevant:
            return 1 / (rank + 1)
    return 0.0


def faithfulness(
    answer: str,
    chunk_texts: Sequence[str],
    refusal_text: str | None = None,
) -> float | None:
    """Fraction of answer sentences supported by some retrieved chunk.

    Refusals carry no factual claims and return None (excluded from the
    aggregate rather than scored as free 1.0s).
    """
    if refusal_text is not None and answer.strip() == refusal_text:
        return None
    sentences = _sentences(answer)
    if not sentences:
        return None
    chunk_term_sets = [_terms(text) for text in chunk_texts]
    supported = sum(
        1 for sentence in sentences if _is_supported(sentence, chunk_term_sets)
    )
    return supported / len(sentences)


def citation_coverage(
    answer: str,
    citations: Sequence[str],
    refusal_text: str | None = None,
) -> float | None:
    """1.0 when a factual answer carries at least one citation."""
    if refusal_text is not None and answer.strip() == refusal_text:
        return None
    return 1.0 if citations else 0.0


def _is_supported(sentence: str, chunk_term_sets: list[set[str]]) -> bool:
    sentence_terms = _terms(sentence)
    if not sentence_terms:
        return True  # nothing factual to support
    return any(
        len(sentence_terms & chunk_terms) / len(sentence_terms)
        >= _SUPPORT_THRESHOLD
        for chunk_terms in chunk_term_sets
    )


def _terms(text: str) -> set[str]:
    tokens = (token.strip(".,") for token in re.findall(r"[a-z0-9$.,-]+", text.lower()))
    return {token for token in tokens if len(token) > 1 and token not in _STOP_WORDS}


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
