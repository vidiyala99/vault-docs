"""Scorer math for the eval suite.

Pure functions, no DB, no network. The scorers are deliberately independent
of app.rag's tokenizer: the yardstick must not move when retrieval changes.
"""

import pytest

from evals.scorers import (
    citation_coverage,
    faithfulness,
    mrr,
    ndcg_at_k,
)

REFUSAL = "I could not find that in your documents."


class TestNDCG:
    def test_relevant_at_rank_one_is_perfect(self):
        assert ndcg_at_k(["a", "b", "c"], {"a"}, k=5) == pytest.approx(1.0)

    def test_relevant_at_rank_two_is_discounted(self):
        # DCG = 1/log2(3); ideal DCG = 1/log2(2) = 1
        import math

        expected = (1 / math.log2(3)) / 1.0
        assert ndcg_at_k(["x", "a", "y"], {"a"}, k=5) == pytest.approx(expected)

    def test_no_relevant_retrieved_is_zero(self):
        assert ndcg_at_k(["x", "y"], {"a"}, k=5) == 0.0

    def test_empty_retrieval_is_zero(self):
        assert ndcg_at_k([], {"a"}, k=5) == 0.0

    def test_only_top_k_counts(self):
        assert ndcg_at_k(["x", "y", "z", "w", "v", "a"], {"a"}, k=5) == 0.0

    def test_multiple_relevant_perfect_ordering(self):
        assert ndcg_at_k(["a", "b", "x"], {"a", "b"}, k=5) == pytest.approx(1.0)


class TestMRR:
    def test_first_hit_at_rank_one(self):
        assert mrr(["a", "x"], {"a"}) == pytest.approx(1.0)

    def test_first_hit_at_rank_three(self):
        assert mrr(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)

    def test_no_hit_is_zero(self):
        assert mrr(["x", "y"], {"a"}) == 0.0


class TestFaithfulness:
    CHUNKS = [
        "Commercial property policy. The property deductible is $10,000. "
        "Water damage is excluded when caused by flood."
    ]

    def test_extractive_answer_is_fully_supported(self):
        score = faithfulness("The property deductible is $10,000.", self.CHUNKS)
        assert score == pytest.approx(1.0)

    def test_fabricated_sentence_scores_zero(self):
        score = faithfulness(
            "The earthquake deductible is $50,000 per occurrence.", self.CHUNKS
        )
        assert score == 0.0

    def test_mixed_answer_scores_fraction_of_supported_sentences(self):
        answer = (
            "The property deductible is $10,000. "
            "Earthquake losses are reimbursed at replacement cost."
        )
        assert faithfulness(answer, self.CHUNKS) == pytest.approx(0.5)

    def test_paraphrase_with_strong_term_overlap_is_supported(self):
        # AI mode paraphrases; support = enough content-term overlap
        # with a single chunk, not verbatim substring match.
        score = faithfulness(
            "Flood-caused water damage is excluded under this policy.", self.CHUNKS
        )
        assert score == pytest.approx(1.0)

    def test_refusal_is_not_scored(self):
        assert faithfulness(REFUSAL, self.CHUNKS, refusal_text=REFUSAL) is None

    def test_no_chunks_means_nothing_is_supported(self):
        assert faithfulness("Any claim at all.", []) == 0.0


class TestCitationCoverage:
    def test_answer_with_citation_is_covered(self):
        assert citation_coverage("The deductible is $10,000.", ["doc-1"]) == 1.0

    def test_answer_without_citation_is_uncovered(self):
        assert citation_coverage("The deductible is $10,000.", []) == 0.0

    def test_refusal_is_not_scored(self):
        assert citation_coverage(REFUSAL, [], refusal_text=REFUSAL) is None
