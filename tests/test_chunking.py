"""Chunker invariants.

Citations point at chunks, so the chunker must be deterministic and
offset-faithful: every chunk's text is exactly `source[start:end]`, chunks
never exceed the size budget, and consecutive chunks overlap so answers
spanning a boundary still retrieve.
"""

import pytest

from app.ingestion.chunk import Chunk, chunk_text

MAX = 1200
OVERLAP = 200


def para(char: str, n: int = 500) -> str:
    return char * n


class TestDegenerateInputs:
    def test_empty_text_yields_no_chunks(self):
        assert chunk_text("") == []

    def test_whitespace_only_yields_no_chunks(self):
        assert chunk_text("   \n\n  \t ") == []

    def test_short_text_is_a_single_chunk(self):
        chunks = chunk_text("Hello world.")
        assert len(chunks) == 1
        assert chunks[0].text == "Hello world."
        assert (chunks[0].start, chunks[0].end) == (0, 12)


class TestInvariants:
    @pytest.fixture
    def source(self) -> str:
        sentences = " ".join(f"Sentence number {i} ends here." for i in range(120))
        return "\n\n".join([para("A"), sentences, para("B"), sentences])

    def test_chunks_respect_size_budget(self, source):
        assert all(len(c.text) <= MAX for c in chunk_text(source, max_chars=MAX))

    def test_chunk_text_matches_source_offsets(self, source):
        for c in chunk_text(source, max_chars=MAX):
            assert c.text == source[c.start : c.end]

    def test_chunks_are_sequentially_indexed(self, source):
        chunks = chunk_text(source, max_chars=MAX)
        assert [c.index for c in chunks] == list(range(len(chunks)))

    def test_no_gaps_and_overlap_between_consecutive_chunks(self, source):
        chunks = chunk_text(source, max_chars=MAX, overlap=OVERLAP)
        assert len(chunks) > 1
        for prev, nxt in zip(chunks, chunks[1:]):
            assert nxt.start <= prev.end  # no content falls between chunks
            assert nxt.start > prev.start  # forward progress, always

    def test_full_coverage_of_source(self, source):
        chunks = chunk_text(source, max_chars=MAX)
        assert chunks[0].start == 0
        assert chunks[-1].end == len(source)

    def test_no_blank_chunks(self, source):
        assert all(c.text.strip() for c in chunk_text(source, max_chars=MAX))

    def test_deterministic(self, source):
        assert chunk_text(source, max_chars=MAX) == chunk_text(source, max_chars=MAX)

    def test_chunk_count_is_proportional_to_source_length(self, source):
        """Guards against boundary-crawl: an early paragraph break must not
        stall the window and spray hundreds of overlapping slivers."""
        chunks = chunk_text(source, max_chars=MAX, overlap=OVERLAP)
        # Each window must consume at least (MAX/2 - OVERLAP) fresh chars.
        upper_bound = len(source) // (MAX // 2 - OVERLAP) + 2
        assert len(chunks) <= upper_bound


class TestBoundaryQuality:
    def test_prefers_paragraph_boundaries(self):
        source = "\n\n".join([para("A"), para("B"), para("C")])
        first = chunk_text(source, max_chars=MAX, overlap=0)[0]
        # A+B fit in budget (1002 chars); the break must land on the
        # paragraph gap, not bleed into C.
        assert first.text.rstrip().endswith("B")
        assert "C" not in first.text

    def test_splits_long_paragraphs_at_sentence_boundaries(self):
        source = " ".join(f"Sentence number {i} ends here." for i in range(100))
        chunks = chunk_text(source, max_chars=MAX)
        assert len(chunks) > 1
        for c in chunks[:-1]:
            assert c.text.rstrip().endswith(".")

    def test_pathological_unbroken_text_still_chunks(self):
        source = "x" * 5000  # no whitespace anywhere
        chunks = chunk_text(source, max_chars=MAX)
        assert all(len(c.text) <= MAX for c in chunks)
        assert chunks[-1].end == len(source)


class TestChunkValue:
    def test_chunk_is_a_value_object(self):
        a = Chunk(index=0, text="abc", start=0, end=3)
        b = Chunk(index=0, text="abc", start=0, end=3)
        assert a == b
