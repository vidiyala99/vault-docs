"""Deterministic, offset-faithful text chunking.

Chunks are sliding windows over the source string: each chunk's text is
exactly ``source[start:end]``, so citations can always be traced back to
the original document. Window boundaries snap to the most natural break
available inside the size budget, in preference order:

    paragraph break  >  sentence end  >  word boundary  >  hard split

Consecutive windows overlap by ``overlap`` characters so a fact straddling
a boundary is retrievable from at least one chunk. No tokenizer dependency:
character budgets keep the chunker fully deterministic and library-free
(~4 chars/token means the 1200-char default is ≈300 tokens).
"""

import re
from dataclasses import dataclass

_PARA_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s")
_WORD_BREAK = re.compile(r"\s")


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    start: int  # char offset into the source text
    end: int


def chunk_text(text: str, *, max_chars: int = 1200, overlap: int = 200) -> list[Chunk]:
    """Split ``text`` into overlapping chunks of at most ``max_chars``."""
    if not text.strip():
        return []
    if len(text) <= max_chars:
        return [Chunk(index=0, text=text, start=0, end=len(text))]

    spans: list[tuple[int, int]] = []
    start = 0
    n = len(text)
    while start < n:
        hard_end = min(start + max_chars, n)
        end = n if hard_end == n else _best_break(text, start, hard_end)
        spans.append((start, end))
        if end >= n:
            break
        # Step back `overlap` chars for the next window, but always advance.
        start = max(end - overlap, start + 1)

    return [
        Chunk(index=i, text=text[s:e], start=s, end=e) for i, (s, e) in enumerate(spans)
    ]


def _best_break(text: str, start: int, hard_end: int) -> int:
    """Best break position in (start, hard_end], by boundary preference.

    A boundary is only accepted if it lands in the second half of the
    window: a break that wastes more than half the budget would stall the
    sliding window (emitting sliver chunks) — better to fall through to a
    finer-grained boundary, or ultimately a hard split.
    """
    window = text[start:hard_end]
    min_end = len(window) // 2
    for pattern in (_PARA_BREAK, _SENTENCE_END, _WORD_BREAK):
        last = None
        for m in pattern.finditer(window):
            last = m
        if last is not None and last.end() >= min_end:
            return start + last.end()
    return hard_end
