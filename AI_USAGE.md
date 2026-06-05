# AI Usage Log

How AI tools were used to build this project. Maintained as a running log
during development, not reconstructed afterwards.

**Primary tool:** Claude Code (Opus 4.8) in multi-session workflow — one session
per work block, with a project-level CLAUDE.md carrying decisions between sessions
so each session starts with full context instead of re-deriving it.

**Division of labor, in general:** I set architecture decisions, constraints, and
review every line; Claude Code drafts implementations test-first, runs the
red-green loop, and writes commit messages. Corrections I make by hand are noted
per session below.

---

## Session 1 — Fri evening (scaffold)

**Goal:** repo + infra scaffold, lifecycle, chunker, storage, extraction — the
deterministic spine that everything AI-flavored sits on top of.

**Delegated to Claude Code:**
- Repo/git setup. It caught two things I'd have fumbled: the `.gitignore` went in
  as the *root commit* (so the OpenAI key in `.env` can never appear in history),
  and a `.gitattributes` forcing LF because dev is on Windows while CI will be Linux.
- Project scaffold: pyproject (uv), docker-compose (pgvector Postgres + Redis),
  FastAPI skeleton, settings with model routing baked into config.
- TDD red-green loops for the lifecycle state machine, chunker, storage, and
  extraction modules. Tests were written first and watched fail before each
  implementation, per CLAUDE.md convention ("TDD where it pays").

**Where the AI workflow actually paid off tonight:** the chunker's invariant
test suite (`test_no_blank_chunks`) caught a window-stall bug in the first
implementation — an early paragraph break could stall the sliding window and
spray hundreds of overlapping sliver chunks, including blank ones. That bug
would have silently wrecked retrieval quality (and the embedding bill) with no
visible error. The fix (boundary accepted only if it lands in the second half
of the window) came with a regression test
(`test_chunk_count_is_proportional_to_source_length`).

**Human decisions (not delegated):**
- Hand-rolled RAG over LangChain/LlamaIndex (see README assumptions — transparency
  and evaluability over framework opacity).
- Model routing: gpt-4o-mini on the hot path, gpt-4o for one-time per-document
  insights; deterministic keyless fallback as a first-class mode.
- Untyped pgvector column so OpenAI (1536-d) and local MiniLM (384-d) embeddings
  coexist; documented the production trade-off in the model's docstring.
- RQ over Celery; content-addressed storage as the dedupe mechanism.

**Corrections / reviews this session:** renamed the `.env` var Claude flagged
(`OPEN_API_KEY` → `OPENAI_API_KEY`); reviewed all transition-table edges
(notably: `ready` is terminal, `failed → queued` allowed for retry).

---

## Prompt engineering log

(Chat/summarization prompt iterations land here as they happen — drafts → final
with reasoning, per the case study's ask.)
