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

**Second half of the session — upload API + pipeline, end-to-end:**
- API contract tests written first (upload/dedupe/415/400, retrieval, full
  pipeline inline, event-log ordering). Two genuine catches before any code ran:
  Postgres freezes `now()` per transaction, so same-transaction events shared a
  timestamp and `created_at` ordering was unstable → ProcessingEvent moved to a
  monotonic integer PK. And a test-fixture bug (FastAPI dependency override must
  *be* a generator function, not return a generator) produced a clean RED that
  was about the fixture, not the app — worth distinguishing before "fixing" code.
- Smoke-tested the real loop, not just tests: uvicorn + curl upload + an actual
  RQ worker draining Redis → document `ready` with chunks in ~1s. Windows quirk
  documented: RQ's default worker forks, so local dev uses `SimpleWorker` with
  `TimerDeathPenalty` (production worker will run in Docker).
- Port remap decision (5433/6380): made by me after Claude hit a port collision
  with another project's containers and asked rather than killing them — also
  protects whoever grades this from their own local Postgres on 5432.

---

## Session 2 — Fri night (UI + product research)

**Delegated to Claude Code:**
- Single-page UI (design system generated via a UI/UX skill: dark ops-dashboard
  aesthetic, lifecycle status colors, a11y pass). Built *contract-first* against
  the not-yet-built chat API so the backend can land behind it with zero UI
  changes.
- Web research on the assignment's domain (commercial insurance document
  intelligence) before writing any RAG code — synthesized into
  `docs/product-context.md`. Key finding: in this market citations and refusal
  correctness are E&O-liability features, not nice-to-haves; that ordering now
  drives the chat/eval build priorities.

**Human decisions:** research-before-code ordering; product priorities
(citations > refusal correctness > everything else); pointing the session at a
prior commercial-insurance codebase to port eval-scorer math and the provider
seam shape from (rebuilt fresh, not copied).

---

## Session 3 - chat spine

**Goal:** land the first end-to-end chat contract without requiring API keys.

**Delegated to Claude Code:**
- Wrote the chat API tests first: ready document -> create session -> ask
  question -> deterministic answer with document/page/snippet citation.
- Implemented persisted chat sessions and messages, a deterministic retriever,
  and an extractive deterministic generator. The first green path answers from
  ready chunks and returns citations shaped for the existing UI.
- Added a refusal regression test where a question shares one term with the
  source text but asks for an unsupported fact. The fix was a conservative
  minimum-support threshold so one keyword cannot justify a fabricated answer.
- Added a message history endpoint for the session/history requirement.

**Human decisions:** keep this keyless path intentionally simple and auditable
before adding embeddings/OpenAI. This preserves the reproducible demo and gives
the eval harness a deterministic baseline.

**Verification:** `uv run pytest -q` -> 54 passed.

---

## PDF requirement correction - API key path

After re-reading the assignment PDF, corrected the implementation posture:
OpenAI API or Claude is listed under the required technical stack, with GPT-4
plus embeddings API called out for OpenAI. Keyless deterministic mode remains
valuable as a fallback and reproducible baseline, but the full assignment path
must use an API key.

Implemented provider seams for:
- OpenAI embeddings during document processing when `OPENAI_API_KEY` is set.
- OpenAI-compatible chat generation behind the existing `/chat/sessions/{id}/ask`
  endpoint.
- Deterministic fallback if no key is configured or the generator raises.

Tests use fakes for the provider seams so CI does not require a live key or burn
tokens. Verification after the correction: provider seam tests pass.

---

## Session 4 - embeddings retrieval and reranking

**Goal:** move from lexical-only retrieval to the required embeddings-backed RAG
path while keeping the deterministic fallback.

**Delegated to Claude Code:**
- Wrote a chat API test proving ask-time retrieval uses embedded query vectors
  when embeddings are available.
- Wired `get_embedder` into the chat route, embedded the question at ask time,
  and used pgvector cosine distance to fetch candidate chunks with stored
  embeddings.
- Added a transparent hybrid rerank: 75% vector score, 25% keyword overlap.
  Structure-aware reranking was intentionally left out of v1.
- Added a focused retrieval test that protects the rerank behavior independently
  of the API route.

**Human decisions:** skip structure-aware boosts until there is an eval baseline;
keep the first production-ish retrieval path explainable and easy to compare.

---

## Session 5 - document insights

**Goal:** close the required AI Insights surface from the PDF.

**Delegated to Claude Code:**
- Wrote the processing/API test first: processed document -> persisted summary,
  key points, document type -> `GET /documents/{id}/insights`.
- Added insight fields to the document model.
- Added `InsightsProvider` with OpenAI-backed JSON extraction and deterministic
  fallback.
- Wired insights into the processing pipeline after extraction/chunking.
- Added a deterministic fallback regression test.

**Human decisions:** keep insights as one-time ingest work rather than a chat
hot-path call; this matches the model-routing decision of stronger model for
per-document processing and cheaper model for chat/evals.

---

## Session 6 - metrics APIs and multi-turn chat

**Goal:** close the two remaining required surfaces before the eval suite:
metrics APIs and multi-turn conversations.

**Delegated to Claude Code:**
- Session pickup itself: the new session read CLAUDE.md + git log + the
  uncommitted diff, verified the WIP was green (60 tests), found and fixed a
  duplicate `__all__` that silently shadowed the chat-model exports, then split
  the uncommitted work into three reviewable commits (provider seam / chat /
  docs) rather than one blob.
- Metrics test-first: tests insert processing events with explicit timestamps
  and assert exact derived numbers (failure rate 1/3, avg 3.0s) — the contract
  pinned is "metrics derive from the event log", not "some counters exist".
  `/metrics/documents` and `/metrics/processing` needed zero new writes
  anywhere in the pipeline; they are views over the append-only event log.
- Multi-turn test-first: "Tell me about the property deductible." then
  "How much is it?" must answer $10,000 with a citation; the same follow-up on
  a fresh session must refuse (nothing to borrow). Implementation is a
  deterministic condensation rule — pronoun or <2 content terms folds the
  previous user turn into the retrieval query — plus history passed to the
  OpenAI generator as real chat messages.
- The follow-up test exposed a real tokenizer bug: a question ending
  "deductible." kept the period inside the term ("." is in the term charset
  for "$10,000") and missed the chunk term "deductible". Fix strips edge
  punctuation while preserving it internally; regression covered.

**Human decisions:** condensation stays deterministic (no LLM query-rewrite
call on the hot path) — same answer for the same session transcript every
time, which the eval suite can then score honestly.

**Verification:** `uv run pytest -q` -> 70 passed.

---

## Session 7 — Sat (eval suite shipped, first live OpenAI run, CI gate)

**Goal:** land the eval suite end-to-end — runner, gold set, committed corpus,
baseline, CI regression gate — and verify the full AI path against the real
OpenAI API for the first time.

**Session pickup:** the previous session terminated mid-work. The new session
reconstructed state from CLAUDE.md + git log + the uncommitted diff, verified
the in-flight work was green (90 tests), and resumed without losing anything —
the multi-session workflow's recovery story, exercised for real.

**Live OpenAI E2E smoke (first one):**
- Upload → RQ worker → real `text-embedding-3-small` vectors (1536-d confirmed
  in pgvector) → real `gpt-4o` insights → real `gpt-4o-mini` chat with correct
  citations → multi-turn follow-up → refusal on an unanswerable trap question.
  Total burn: a few cents.
- The smoke immediately caught **schema drift**: `create_all` creates missing
  tables but never alters existing ones, so the long-running dev DB lacked the
  insights columns added in Session 5. Tests never see this because they build
  a fresh schema per run. Fixed with `ALTER TABLE`; documented as a take-home
  scope limit (no Alembic).
- Flagged for polish: a refusal answer still returns one citation — should
  refusals return an empty citations list?

**Eval suite → bugs → unit tests:** running the evals against the corpus caught
two real answering bugs (question scaffolding words counted as evidence terms;
form-style period-less text winning term-overlap ties). Each was reduced to a
pinning unit test before the fix — the eval finds the weakness, the unit test
pins the fix.

**Baseline + CI:**
- Added `--write-baseline`: aggregate metrics only, keyed by mode, so an AI run
  is never compared against the deterministic yardstick and baseline refreshes
  diff cleanly.
- Committed deterministic baseline: NDCG@5 1.0, MRR 1.0, answer accuracy 0.92,
  faithfulness 1.0, citation coverage 1.0, refusal accuracy 0.75. The two
  misses are documented keyless-mode limitations, committed honestly rather
  than tuned away.
- CI (pgvector service + pytest + eval-vs-baseline) caught two repo-breaking
  bugs on its first runs that no local check could ever see:
  1. An unanchored `storage/` gitignore rule had silently excluded
     `app/storage/` from version control — every clone of the repo was broken
     while every local run passed.
  2. Service containers can't mount `db/init/`, so the app database needed its
     pgvector extension created explicitly in the workflow.

**Human decisions:** self-funded $5 OpenAI key (swap to a provided key is one
env var); commit the baseline with its known misses instead of tuning the
refusal threshold the night before submission; prefer `ALTER TABLE` over
dropping the dev database when the schema drifted.

**Verification:** 90 passed locally; CI green on a clean clone including the
eval regression gate.

---

## Prompt engineering log

(Chat/summarization prompt iterations land here as they happen — drafts → final
with reasoning, per the case study's ask.)
