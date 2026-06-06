# Vault Docs

AI-powered document vault for broker-style document review: upload files, process
them asynchronously, preserve source provenance, and answer questions with
citations.

This is a take-home implementation for an insurance-document intelligence use
case. The design bias is explicit: deterministic spine first, AI features behind
clear boundaries, and citations/refusals treated as product-critical behavior.

## Current Status

Implemented:

- FastAPI app with a served single-page UI at `/`
- Document upload, content-hash dedupe, and retrieval APIs
- Local file storage
- PostgreSQL models for documents, chunks, and append-only processing events
- RQ-compatible processing pipeline: extract text, chunk by page, write chunks
- Guarded document lifecycle transitions
- Chat sessions, ask endpoint, and message history
- Deterministic keyless retrieval/answering with source citations and refusal
  behavior
- OpenAI provider seams for chat generation and embeddings when `OPENAI_API_KEY`
  is configured
- pgvector retrieval over stored embeddings with simple hybrid vector/keyword
  reranking
- AI document insights: summary, key points, and document type, with OpenAI
  provider and deterministic fallback
- Tests for storage, lifecycle, extraction, chunking, document APIs, and chat

In progress / next:

- Metrics API and eval harness

## Architecture

```text
Browser UI
   |
FastAPI
   |
   +-- Documents API
   |     +-- content-hash dedupe
   |     +-- local storage
   |     +-- queued processing job
   |
   +-- Processing worker
   |     +-- extract pages
   |     +-- deterministic chunking
   |     +-- document/page/offset provenance
   |
   +-- RAG layer
         +-- embed
         +-- retrieve ranked chunks
         +-- answer with citations or refuse
```

The first milestone is the ingestion spine. The RAG layer is designed to attach
to chunks that already carry document id, page number, and character offsets, so
answers can cite the original source instead of only citing model output.

The current chat path can run two ways. With `OPENAI_API_KEY`, processing calls
the OpenAI embeddings API, generates document insights with the configured
insights model, and chat uses the configured OpenAI-compatible chat model.
Retrieval uses pgvector over stored chunk embeddings when embeddings are
available, then reranks candidates with a simple transparent blend:
`0.75 * vector_score + 0.25 * keyword_overlap`. Without embeddings or a provider
key, the app falls back to deterministic insights and lexical retrieval:
conservative matching, extractive answers when the context supports the
question, and refusal when the available chunks do not support the requested
fact. The fallback is for reproducible tests and graceful degradation; the full
assignment path uses an API key.

## Provider Seam

The provider seam is the accountability boundary of the system: embedding,
retrieval, generation, and insights are separate capabilities instead of being
hard-wired into an API route.

Where it matters in insurance: broker workflows need graceful degradation. If
an AI provider is unavailable during a renewal review, the system should not turn
into a 500 error or silently change behavior. It should fall back to
deterministic retrieval or answering where possible, mark the response mode
explicitly, and preserve citations. A weaker cited answer is safer than an
unavailable assistant or an uncited fluent one.

Why not just call OpenAI directly: direct calls are faster to write, but they
make three things harder to evaluate: retrieval quality independent of answer
style, refusal behavior independent of model variance, and keyless
reproducibility for graders. The seam lets the same `ask()` pipeline run with
either a real LLM provider or a deterministic fallback, and lets the eval harness
compare provider stacks instead of mixing them into one score.

The intended split is:

- `Embedder`: text to vector, with OpenAI embeddings when keyed and local or
  deterministic fallback when not.
- `Retriever`: query to ranked chunks with document/page/offset provenance.
- `Generator`: question plus cited context to answer plus citations, or refusal
  when context is insufficient.
- `InsightsProvider`: one-time document summary, key points, and document type,
  separate from the hot chat path.

This is intentionally smaller than a framework-level provider abstraction. The
goal is not arbitrary provider swapping; the goal is to keep the graded surfaces
visible: prompts, ranking, citation assembly, fallback behavior, and eval
results.

Because retrieval is behind one function, experiments can be measured instead of
argued. A hybrid keyword/vector retriever or a structure-aware retriever can be
dropped into the same eval harness and compared against the deterministic
baseline on NDCG@5, MRR, faithfulness, and refusal correctness. That keeps
experimentation aligned with the differentiator: better cited answers, not more
architecture.

## Alternatives Considered

### LangChain / LlamaIndex

Where it would help here: fastest path to working chat, mature loaders for messy
broker documents, and specifically the parent-document retriever pattern. That
matters for policies: you may match on a small chunk like "water damage" but
need the whole clause with its lead-in conditions passed to the LLM, because an
exclusion's meaning lives in its surrounding sub-conditions.

Why this build still hand-rolls the RAG layer: prompts, retrieval internals, and
citation mechanics are exactly what should be inspected and evaluated in this
assignment. The project keeps those in owned code rather than behind retriever
abstractions. It still borrows the parent-context idea: chunk overlap plus
offset-faithful chunks let the system widen context around a hit without adopting
a framework.

### Agentic RAG / LangGraph

Where it would win in insurance: high-value broker workflows are multi-hop.
Coverage gap analysis, for example, requires reading a lease's insurance clause,
finding the policy's limits, checking endorsements that modify them, and
reconciling the result. That is a retrieve, reason, retrieve-again loop rather
than a lookup.

Why not v1: the current question set is single-hop lookup. An agent loop makes
retrieval non-deterministic, makes NDCG/MRR baselines less clear, cannot run
keyless, and adds latency on the hot path. The better first version is a
deterministic `ask()` pipeline whose retrieve step can later become a bounded
loop without rewriting the app.

### PageIndex / Vectorless Reasoning RAG

Where it would win in insurance: long policies are often structurally navigated,
not semantically searched. "Deductible" may appear many times; the operative one
is often in the Declarations. Endorsements cross-reference sections they amend.
An underwriter reads through structure: declarations, insuring agreement,
exclusions, endorsements. Similarity search is weak when the answer is defined
by where it is, not what it resembles.

Why not v1: structural reasoning retrieval itself requires an LLM, which breaks
keyless evaluation. The current sample corpus also does not have the 80-page
policy depth that makes the trade worth it. What the build absorbs now is the
non-agentic part: structure-aware chunking discipline, tables kept whole where
possible, and page provenance preserved.

### Karpathy's LLM-Wiki Pattern

Where it would win in insurance: a broker account is closer to a client file
than a pile of per-query snippets. A wiki layer that maintains an entity page per
insured, with policies, claims history, renewal deltas, and correspondence
summaries, is a better long-run product than query-only document chat.

Why not v1: answers from synthesized pages weaken document/page citation
provenance, and in an E&O-sensitive workflow provenance is the product.
Compilation also needs an LLM at ingest, so keyless mode cannot build it. The
project absorbs the smaller useful piece: a planned insights pass that compiles
per-document summaries and key points at ingest while keeping citations tied to
source documents.

## Running Locally

Start infrastructure:

```powershell
docker compose up -d
```

Install dependencies:

```powershell
uv sync
```

Configure the AI provider:

```powershell
Copy-Item .env.example .env
# Set OPENAI_API_KEY in .env for the full GPT + embeddings path.
```

Run the API:

```powershell
uv run uvicorn app.main:app --reload
```

Run a worker in another shell:

```powershell
uv run rq worker --url redis://localhost:6380/0
```

Open:

- UI: `http://localhost:8000/`
- API docs: `http://localhost:8000/docs`

## Testing

```powershell
uv run pytest
```

The current test suite covers the deterministic foundation and provider seams:
storage, lifecycle transitions, extraction, chunking invariants, upload/dedupe
behavior, the inline processing pipeline, embedding injection, cited
deterministic answers, refusal behavior, AI-generator injection, pgvector hybrid
retrieval/reranking, AI insights generation, and chat history.

## AI Usage

AI-assisted development is documented in `AI_USAGE.md`. That file is maintained
as a running log rather than reconstructed after the fact: what was delegated,
what was reviewed by hand, what bugs tests caught, and what prompt/model
decisions were made.
