# Product Context — why this is built the way it is

Before writing the RAG/chat layer, I researched the domain this assignment comes
from: AI document intelligence for **commercial insurance brokerages**. This doc
records what that research found and the product decisions it drove.

## The domain

Platforms in this space (e.g. OutMarket AI — "the intelligence layer for
insurance") serve independent and commercial insurance **brokerages**: producers,
account managers, ops teams. Their daily documents: carrier quotes, policies,
endorsements, submissions, applications, ACORD forms, **loss runs**. The jobs to
be done: quote comparison, coverage gap analysis, policy checking, proposal
generation — and notably a vault-style "chat with any document" surface.

What buyers in this market actually pay for:

- **Hours back** — producers spend hours reading dense documents; minutes matter
  on quote turnaround.
- **Error reduction with receipts** — a missed exclusion in a policy review is a
  potential **E&O (errors & omissions) claim**. Marketing in this space leads
  with "source-cited data points" for a reason.
- **Junior leverage** — newer producers handling unfamiliar lines of business
  need answers they can *verify*, not answers they must take on faith.

## What this implies for a document-chat system (our build priorities)

1. **Citations are the product, not a feature.** Every factual claim in an
   answer must be traceable to a document, page, and passage. An uncited answer
   in an E&O world is a liability, not a convenience. (Hence: offset-faithful
   chunks, citations carrying document + page + snippet, and a *faithfulness*
   eval scorer.)
2. **Refusal correctness is the second-most-important behavior.** "That's not in
   your documents" must beat a fluent hallucination every time. A producer who
   gets a confident wrong answer about a coverage limit is worse off than one
   who gets an honest refusal. (Hence: refusal scenarios in the gold eval set.)
3. **Workflows over chatbots / trust through determinism.** The industry framing
   is deterministic, repeatable, *audited* processes — not generative novelty.
   (Hence: typed lifecycle with guarded transitions, append-only event log,
   deterministic fallback mode as a first-class citizen, eval suite with a CI
   regression gate.)
4. **Sample corpus should look like the real desk.** The committed sample
   documents and gold Q&A mirror broker reality: a commercial property policy,
   an ACORD-style application, a loss-run table, and one plain article as the
   out-of-domain control. Eval questions mirror producer questions:
   "What's the deductible?", "Is water damage excluded?", "What were total
   incurred losses in 2024?"

## Architecture patterns ported (rebuilt fresh, not copied)

From a prior commercial-insurance build (Nightline Risk OS):

- **Provider seam** — per-capability resolution (`OPENAI_API_KEY` present →
  OpenAI; missing/error → deterministic generation + local embeddings); any
  provider exception degrades to deterministic rather than failing the request;
  the response always carries `mode` so degradation is visible, never silent.
- **Eval layering** — gold scenarios + scorers (NDCG@5, MRR, faithfulness,
  refusal correctness) → committed `baseline.json` **keyed by provider stack**
  (deterministic baseline reproducible with zero keys, LLM rows tracked
  separately) → CI fails on any scorer regression.
- **Eval philosophy:** the baseline is a regression floor, not a vanity metric —
  misses are documented, never fudged.
