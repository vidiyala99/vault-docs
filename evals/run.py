"""Eval runner: ingest the committed corpus, ask the gold questions, score.

Usage:
    uv run python -m evals.run                      # print metrics JSON
    uv run python -m evals.run --output out.json    # also write to file
    uv run python -m evals.run --compare-baseline   # exit 1 on regression

The corpus is ingested through the REAL pipeline (process_document, real
storage, real chunker) into an isolated `vault_eval` database, and answers
go through the same retrieve/generate path the chat API uses — the eval
measures the product, not a parallel reimplementation.

Keyless by default: with no OPENAI_API_KEY the run is fully deterministic
and reproducible, which is what makes the committed baseline meaningful.
Baselines are keyed by mode ("deterministic" vs the model name), so an AI
run never gets compared against the deterministic yardstick.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path
from statistics import mean

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from evals.scorers import citation_coverage, faithfulness, mrr, ndcg_at_k

_EVAL_DIR = Path(__file__).resolve().parent
_CORPUS_DIR = _EVAL_DIR / "corpus"
_GOLD_PATH = _EVAL_DIR / "gold.json"
_BASELINE_PATH = _EVAL_DIR / "baseline.json"

# Regressions smaller than this are float noise, not signal.
_TOLERANCE = 0.005


def build_eval_database():
    """Drop/create `vault_eval` next to the dev database (docker-compose)."""
    from app.config import get_settings
    from app.db import Base

    dev_url = get_settings().database_url
    eval_url = dev_url.rsplit("/", 1)[0] + "/vault_eval"

    admin = create_engine(dev_url, isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(text("DROP DATABASE IF EXISTS vault_eval WITH (FORCE)"))
        conn.execute(text("CREATE DATABASE vault_eval"))
    admin.dispose()

    engine = create_engine(eval_url)
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    from app import models  # noqa: F401  (register tables on Base)

    Base.metadata.create_all(engine)
    return engine


def ingest_corpus(session_factory, storage, embedder) -> dict[str, str]:
    """Upload + process every corpus file through the real pipeline.

    Returns {filename: document_id}.
    """
    from app.lifecycle import DocumentStatus
    from app.models import Document
    from app.services import apply_transition, record_created
    from app.storage.local import content_hash
    from app.tasks.process import process_document

    doc_ids: dict[str, str] = {}
    for path in sorted(_CORPUS_DIR.glob("*.txt")):
        data = path.read_bytes()
        with session_factory() as session:
            storage.save(data)
            doc = Document(
                filename=path.name,
                content_hash=content_hash(data),
                size_bytes=len(data),
                status=DocumentStatus.UPLOADED.value,
            )
            session.add(doc)
            session.flush()
            record_created(session, doc)
            apply_transition(session, doc, DocumentStatus.QUEUED)
            session.commit()
            doc_ids[path.name] = doc.id

        process_document(
            doc_ids[path.name],
            session_factory=session_factory,
            storage=storage,
            embedder=embedder,
            insights_provider=None,  # insights are not under eval here
        )

    with session_factory() as session:
        for filename, doc_id in doc_ids.items():
            status = session.get(Document, doc_id).status
            if status != DocumentStatus.READY.value:
                raise RuntimeError(f"corpus ingest failed: {filename} is {status}")
    return doc_ids


def relevant_chunk_ids(session, question: dict, doc_ids: dict[str, str]) -> set[str]:
    """Ground truth: chunks of the expected doc containing the expected answer."""
    from app.models import DocumentChunk

    chunks = (
        session.query(DocumentChunk)
        .filter(DocumentChunk.document_id == doc_ids[question["expected_doc"]])
        .all()
    )
    return {c.id for c in chunks if question["expected_substring"] in c.text}


def run_question(session, generator, embedder, question_text: str):
    """Mirror the chat route: retrieve, generate (with fallback), cite."""
    from app.providers import DeterministicGenerator
    from app.rag import retrieve_chunks

    query_embedding = None
    if embedder is not None:
        try:
            query_embedding = embedder.embed_documents([question_text])[0]
        except Exception:
            query_embedding = None  # same degradation the chat route applies

    ranked = retrieve_chunks(
        session, question_text, limit=5, query_embedding=query_embedding
    )
    answering = ranked[:1]  # the chat route answers from the top chunk
    try:
        answer = generator.generate(question_text, answering)
        mode = generator.mode
    except Exception:
        fallback = DeterministicGenerator()
        answer = fallback.generate(question_text, answering)
        mode = fallback.mode
    return {
        "ranked_chunk_ids": [item.chunk.id for item in ranked],
        "answer_chunk_texts": [item.chunk.text for item in answering],
        "citations": [item.document.filename for item in answering],
        "answer": answer,
        "mode": mode,
    }


def evaluate(session_factory, generator, embedder, doc_ids: dict[str, str]) -> dict:
    gold = json.loads(_GOLD_PATH.read_text(encoding="utf-8"))
    refusal_text = gold["refusal_text"]

    ndcgs, mrrs, accuracies, faiths, coverages = [], [], [], [], []
    refusal_hits = []
    per_question = []
    modes = set()

    with session_factory() as session:
        for q in gold["questions"]:
            result = run_question(session, generator, embedder, q["question"])
            modes.add(result["mode"])
            detail = {"id": q["id"], "type": q["type"], "answer": result["answer"]}

            if q["type"] == "answerable":
                relevant = relevant_chunk_ids(session, q, doc_ids)
                detail["ndcg@5"] = ndcg_at_k(result["ranked_chunk_ids"], relevant)
                detail["mrr"] = mrr(result["ranked_chunk_ids"], relevant)
                detail["correct"] = q["expected_substring"] in result["answer"]
                fa = faithfulness(
                    result["answer"], result["answer_chunk_texts"], refusal_text
                )
                cov = citation_coverage(
                    result["answer"], result["citations"], refusal_text
                )
                ndcgs.append(detail["ndcg@5"])
                mrrs.append(detail["mrr"])
                accuracies.append(1.0 if detail["correct"] else 0.0)
                if fa is not None:
                    faiths.append(fa)
                if cov is not None:
                    coverages.append(cov)
            else:  # refusal
                refused = (
                    result["answer"].strip() == refusal_text
                    and not result["citations"]
                )
                detail["refused"] = refused
                refusal_hits.append(1.0 if refused else 0.0)

            per_question.append(detail)

    def agg(values):
        return round(mean(values), 4) if values else None

    return {
        "mode": "+".join(sorted(modes)),
        "n_questions": len(per_question),
        "metrics": {
            "retrieval_ndcg@5": agg(ndcgs),
            "retrieval_mrr": agg(mrrs),
            "answer_accuracy": agg(accuracies),
            "faithfulness": agg(faiths),
            "citation_coverage": agg(coverages),
            "refusal_accuracy": agg(refusal_hits),
        },
        "per_question": per_question,
    }


def compare_with_baseline(report: dict) -> list[str]:
    """Regressions vs the committed baseline for this mode. Empty = pass."""
    if not _BASELINE_PATH.exists():
        return [f"no baseline file at {_BASELINE_PATH}"]
    baselines = json.loads(_BASELINE_PATH.read_text(encoding="utf-8"))
    baseline = baselines.get(report["mode"])
    if baseline is None:
        return [f"no baseline recorded for mode '{report['mode']}'"]

    problems = []
    for metric, expected in baseline["metrics"].items():
        actual = report["metrics"].get(metric)
        if expected is None:
            continue
        if actual is None or actual < expected - _TOLERANCE:
            problems.append(f"{metric}: baseline {expected} -> current {actual}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--compare-baseline", action="store_true")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="force keyless mode regardless of OPENAI_API_KEY — this is how "
        "the committed baseline is produced and how CI runs",
    )
    args = parser.parse_args(argv)

    from app.providers import DeterministicGenerator, get_embedder, get_generator
    from app.storage.local import LocalStorage

    engine = build_eval_database()
    session_factory = sessionmaker(
        bind=engine, autoflush=False, expire_on_commit=False
    )
    if args.deterministic:
        embedder, generator = None, DeterministicGenerator()
    else:
        embedder, generator = get_embedder(), get_generator()

    with tempfile.TemporaryDirectory() as tmp:
        storage = LocalStorage(root=Path(tmp) / "blobs")
        doc_ids = ingest_corpus(session_factory, storage, embedder)
        report = evaluate(session_factory, generator, embedder, doc_ids)
    engine.dispose()

    print(json.dumps(report, indent=2))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if args.compare_baseline:
        problems = compare_with_baseline(report)
        if problems:
            print("\nBASELINE REGRESSION:", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            return 1
        print("\nbaseline check: OK", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
