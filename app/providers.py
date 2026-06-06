"""AI provider seams for embeddings, answer generation, and insights."""

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

from app.config import get_settings
from app.ingestion.extract import PageText
from app.rag import (
    REFUSAL_TEXT,
    RetrievedChunk,
    answer_from_chunks,
    condense_question,
)

# (role, content) pairs from earlier turns in the chat session.
History = tuple[tuple[str, str], ...]


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


class Generator(Protocol):
    mode: str

    def generate(
        self, question: str, chunks: list[RetrievedChunk], history: History = ()
    ) -> str: ...


@dataclass(frozen=True)
class DocumentInsights:
    summary: str
    key_points: list[str]
    document_type: str


class InsightsProvider(Protocol):
    def generate(self, filename: str, pages: list[PageText]) -> DocumentInsights: ...


class DeterministicGenerator:
    mode = "deterministic"

    def generate(
        self, question: str, chunks: list[RetrievedChunk], history: History = ()
    ) -> str:
        # The condensed question carries the previous turn's terms, which is
        # what the extractive sentence selector needs for follow-ups.
        prior_user = [content for role, content in history if role == "user"]
        return answer_from_chunks(condense_question(question, prior_user), chunks)


class DeterministicInsightsProvider:
    def generate(self, filename: str, pages: list[PageText]) -> DocumentInsights:
        text = " ".join(page.text.strip() for page in pages if page.text.strip())
        sentences = _sentences(text)
        summary = sentences[0] if sentences else f"No extractable text found in {filename}."
        key_points = sentences[:3] or [summary]
        return DocumentInsights(
            summary=summary,
            key_points=key_points,
            document_type=_document_type(filename, text),
        )


@dataclass
class OpenAIEmbedder:
    api_key: str
    base_url: str
    model: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.embeddings.create(model=self.model, input=texts)
        return [list(item.embedding) for item in response.data]


@dataclass
class OpenAIGenerator:
    api_key: str
    base_url: str
    model: str
    max_tokens: int

    mode = "ai"

    def generate(
        self, question: str, chunks: list[RetrievedChunk], history: History = ()
    ) -> str:
        from openai import OpenAI

        context = "\n\n".join(
            f"Source {index + 1}: {item.document.filename} p.{item.chunk.page_number}\n"
            f"{item.chunk.text}"
            for index, item in enumerate(chunks)
        )
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer only from the provided document context. "
                        "Advice questions ('what should I pick', 'which is "
                        "best') are answerable when the documents themselves "
                        "recommend or rank options — quote or paraphrase the "
                        "document's recommendation with attribution. "
                        "If the context contains nothing that answers the "
                        f"question, say exactly: {REFUSAL_TEXT}"
                    ),
                },
                # Earlier turns ride along as real chat messages so the
                # model can resolve follow-ups ("is that per occurrence?").
                *({"role": role, "content": content} for role, content in history),
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                },
            ],
            max_tokens=self.max_tokens,
            temperature=0,
        )
        return response.choices[0].message.content or ""


@dataclass
class OpenAIInsightsProvider:
    api_key: str
    base_url: str
    model: str
    max_tokens: int

    def generate(self, filename: str, pages: list[PageText]) -> DocumentInsights:
        from openai import OpenAI

        text = "\n\n".join(f"Page {page.page_number}: {page.text}" for page in pages)
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract concise insights from the uploaded document. "
                        "Return strict JSON with keys summary, key_points, document_type. "
                        "document_type names what the document actually is, judged from "
                        "its content (e.g. 'Commercial Property Insurance Policy', "
                        "'Loss Run Report', 'Research Paper', 'Meeting Notes'). "
                        "key_points must be an array of short strings."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Filename: {filename}\n\nDocument text:\n{text[:12000]}",
                },
            ],
            max_tokens=self.max_tokens,
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)
        return DocumentInsights(
            summary=str(data.get("summary") or ""),
            key_points=[str(item) for item in data.get("key_points") or []],
            document_type=str(data.get("document_type") or "document"),
        )


@lru_cache
def get_embedder() -> Embedder | None:
    settings = get_settings()
    if not settings.openai_api_key:
        return None
    return OpenAIEmbedder(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.embedding_model,
    )


@lru_cache
def get_generator() -> Generator:
    settings = get_settings()
    if not settings.openai_api_key:
        return DeterministicGenerator()
    return OpenAIGenerator(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.chat_model,
        max_tokens=settings.max_tokens_chat,
    )


@lru_cache
def get_insights_provider() -> InsightsProvider:
    settings = get_settings()
    if not settings.openai_api_key:
        return DeterministicInsightsProvider()
    return OpenAIInsightsProvider(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.insights_model,
        max_tokens=settings.max_tokens_summary,
    )


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def _document_type(filename: str, text: str) -> str:
    haystack = f"{filename} {text}".lower()
    if "loss" in haystack and "run" in haystack:
        return "loss_run"
    if "acord" in haystack or "application" in haystack:
        return "application"
    if "policy" in haystack:
        return "policy"
    return "document"
