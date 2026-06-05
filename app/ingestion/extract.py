"""File-type routing: raw upload bytes → per-page text.

Supported: .pdf (pdfplumber, real page numbers), .docx (python-docx,
single logical page — docx has no page concept until rendered), .txt/.md
(single page). Corrupt files surface as ValueError so the worker can mark
the document `failed` with a useful message instead of crashing opaquely.
"""

import io
from dataclasses import dataclass
from pathlib import Path


class UnsupportedFileTypeError(Exception):
    def __init__(self, extension: str) -> None:
        super().__init__(f"unsupported file type: '{extension}'")
        self.extension = extension


@dataclass(frozen=True)
class PageText:
    page_number: int  # 1-based
    text: str


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def extract_pages(data: bytes, filename: str) -> list[PageText]:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return _extract_pdf(data)
    if ext == ".docx":
        return _extract_docx(data)
    if ext in {".txt", ".md"}:
        return [PageText(page_number=1, text=data.decode("utf-8", errors="replace"))]
    raise UnsupportedFileTypeError(ext)


def _extract_pdf(data: bytes) -> list[PageText]:
    import pdfplumber

    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            return [
                PageText(page_number=i, text=page.extract_text() or "")
                for i, page in enumerate(pdf.pages, start=1)
            ]
    except Exception as exc:
        raise ValueError(f"could not parse PDF: {exc}") from exc


def _extract_docx(data: bytes) -> list[PageText]:
    import docx

    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:
        raise ValueError(f"could not parse DOCX: {exc}") from exc
    text = "\n".join(p.text for p in document.paragraphs)
    return [PageText(page_number=1, text=text)]
