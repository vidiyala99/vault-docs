"""Extraction routing: file bytes + filename → per-page text.

Page numbers are load-bearing — citations surface them — so the PDF path
must preserve 1-based page numbering exactly.
"""

import io

import pytest

from app.ingestion.extract import PageText, UnsupportedFileTypeError, extract_pages


def make_pdf(*page_texts: str) -> bytes:
    """Hand-assemble a minimal valid PDF, one page per string."""
    n = len(page_texts)
    objects: list[bytes] = []
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n))
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode())
    font_ref = 3 + 2 * n
    for i, text in enumerate(page_texts):
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {4 + 2 * i} 0 R "
            f"/Resources << /Font << /F1 {font_ref} 0 R >> >> >>".encode()
        )
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
        objects.append(
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n"
            + stream + b"\nendstream"
        )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(f"{i} 0 obj\n".encode() + body + b"\nendobj\n")
    xref_at = out.tell()
    out.write(f"xref\n0 {len(objects) + 1}\n".encode())
    out.write(b"0000000000 65535 f \n")
    for off in offsets:
        out.write(f"{off:010d} 00000 n \n".encode())
    out.write(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_at}\n%%EOF\n".encode()
    )
    return out.getvalue()


def make_docx(*paragraphs: str) -> bytes:
    import docx

    doc = docx.Document()
    for p in paragraphs:
        doc.add_paragraph(p)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


class TestPlainText:
    def test_txt_is_a_single_page(self):
        pages = extract_pages(b"hello vault", "notes.txt")
        assert pages == [PageText(page_number=1, text="hello vault")]

    def test_markdown_treated_as_plain_text(self):
        pages = extract_pages(b"# Title\n\nBody.", "readme.md")
        assert pages[0].text == "# Title\n\nBody."

    def test_extension_check_is_case_insensitive(self):
        assert extract_pages(b"x", "FILE.TXT")[0].text == "x"


class TestPdf:
    def test_single_page_pdf(self):
        pages = extract_pages(make_pdf("Hello vault document"), "doc.pdf")
        assert len(pages) == 1
        assert pages[0].page_number == 1
        assert "Hello vault document" in pages[0].text

    def test_multi_page_pdf_preserves_page_numbers(self):
        pages = extract_pages(make_pdf("First page", "Second page"), "doc.pdf")
        assert [p.page_number for p in pages] == [1, 2]
        assert "First page" in pages[0].text
        assert "Second page" in pages[1].text


class TestDocx:
    def test_docx_paragraphs_become_one_page(self):
        pages = extract_pages(make_docx("Alpha paragraph.", "Beta paragraph."), "memo.docx")
        assert len(pages) == 1  # docx has no page concept pre-rendering
        assert "Alpha paragraph." in pages[0].text
        assert "Beta paragraph." in pages[0].text


class TestErrors:
    def test_unsupported_extension_raises(self):
        with pytest.raises(UnsupportedFileTypeError, match="exe"):
            extract_pages(b"MZ...", "malware.exe")

    def test_corrupt_pdf_raises_value_error(self):
        with pytest.raises(ValueError):
            extract_pages(b"%PDF-1.4 then garbage", "broken.pdf")
