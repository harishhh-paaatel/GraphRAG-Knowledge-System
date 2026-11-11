"""Builders for real sample documents used across tests."""

import io

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def make_pdf(pages: list[str]) -> bytes:
    """Create a real PDF with one text page per entry (empty string -> blank page)."""
    import pymupdf

    pdf = pymupdf.open()
    for text in pages:
        page = pdf.new_page()
        if text:
            page.insert_textbox(pymupdf.Rect(50, 50, 550, 780), text, fontsize=11)
    data = pdf.tobytes()
    pdf.close()
    return data


def make_docx(
    blocks: list[tuple[str, str]], table: list[list[str]] | None = None, title: str | None = None
) -> bytes:
    """Create a real DOCX. ``blocks`` are (kind, text) with kind 'h1' or 'p'."""
    from docx import Document

    doc = Document()
    if title:
        doc.core_properties.title = title
    for kind, text in blocks:
        if kind == "h1":
            doc.add_heading(text, level=1)
        else:
            doc.add_paragraph(text)
    if table:
        t = doc.add_table(rows=len(table), cols=len(table[0]))
        for r, row in enumerate(table):
            for c, val in enumerate(row):
                t.cell(r, c).text = val
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


SAMPLE_TEXT = (
    "Python is a programming language. FastAPI is a web framework written in Python.\n\n"
    "FastAPI was created by Sebastian Ramirez. It uses Pydantic for data validation."
)

SAMPLE_MARKDOWN = (
    "# Overview\n\nGraphRAG combines graphs with retrieval.\n\n"
    "## Details\n\nEntities and relationships are extracted from text.\n\n"
    "```\n# not a heading\n```\n"
)
