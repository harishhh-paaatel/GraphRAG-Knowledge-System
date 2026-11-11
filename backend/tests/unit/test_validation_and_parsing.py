import pytest

from app.core.errors import AppError
from app.services.ingestion.errors import ParseError
from app.services.ingestion.normalize import normalize_text
from app.services.ingestion.parsers import parse_file
from app.services.ingestion.validation import (
    detect_file_type,
    sanitize_filename,
    sniff_content,
    validate_declared_mime,
)
from tests.helpers import SAMPLE_MARKDOWN, SAMPLE_TEXT, make_docx, make_pdf


def test_sanitize_filename_strips_paths_and_junk():
    assert sanitize_filename("../../etc/passwd.txt") == "passwd.txt"
    assert sanitize_filename("C:\\Users\\x\\report final.pdf") == "report final.pdf"
    assert sanitize_filename("we<ir>d|name?.md") == "we_ir_d_name_.md"
    assert sanitize_filename("..hidden.txt") == "hidden.txt"
    assert sanitize_filename("\x00\x01") == "document"
    assert sanitize_filename(None) == "document"
    assert len(sanitize_filename("a" * 400 + ".pdf")) <= 255
    assert sanitize_filename("a" * 400 + ".pdf").endswith(".pdf")


def test_detect_file_type():
    assert detect_file_type("A.PDF") == "pdf"
    for bad in ("x.exe", "x", "x.doc", "x.pdf.exe"):
        with pytest.raises(AppError) as e:
            detect_file_type(bad)
        assert e.value.status_code == 415


def test_declared_mime_checks():
    validate_declared_mime("pdf", "application/pdf")
    validate_declared_mime("pdf", "application/octet-stream")
    validate_declared_mime("md", "text/plain; charset=utf-8")
    validate_declared_mime("txt", None)
    with pytest.raises(AppError) as e:
        validate_declared_mime("pdf", "image/png")
    assert e.value.code == "mime_mismatch"


def test_sniffing(tmp_path):
    def write(name, data):
        p = tmp_path / name
        p.write_bytes(data)
        return p

    sniff_content("pdf", write("ok.pdf", make_pdf(["hi"])))
    sniff_content("docx", write("ok.docx", make_docx([("p", "hi")])))
    sniff_content("txt", write("ok.txt", b"hello"))
    for ftype, data in [
        ("pdf", b"not a pdf"),
        ("docx", b"PK\x03\x04 garbage"),
        ("docx", b"plain text"),
        ("txt", b"abc\x00def"),
        ("txt", b""),
    ]:
        with pytest.raises(AppError) as e:
            sniff_content(ftype, write("bad", data))
        assert e.value.status_code == 400


def test_normalize_text():
    raw = "Hello\r\n  world \t  again\u00a0x\x00\n\n\n\n\nNext   para  "
    assert normalize_text(raw) == "Hello\nworld again x\n\nNext para"


def test_parse_pdf_preserves_page_numbers(tmp_path):
    p = tmp_path / "a.pdf"
    p.write_bytes(make_pdf(["First page about Python.", "", "Third page about FastAPI."]))
    parsed = parse_file(p, "pdf")
    assert [s.page_number for s in parsed.segments] == [1, 3]  # blank page 2 skipped
    assert "Python" in parsed.segments[0].text
    assert parsed.metadata["page_count"] == 3
    assert parsed.metadata["blank_pages"] == 1


def test_parse_pdf_without_text_is_an_error(tmp_path):
    p = tmp_path / "blank.pdf"
    p.write_bytes(make_pdf(["", ""]))
    with pytest.raises(ParseError, match="No extractable text"):
        parse_file(p, "pdf")


def test_parse_corrupted_pdf_is_an_error(tmp_path):
    p = tmp_path / "bad.pdf"
    p.write_bytes(b"%PDF-1.7\n" + b"\x00garbage" * 50)
    with pytest.raises(ParseError):
        parse_file(p, "pdf")


def test_parse_docx_sections_and_tables(tmp_path):
    p = tmp_path / "a.docx"
    p.write_bytes(
        make_docx(
            [("p", "Preface text."), ("h1", "Methods"), ("p", "We used FastAPI.")],
            table=[["Tool", "Use"], ["Python", "Backend"]],
            title="My Doc",
        )
    )
    parsed = parse_file(p, "docx")
    assert [s.section for s in parsed.segments] == [None, "Methods"]
    assert "Python | Backend" in parsed.segments[1].text
    assert parsed.metadata["title"] == "My Doc"


def test_parse_empty_docx_is_an_error(tmp_path):
    p = tmp_path / "e.docx"
    p.write_bytes(make_docx([]))
    with pytest.raises(ParseError, match="No extractable text"):
        parse_file(p, "docx")


def test_parse_txt_and_utf8_bom(tmp_path):
    p = tmp_path / "a.txt"
    p.write_bytes(b"\xef\xbb\xbf" + SAMPLE_TEXT.encode())
    assert parse_file(p, "txt").segments[0].text.startswith("Python is")


def test_parse_txt_rejects_non_utf8_and_whitespace_only(tmp_path):
    p = tmp_path / "a.txt"
    p.write_bytes("caf\xe9".encode("latin-1"))
    with pytest.raises(ParseError, match="UTF-8"):
        parse_file(p, "txt")
    p.write_text("  \n\n \t ")
    with pytest.raises(ParseError, match="No extractable text"):
        parse_file(p, "txt")


def test_parse_markdown_sections_ignore_fenced_code(tmp_path):
    p = tmp_path / "a.md"
    p.write_text(SAMPLE_MARKDOWN, encoding="utf-8")
    parsed = parse_file(p, "md")
    assert [s.section for s in parsed.segments] == ["Overview", "Details"]
    assert "# not a heading" in parsed.segments[1].text
