from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.models import Document, DocumentChunk, ProcessingJob
from tests.helpers import DOCX_MIME, SAMPLE_MARKDOWN, SAMPLE_TEXT, make_docx, make_pdf

URL = "/api/v1/documents"


@pytest.fixture
def upload(client, app):
    def _upload(name, data, mime="application/octet-stream", wait=True):
        r = client.post(f"{URL}/upload", files={"file": (name, data, mime)})
        if wait:
            assert app.state.worker.drain(timeout=30)
        return r

    return _upload


def _chunks(session, doc_id):
    session.expire_all()
    return (
        session.query(DocumentChunk)
        .filter_by(document_id=doc_id)
        .order_by(DocumentChunk.chunk_index)
        .all()
    )


# ---- happy paths for every format -------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "mime", "builder"),
    [
        ("a.txt", "text/plain", lambda: SAMPLE_TEXT.encode()),
        ("a.md", "text/markdown", lambda: SAMPLE_MARKDOWN.encode()),
        ("a.pdf", "application/pdf", lambda: make_pdf(["Python and FastAPI.", "More text."])),
        ("a.docx", DOCX_MIME, lambda: make_docx([("h1", "Intro"), ("p", "Python and FastAPI.")])),
    ],
)
def test_upload_each_format_end_to_end(client, upload, session, name, mime, builder):
    r = upload(name, builder(), mime, wait=False)
    assert r.status_code == 202
    body = r.json()
    doc_id = body["document"]["id"]
    assert body["job"]["status"] == "pending"  # response reflects state at acceptance
    assert body["retry"] is False
    assert "stored_path" not in body["document"]

    client.app.state.worker.drain(timeout=30)
    status = client.get(f"{URL}/{doc_id}/status").json()
    assert status["processing_status"] == "completed"
    assert status["job"]["status"] == "completed"
    assert status["job"]["stage"] == "done"
    assert status["job"]["started_at"] and status["job"]["completed_at"]

    detail = client.get(f"{URL}/{doc_id}").json()
    assert detail["chunk_count"] == len(_chunks(session, doc_id)) > 0
    assert detail["metadata"]["chunk_count"] == detail["chunk_count"]
    assert detail["file_type"] == name.split(".")[1]


def test_pdf_page_numbers_persisted(upload, session):
    r = upload("p.pdf", make_pdf(["Alpha page one.", "Beta page two.", "Gamma page three."]))
    chunks = _chunks(session, r.json()["document"]["id"])
    assert [c.page_number for c in chunks] == [1, 2, 3]
    assert [c.chunk_index for c in chunks] == [0, 1, 2]


def test_markdown_sections_persisted(upload, session):
    r = upload("m.md", SAMPLE_MARKDOWN.encode(), "text/markdown")
    assert [c.section for c in _chunks(session, r.json()["document"]["id"])] == [
        "Overview",
        "Details",
    ]


def test_chunk_ids_are_deterministic_across_apps(tmp_path):
    ids = []
    for sub in ("a", "b"):
        settings = Settings(data_dir=tmp_path / sub, environment="test", _env_file=None)
        application = create_app(settings)
        from app.models import Base

        Base.metadata.create_all(application.state.engine)
        with TestClient(application) as c:
            r = c.post(
                f"{URL}/upload", files={"file": ("a.txt", SAMPLE_TEXT.encode(), "text/plain")}
            )
            application.state.worker.drain(30)
            with application.state.session_factory() as s:
                ids.append([x.id for x in _chunks(s, r.json()["document"]["id"])])
    assert ids[0] == ids[1] and ids[0]


# ---- rejected uploads --------------------------------------------------------------------


def test_unsupported_extension_415(upload):
    r = upload("evil.exe", b"MZ", wait=False)
    assert r.status_code == 415
    assert r.json()["error"]["code"] == "unsupported_file_type"


def test_mime_mismatch_415(upload):
    assert upload("a.pdf", b"%PDF-1.4", "image/png", wait=False).status_code == 415


def test_empty_file_400(upload):
    r = upload("a.txt", b"", "text/plain", wait=False)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "empty_file"


def test_content_not_matching_extension_400(upload):
    assert upload("a.pdf", b"just text", "application/pdf", wait=False).status_code == 400
    assert upload("a.docx", b"just text", wait=False).status_code == 400
    assert upload("a.txt", b"bin\x00ary", wait=False).status_code == 400


def test_missing_file_field_422(client):
    r = client.post(f"{URL}/upload")
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_error"


def test_oversized_file_413_and_nothing_left_behind(tmp_path):
    settings = Settings(data_dir=tmp_path, max_upload_mb=1, environment="test", _env_file=None)
    application = create_app(settings)
    from app.models import Base

    Base.metadata.create_all(application.state.engine)
    with TestClient(application) as c:
        r = c.post(
            f"{URL}/upload", files={"file": ("big.txt", b"a" * (1024 * 1024 + 1), "text/plain")}
        )
        assert r.status_code == 413
        assert r.json()["error"]["code"] == "file_too_large"
        ok = c.post(f"{URL}/upload", files={"file": ("ok.txt", b"a" * 1024, "text/plain")})
        assert ok.status_code == 202
    assert list(settings.tmp_dir.iterdir()) == []
    assert len(list(settings.documents_dir.iterdir())) == 1  # only the accepted file


def test_path_traversal_filename_is_neutralised(upload, app):
    r = upload("../../../evil.txt", b"hello world", "text/plain")
    assert r.status_code == 202
    doc = r.json()["document"]
    assert doc["filename"] == "evil.txt"
    stored = list(app.state.settings.documents_dir.rglob("*.txt"))
    assert [p.name for p in stored] == ["source.txt"]
    assert stored[0].parent.parent == app.state.settings.documents_dir


# ---- duplicates --------------------------------------------------------------------------


def test_duplicate_upload_409_reports_existing_document(upload, session):
    first = upload("a.txt", SAMPLE_TEXT.encode(), "text/plain")
    dup = upload("renamed.txt", SAMPLE_TEXT.encode(), "text/plain")
    assert dup.status_code == 409
    err = dup.json()["error"]
    assert err["code"] == "duplicate_document"
    assert err["details"]["document_id"] == first.json()["document"]["id"]
    assert session.query(Document).count() == 1


# ---- failures surface, never silent ------------------------------------------------------


def test_corrupted_pdf_fails_visibly(client, upload):
    r = upload("bad.pdf", b"%PDF-1.7\n" + b"\x00junk" * 100, "application/pdf")
    assert r.status_code == 202
    doc_id = r.json()["document"]["id"]
    status = client.get(f"{URL}/{doc_id}/status").json()
    assert status["processing_status"] == "failed"
    assert status["job"]["status"] == "failed"
    assert "corrupted" in status["job"]["error_message"].lower()
    assert client.get(f"{URL}/{doc_id}").json()["chunk_count"] == 0


@pytest.mark.parametrize(
    ("name", "data"),
    [
        ("blank.txt", b"   \n\n  "),
        ("blank.pdf", None),
        ("blank.docx", None),
    ],
)
def test_documents_without_text_fail_visibly(client, upload, name, data):
    data = data or (make_pdf(["", ""]) if name.endswith("pdf") else make_docx([]))
    r = upload(name, data)
    assert r.status_code == 202
    status = client.get(f"{URL}/{r.json()['document']['id']}/status").json()
    assert status["processing_status"] == "failed"
    assert "No extractable text" in status["job"]["error_message"]


def test_unexpected_error_does_not_leak_details(client, upload, monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("secret internal detail /etc/shadow")

    monkeypatch.setattr("app.services.ingestion.pipeline.parse_file", boom)
    r = upload("a.txt", SAMPLE_TEXT.encode(), "text/plain")
    status = client.get(f"{URL}/{r.json()['document']['id']}/status").json()
    assert status["processing_status"] == "failed"
    assert "secret" not in status["job"]["error_message"]
    assert "Internal processing error" in status["job"]["error_message"]


def test_failed_document_can_be_retried_by_reupload(client, upload, session, monkeypatch):
    import app.services.ingestion.pipeline as pipeline

    def broken(*_a):
        raise RuntimeError("x")

    original = pipeline.parse_file
    monkeypatch.setattr(pipeline, "parse_file", broken)
    first = upload("a.txt", SAMPLE_TEXT.encode(), "text/plain")
    doc_id = first.json()["document"]["id"]
    assert client.get(f"{URL}/{doc_id}/status").json()["processing_status"] == "failed"

    monkeypatch.setattr(pipeline, "parse_file", original)
    again = upload("a.txt", SAMPLE_TEXT.encode(), "text/plain")
    assert again.status_code == 202
    assert again.json()["retry"] is True
    assert again.json()["document"]["id"] == doc_id
    assert again.json()["job"]["attempt"] == 2
    status = client.get(f"{URL}/{doc_id}/status").json()
    assert status["processing_status"] == "completed"
    assert status["job"]["attempt"] == 2
    assert session.query(Document).count() == 1
    assert session.query(ProcessingJob).filter_by(document_id=doc_id).count() == 2


# ---- listing / details / status ----------------------------------------------------------


def test_list_pagination_and_status_filter(client, upload):
    for i in range(3):
        upload(f"d{i}.txt", f"document number {i} text".encode(), "text/plain")
    upload("bad.pdf", b"%PDF-1.4 junk" * 20, "application/pdf")

    page = client.get(URL, params={"limit": 2, "offset": 0}).json()
    assert page["total"] == 4 and len(page["items"]) == 2
    rest = client.get(URL, params={"limit": 2, "offset": 2}).json()
    assert {d["id"] for d in page["items"]}.isdisjoint({d["id"] for d in rest["items"]})
    failed = client.get(URL, params={"status": "failed"}).json()
    assert failed["total"] == 1 and failed["items"][0]["filename"] == "bad.pdf"
    assert client.get(URL, params={"status": "bogus"}).status_code == 422
    assert client.get(URL, params={"limit": 0}).status_code == 422


def test_unknown_document_404(client):
    for path in ("/nope", "/nope/status"):
        r = client.get(URL + path)
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "document_not_found"
    assert client.delete(f"{URL}/nope").status_code == 404


# ---- deletion ----------------------------------------------------------------------------


def test_delete_removes_rows_and_files(client, upload, session, app):
    r = upload("a.txt", SAMPLE_TEXT.encode(), "text/plain")
    doc_id = r.json()["document"]["id"]
    folder = app.state.settings.documents_dir / doc_id
    assert folder.is_dir()

    assert client.delete(f"{URL}/{doc_id}").status_code == 204
    session.expire_all()
    assert session.query(Document).count() == 0
    assert session.query(DocumentChunk).count() == 0
    assert session.query(ProcessingJob).count() == 0
    assert not folder.exists()
    assert client.get(f"{URL}/{doc_id}").status_code == 404


def test_delete_blocked_while_processing(client, session, upload):
    r = upload("a.txt", SAMPLE_TEXT.encode(), "text/plain")
    doc_id = r.json()["document"]["id"]
    session.get(Document, doc_id).processing_status = "processing"
    session.commit()
    d = client.delete(f"{URL}/{doc_id}")
    assert d.status_code == 409
    assert d.json()["error"]["code"] == "document_busy"
    assert Path(session.get(Document, doc_id).stored_path).exists()


def test_reupload_allowed_after_delete(client, upload):
    first = upload("a.txt", SAMPLE_TEXT.encode(), "text/plain")
    client.delete(f"{URL}/{first.json()['document']['id']}")
    assert upload("a.txt", SAMPLE_TEXT.encode(), "text/plain").status_code == 202
