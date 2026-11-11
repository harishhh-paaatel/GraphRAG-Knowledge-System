import logging

from app.core.config import Settings
from app.core.logging import RedactingFilter


def test_default_database_url_is_sqlite_in_data_dir(tmp_path):
    s = Settings(data_dir=tmp_path, _env_file=None)
    assert s.resolved_database_url.startswith("sqlite:///")
    assert s.resolved_database_url.endswith("graphrag.db")


def test_explicit_database_url_wins(tmp_path):
    s = Settings(data_dir=tmp_path, database_url="sqlite:///:memory:", _env_file=None)
    assert s.resolved_database_url == "sqlite:///:memory:"


def test_env_prefix(monkeypatch, tmp_path):
    monkeypatch.setenv("GRAPHRAG_MAX_UPLOAD_MB", "7")
    assert Settings(data_dir=tmp_path, _env_file=None).max_upload_mb == 7


def test_log_redaction():
    rec = logging.LogRecord(
        "x", logging.INFO, "", 0, "calling with api_key=sk-abc123 ok", None, None
    )
    RedactingFilter().filter(rec)
    assert "sk-abc123" not in rec.msg
    assert "api_key=***" in rec.msg


def test_log_redaction_authorization_header():
    rec = logging.LogRecord("x", logging.INFO, "", 0, "Authorization: Bearer", None, None)
    rec2 = logging.LogRecord("x", logging.INFO, "", 0, "password: hunter2", None, None)
    RedactingFilter().filter(rec)
    RedactingFilter().filter(rec2)
    assert "hunter2" not in rec2.msg
