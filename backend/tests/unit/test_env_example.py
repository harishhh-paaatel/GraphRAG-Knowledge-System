"""`.env.example` must stay truthful: every key is a real setting and the defaults parse."""

import re
from pathlib import Path

import pytest

from app.core.config import DEFAULT_EMBEDDING_REVISION, Settings

ENV_EXAMPLE = Path(__file__).resolve().parents[3] / ".env.example"
KEY = re.compile(r"^#?\s*GRAPHRAG_([A-Z_]+)=")


def documented_keys() -> set[str]:
    return {
        m.group(1).lower()
        for line in ENV_EXAMPLE.read_text("utf-8").splitlines()
        if (m := KEY.match(line))
    }


def test_every_documented_variable_is_a_real_setting():
    unknown = documented_keys() - set(Settings.model_fields)
    assert not unknown, f".env.example documents settings that do not exist: {sorted(unknown)}"


def test_every_embedding_vector_and_hybrid_setting_is_documented():
    wanted = {
        name
        for name in Settings.model_fields
        if name.startswith(("embedding_", "vector_", "hybrid_", "llm_", "answer_", "query_"))
    }
    assert wanted <= documented_keys(), sorted(wanted - documented_keys())


def test_the_uncommented_defaults_load_and_match_the_code(monkeypatch, tmp_path):
    for line in ENV_EXAMPLE.read_text("utf-8").splitlines():
        if line.startswith("GRAPHRAG_") and "=" in line:
            name, value = line.split("=", 1)
            monkeypatch.setenv(name, value)
    monkeypatch.setenv("GRAPHRAG_DATA_DIR", str(tmp_path))
    settings = Settings(_env_file=None)
    assert settings.embedding_model_revision == DEFAULT_EMBEDDING_REVISION
    assert settings.embedding_provider == "sentence_transformers"  # the example is for real use
    assert settings.chunk_size == 800 and settings.vector_max_top_k == 50


@pytest.mark.parametrize("secret", ["sk-", "ghp_", "password", "BEGIN PRIVATE KEY"])
def test_no_secret_looking_content(secret):
    assert secret not in ENV_EXAMPLE.read_text("utf-8")
