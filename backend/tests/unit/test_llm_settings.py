"""Answer-generation settings: defaults, bounds, cross-field rules and privacy."""

import pytest
from pydantic import ValidationError

from app.core.config import PROMPT_OVERHEAD_TOKENS, Settings


def make(tmp_path, **kw) -> Settings:
    return Settings(data_dir=tmp_path, _env_file=None, **kw)


def test_the_defaults_need_no_model_no_network_and_no_remote_host(tmp_path):
    s = make(tmp_path)
    assert s.llm_provider == "extractive"  # the default provider is not a model
    assert (s.llm_base_url, s.llm_model, s.llm_api_key_env) == (None, None, None)
    assert s.llm_allow_remote is False and s.llm_fallback_to_extractive is False
    assert s.llm_debug_prompt is False
    assert (s.llm_timeout_seconds, s.llm_retries, s.llm_max_concurrency) == (120.0, 1, 1)
    assert (s.llm_context_window, s.llm_max_answer_tokens, s.answer_context_tokens) == (
        4096,
        512,
        2048,
    )
    assert s.answer_grounding_policy == "withhold" and s.answer_min_top_similarity is None
    assert s.answer_functional_relationships == [] and s.query_history is True
    assert s.answer_chars_per_token == 3.0


def test_the_default_context_budget_fits_the_default_window(tmp_path):
    s = make(tmp_path)
    assert s.answer_context_tokens + s.llm_max_answer_tokens + PROMPT_OVERHEAD_TOKENS <= (
        s.llm_context_window
    )


@pytest.mark.parametrize("provider", ["extractive", "mock"])
def test_model_free_providers_need_no_model_or_url(tmp_path, provider):
    assert make(tmp_path, llm_provider=provider).llm_provider == provider


def test_an_unknown_provider_is_rejected(tmp_path):
    with pytest.raises(ValidationError):
        make(tmp_path, llm_provider="gpt-hosted")


def test_ollama_needs_a_model(tmp_path):
    with pytest.raises(ValidationError, match="llm_model must be set"):
        make(tmp_path, llm_provider="ollama")
    s = make(tmp_path, llm_provider="ollama", llm_model="qwen2.5:1.5b-instruct")
    assert s.llm_base_url is None  # the factory applies Ollama's default later


def test_an_openai_compatible_server_needs_a_model_and_a_url(tmp_path):
    with pytest.raises(ValidationError, match="llm_model must be set"):
        make(tmp_path, llm_provider="openai_compatible", llm_base_url="http://127.0.0.1:8080/v1")
    with pytest.raises(ValidationError, match="llm_base_url must be set"):
        make(tmp_path, llm_provider="openai_compatible", llm_model="m")
    s = make(
        tmp_path,
        llm_provider="openai_compatible",
        llm_model="m",
        llm_base_url="http://127.0.0.1:8080/v1/",
    )
    assert s.llm_base_url == "http://127.0.0.1:8080/v1"  # normalised


# ---------------------------------------------------------------------------- privacy


def test_a_remote_provider_url_is_refused_unless_explicitly_allowed(tmp_path):
    with pytest.raises(ValidationError, match="GRAPHRAG_LLM_ALLOW_REMOTE"):
        make(tmp_path, llm_base_url="https://api.example.com/v1")
    s = make(tmp_path, llm_base_url="https://api.example.com/v1", llm_allow_remote=True)
    assert s.llm_base_url == "https://api.example.com/v1"


@pytest.mark.parametrize(
    "url",
    [
        "http://user:pw@localhost:11434",
        "ftp://localhost",
        "http://localhost?k=v",
        "localhost:11434",
    ],
)
def test_unsafe_url_shapes_are_refused_even_when_remote_is_allowed(tmp_path, url):
    with pytest.raises(ValidationError, match="llm_base_url"):
        make(tmp_path, llm_base_url=url, llm_allow_remote=True)


@pytest.mark.parametrize(
    "value", ["sk-live-abc123", "my key", "KEY=value", "1STARTS_WITH_DIGIT", "a.b"]
)
def test_the_api_key_setting_must_be_a_variable_name_not_a_key(tmp_path, value):
    with pytest.raises(ValidationError, match="environment variable NAME"):
        make(tmp_path, llm_api_key_env=value)


def test_the_key_value_never_enters_the_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_SECRET_KEY", "sk-super-secret-value")
    s = make(tmp_path, llm_api_key_env="MY_SECRET_KEY")
    assert s.llm_api_key_env == "MY_SECRET_KEY"
    for rendering in (repr(s), str(s), s.model_dump_json(), str(s.model_dump())):
        assert "sk-super-secret-value" not in rendering


# ------------------------------------------------------------------------ bounds and rules


@pytest.mark.parametrize(
    "field, bad",
    [
        ("llm_timeout_seconds", 0),
        ("llm_timeout_seconds", 3601),
        ("llm_retries", -1),
        ("llm_retries", 6),
        ("llm_context_window", 511),
        ("llm_max_answer_tokens", 15),
        ("llm_max_concurrency", 0),
        ("llm_max_concurrency", 17),
        ("answer_context_tokens", 127),
        ("answer_chars_per_token", 0.4),
        ("answer_chars_per_token", 10.5),
        ("answer_min_top_similarity", 1.5),
        ("answer_grounding_policy", "ignore"),
    ],
)
def test_out_of_range_values_are_rejected(tmp_path, field, bad):
    with pytest.raises(ValidationError):
        make(tmp_path, **{field: bad})


def test_the_context_budget_must_fit_the_model_window(tmp_path):
    with pytest.raises(ValidationError, match="exceed llm_context_window"):
        make(tmp_path, answer_context_tokens=3500)
    with pytest.raises(ValidationError, match="prompt overhead"):
        make(tmp_path, llm_context_window=2048)
    ok = make(tmp_path, llm_context_window=8192, answer_context_tokens=6000)
    assert ok.answer_context_tokens == 6000
    exact = 4096 - 512 - PROMPT_OVERHEAD_TOKENS
    assert make(tmp_path, answer_context_tokens=exact).answer_context_tokens == exact
    with pytest.raises(ValidationError):
        make(tmp_path, answer_context_tokens=exact + 1)


def test_functional_relationship_types_must_be_known(tmp_path):
    assert make(
        tmp_path, answer_functional_relationships=["LOCATED_IN"]
    ).answer_functional_relationships == ["LOCATED_IN"]
    with pytest.raises(ValidationError, match="unknown types"):
        make(tmp_path, answer_functional_relationships=["LIVES_ON_MARS"])


def test_environment_variables_are_read(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPHRAG_LLM_PROVIDER", "mock")
    monkeypatch.setenv("GRAPHRAG_LLM_RETRIES", "3")
    monkeypatch.setenv("GRAPHRAG_ANSWER_FUNCTIONAL_RELATIONSHIPS", '["LOCATED_IN", "PART_OF"]')
    monkeypatch.setenv("GRAPHRAG_QUERY_HISTORY", "false")
    s = make(tmp_path)
    assert (s.llm_provider, s.llm_retries, s.query_history) == ("mock", 3, False)
    assert s.answer_functional_relationships == ["LOCATED_IN", "PART_OF"]
