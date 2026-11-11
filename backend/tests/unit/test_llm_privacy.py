"""Privacy rules for provider URLs, API keys and error text."""

import pytest

from app.core.privacy import (
    MissingApiKey,
    UnsafeProviderUrl,
    check_provider_url,
    is_loopback_host,
    redact,
    resolve_api_key,
    sanitize_message,
)

# ------------------------------------------------------------------ loopback detection


@pytest.mark.parametrize(
    "host",
    [
        "localhost",
        "LOCALHOST",
        "localhost.",
        "api.localhost",
        "127.0.0.1",
        "127.1.2.3",
        "::1",
        "[::1]",
        "0:0:0:0:0:0:0:1",
    ],
)
def test_loopback_hosts_are_recognised(host):
    assert is_loopback_host(host)


@pytest.mark.parametrize(
    "host",
    [
        "example.com",
        "10.0.0.5",
        "192.168.1.10",
        "8.8.8.8",
        "0.0.0.0",
        "::",
        "127.0.0.1.evil.com",
        "localhost.evil.com",
        "evil-localhost",
        "notlocalhost",
        "2130706433",  # 127.0.0.1 written as one integer: treated as a name, hence remote
        "0x7f.0.0.1",
        "",
    ],
)
def test_everything_else_is_treated_as_remote(host):
    assert not is_loopback_host(host)


# ------------------------------------------------------------------------ URL validation


@pytest.mark.parametrize(
    "url, normalised",
    [
        ("http://127.0.0.1:11434", "http://127.0.0.1:11434"),
        ("http://localhost:11434/", "http://localhost:11434"),
        ("http://localhost:8080/v1/", "http://localhost:8080/v1"),
        ("https://localhost/v1", "https://localhost/v1"),
        ("http://[::1]:8000", "http://[::1]:8000"),
    ],
)
def test_loopback_urls_are_accepted_and_normalised(url, normalised):
    assert check_provider_url(url) == normalised


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com",
        "https://api.example.com/v1",
        "http://10.1.2.3:11434",
        "http://0.0.0.0:80",
    ],
)
def test_a_remote_host_is_refused_by_default_and_names_the_setting(url):
    with pytest.raises(UnsafeProviderUrl, match="GRAPHRAG_LLM_ALLOW_REMOTE"):
        check_provider_url(url)


def test_a_remote_host_is_accepted_only_with_the_explicit_opt_in():
    assert (
        check_provider_url("https://llm.internal/v1/", allow_remote=True)
        == "https://llm.internal/v1"
    )


@pytest.mark.parametrize("allow", [False, True])
@pytest.mark.parametrize(
    "url, message",
    [
        ("", "empty"),
        ("   ", "empty"),
        ("ftp://localhost/x", "http:// or https://"),
        ("file:///etc/passwd", "http:// or https://"),
        ("localhost:11434", "http:// or https://"),
        ("//localhost:11434", "http:// or https://"),
        ("http://user:pass@localhost:11434", "credentials"),
        ("http://user@localhost:11434", "credentials"),
        ("http://localhost:11434?key=abc", "query string"),
        ("http://localhost:11434/#frag", "query string or fragment"),
        ("  http://127.0.0.1:1234  ", "whitespace or control"),
        ("http:///nohost", "no host"),
        ("http://localhost:99999", "invalid port"),
        ("http://localhost:abc", "invalid port"),
        ("http://local host:11434", "whitespace"),
        ("http://localhost:11434/\nHost: evil", "whitespace or control"),
        ("http://localhost:11434/\x00", "whitespace or control"),
    ],
)
def test_malformed_or_dangerous_urls_are_refused_even_when_remote_is_allowed(url, message, allow):
    with pytest.raises(UnsafeProviderUrl, match=message):
        check_provider_url(url, allow_remote=allow)


def test_a_non_string_url_is_refused():
    with pytest.raises(UnsafeProviderUrl):
        check_provider_url(None)  # type: ignore[arg-type]


# ----------------------------------------------------------------------------- API keys


def test_the_key_is_read_from_the_named_variable_only():
    env = {"MY_KEY": "  sk-test-123  ", "OTHER": "other-value"}
    assert resolve_api_key("MY_KEY", env) == "sk-test-123"
    assert resolve_api_key(None, env) is None
    assert resolve_api_key("", env) is None


def test_a_configured_but_missing_key_names_the_variable_and_never_a_value():
    with pytest.raises(MissingApiKey, match="MY_KEY is not set"):
        resolve_api_key("MY_KEY", {"OTHER": "secret-value"})
    with pytest.raises(MissingApiKey, match="MY_KEY"):
        resolve_api_key("MY_KEY", {"MY_KEY": "   "})


def test_something_that_is_not_a_variable_name_is_refused_without_echoing_it():
    with pytest.raises(MissingApiKey) as caught:
        resolve_api_key("sk-live-abcdef0123456789", {})
    assert "sk-live" not in str(caught.value)


# ------------------------------------------------------------------------- redaction


def test_secrets_are_replaced_wherever_they_appear():
    text = "401 for key sk-abc123 and again sk-abc123 (not sk-other)"
    assert (
        redact(text, ["sk-abc123"]) == "401 for key [redacted] and again [redacted] (not sk-other)"
    )
    assert redact(text, ["", None]) == text  # type: ignore[list-item]


def test_sanitized_messages_are_single_line_short_and_secret_free():
    raw = "line one\n  line two\twith key sk-abc123 " + "x" * 500
    out = sanitize_message(raw, secrets=["sk-abc123"])
    assert "\n" not in out and "\t" not in out and "sk-abc123" not in out
    assert len(out) <= 200 and out.endswith("…")
    assert sanitize_message("short") == "short"
    assert sanitize_message(ValueError("boom")) == "boom"
