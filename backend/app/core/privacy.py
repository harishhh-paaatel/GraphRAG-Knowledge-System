"""Privacy rules for talking to an LLM provider (stdlib only, no application imports).

Retrieved document text is sent to the configured provider, so the default must keep it on the
machine. A provider URL therefore has to be loopback unless the operator explicitly allows a
remote host, and secrets must never reach logs or API responses.

* URLs come only from configuration, never from a request (enforced by the API schema);
* no DNS lookups happen here: a name other than ``localhost`` / ``*.localhost`` is treated as
  remote, which is the safe direction (``127.0.0.1.example.com`` is not loopback);
* the API key is never stored in settings: settings hold only the NAME of an environment
  variable, and this module reads the value at the moment a provider is built.
"""

import ipaddress
import os
import re
from collections.abc import Iterable, Mapping
from urllib.parse import urlsplit

_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_FORBIDDEN = re.compile(r"[\s\x00-\x1f\x7f]")
MAX_MESSAGE_CHARS = 200


class UnsafeProviderUrl(ValueError):
    """The provider URL is malformed or would send document text off this machine."""


class MissingApiKey(LookupError):
    """An API-key environment variable is configured but not set."""


def is_loopback_host(host: str) -> bool:
    """True for ``localhost``, ``*.localhost`` and loopback IP literals (127.0.0.0/8, ::1)."""
    host = host.strip("[]").lower().rstrip(".")
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False  # a name, or an unusual spelling of an address: treat as remote


def check_provider_url(url: str, *, allow_remote: bool = False) -> str:
    """Validate a provider base URL and return it normalised (no trailing slash).

    Raises ``UnsafeProviderUrl`` for a bad scheme, embedded credentials, a query string or
    fragment, whitespace/control characters, a missing host, or a non-loopback host when
    ``allow_remote`` is false.
    """
    if not isinstance(url, str) or not url.strip():
        raise UnsafeProviderUrl("the provider URL is empty")
    if _FORBIDDEN.search(url):
        raise UnsafeProviderUrl("the provider URL contains whitespace or control characters")
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise UnsafeProviderUrl("the provider URL must start with http:// or https://")
    if parts.username is not None or parts.password is not None:
        raise UnsafeProviderUrl("credentials must not be embedded in the provider URL")
    if parts.query or parts.fragment:
        raise UnsafeProviderUrl("the provider URL must not carry a query string or fragment")
    host = parts.hostname
    if not host:
        raise UnsafeProviderUrl("the provider URL has no host")
    try:
        parts.port  # noqa: B018 - raises ValueError for an out-of-range or non-numeric port
    except ValueError as exc:
        raise UnsafeProviderUrl("the provider URL has an invalid port") from exc
    if not allow_remote and not is_loopback_host(host):
        raise UnsafeProviderUrl(
            f"the provider host {host!r} is not loopback: document text would leave this "
            "machine. Set GRAPHRAG_LLM_ALLOW_REMOTE=true to allow it deliberately"
        )
    return url.strip().rstrip("/")


def valid_env_name(name: str) -> bool:
    return bool(_ENV_NAME.fullmatch(name))


def resolve_api_key(env_name: str | None, environ: Mapping[str, str] | None = None) -> str | None:
    """The key held in the environment variable ``env_name`` (None if no name is configured).

    Raises ``MissingApiKey`` (naming the variable, never any value) if a name is configured
    but the variable is unset or empty.
    """
    if not env_name:
        return None
    if not valid_env_name(env_name):
        raise MissingApiKey("the API-key variable name is not a valid environment variable name")
    value = (environ if environ is not None else os.environ).get(env_name, "")
    if not value.strip():
        raise MissingApiKey(f"the environment variable {env_name} is not set")
    return value.strip()


def redact(text: str, secrets: Iterable[str]) -> str:
    """Replace every occurrence of a secret in ``text``."""
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[redacted]")
    return text


def sanitize_message(
    text: object, *, secrets: Iterable[str] = (), max_chars: int = MAX_MESSAGE_CHARS
) -> str:
    """A short, single-line, secret-free rendering of an error message or response excerpt."""
    cleaned = " ".join(str(text).split())
    cleaned = redact(cleaned, secrets)
    return cleaned if len(cleaned) <= max_chars else cleaned[: max_chars - 1] + "…"
