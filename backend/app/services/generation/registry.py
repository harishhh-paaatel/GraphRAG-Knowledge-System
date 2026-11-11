"""Provider registry and the lazily-built runtime used for dependency injection.

Providers register a factory under their configured name. ``LLMRuntime`` builds the configured
provider on first use (not at start-up, so a missing or unfinished provider can never stop the
application from starting), caches it, and closes it at shutdown. Tests inject their own
registry or provider; production code asks the runtime and never constructs a provider itself.
"""

import logging
import threading
from collections.abc import Callable

from app.core.config import Settings
from app.core.privacy import UnsafeProviderUrl, check_provider_url
from app.services.generation.errors import GenerationError, ProviderNotConfigured
from app.services.generation.providers.base import LLMProvider

logger = logging.getLogger(__name__)

ProviderFactory = Callable[[Settings], LLMProvider]


class ProviderRegistry:
    """name -> factory. Registering the same name twice replaces it (used by tests)."""

    def __init__(self) -> None:
        self._factories: dict[str, ProviderFactory] = {}

    def register(self, name: str, factory: ProviderFactory) -> None:
        if not name or not isinstance(name, str):
            raise ValueError("a provider needs a name")
        self._factories[name] = factory

    def names(self) -> list[str]:
        return sorted(self._factories)

    def __contains__(self, name: object) -> bool:
        return name in self._factories

    def build(self, settings: Settings) -> LLMProvider:
        """Create the provider named by ``settings.llm_provider``.

        Re-checks the privacy rule on the configured URL (settings already validate it, but a
        provider can also be built from hand-made settings), and wraps any unexpected factory
        failure in ``ProviderNotConfigured`` so the caller sees one error type.
        """
        name = settings.llm_provider
        factory = self._factories.get(name)
        if factory is None:
            raise ProviderNotConfigured(
                f"the LLM provider {name!r} is not available in this build "
                f"(registered: {', '.join(self.names()) or 'none'})"
            )
        if settings.llm_base_url is not None:
            try:
                check_provider_url(settings.llm_base_url, allow_remote=settings.llm_allow_remote)
            except UnsafeProviderUrl as exc:
                raise ProviderNotConfigured(f"llm_base_url: {exc}") from exc
        try:
            provider = factory(settings)
        except GenerationError:
            raise
        except Exception as exc:
            logger.exception("Building the %s provider failed", name)
            raise ProviderNotConfigured(
                f"the {name!r} provider could not be built ({type(exc).__name__})"
            ) from exc
        return provider


def default_registry() -> ProviderRegistry:
    """The providers that ship with the application (registered in M1)."""
    from app.services.generation.providers.factory import register_all

    reg = ProviderRegistry()
    register_all(reg)
    return reg


class LLMRuntime:
    """Owns the configured provider for the lifetime of the application."""

    def __init__(self, settings: Settings, registry: ProviderRegistry | None = None) -> None:
        self._settings = settings
        self.registry = registry if registry is not None else default_registry()
        self._provider: LLMProvider | None = None
        self._lock = threading.Lock()

    def provider(self) -> LLMProvider:
        """The configured provider, built on first use. A failed build is not cached."""
        with self._lock:
            if self._provider is None:
                self._provider = self.registry.build(self._settings)
            return self._provider

    def extractive(self) -> LLMProvider:
        """The extractive provider, used only for an explicitly configured fallback."""
        factory = self.registry._factories.get("extractive")
        if factory is None:
            raise ProviderNotConfigured("the extractive provider is not available in this build")
        return factory(self._settings)

    def describe(self) -> dict:
        """Non-sensitive state for diagnostics (no URL, no key, no model prompt)."""
        s = self._settings
        return {
            "configured_provider": s.llm_provider,
            "registered": s.llm_provider in self.registry,
            "built": self._provider is not None,
            "remote_allowed": s.llm_allow_remote,
            "fallback_to_extractive": s.llm_fallback_to_extractive,
        }

    def close(self) -> None:
        with self._lock:
            provider, self._provider = self._provider, None
        if provider is not None:
            try:
                provider.close()
            except Exception:
                logger.exception("Closing the LLM provider failed")
