"""Provider factory: register and build the providers that ship with the application.

``register_all`` populates a ``ProviderRegistry`` with one factory per named provider.
``default_registry()`` (in ``registry.py``) calls this once; tests that need the real
providers call it on their own registry.

Adding a new provider:
1. Create ``providers/<name>.py`` with a ``build(settings) -> LLMProvider`` function.
2. Add its name to the ``_PROVIDERS`` mapping below.
3. Add its name to the ``Literal[...]`` in ``core/config.py`` and to ``.env.example``.
"""

from __future__ import annotations

from app.core.config import Settings
from app.services.generation.providers import (
    extractive as _extractive,
)
from app.services.generation.providers import (
    mock as _mock,
)
from app.services.generation.providers import (
    ollama as _ollama,
)
from app.services.generation.providers import (
    openai_compat as _openai_compat,
)
from app.services.generation.providers.base import LLMProvider
from app.services.generation.registry import ProviderFactory, ProviderRegistry

# name → build function
_PROVIDERS: dict[str, ProviderFactory] = {
    "extractive": _extractive.build,
    "mock": _mock.build,
    "ollama": _ollama.build,
    "openai_compatible": _openai_compat.build,
}


def register_all(registry: ProviderRegistry) -> None:
    """Register every shipped provider with ``registry``."""
    for name, factory in _PROVIDERS.items():
        registry.register(name, factory)


def build_provider(settings: Settings) -> LLMProvider:
    """Build the configured provider without touching the global registry.

    Useful for one-off instantiation in scripts and manual tests.
    """
    reg = ProviderRegistry()
    register_all(reg)
    return reg.build(settings)
