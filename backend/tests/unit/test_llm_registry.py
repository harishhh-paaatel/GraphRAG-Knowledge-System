"""Provider registry and the lazily-built runtime (dependency injection)."""

import threading

import pytest

from app.core.config import Settings
from app.services.generation.errors import ProviderNotConfigured, ProviderUnavailable
from app.services.generation.providers.base import LLMProvider
from app.services.generation.registry import LLMRuntime, ProviderRegistry, default_registry
from tests.llm_helpers import ScriptedProvider


def settings(tmp_path, **kw) -> Settings:
    return Settings(data_dir=tmp_path, _env_file=None, **kw)


def registry_with(factory, name="extractive") -> ProviderRegistry:
    reg = ProviderRegistry()
    reg.register(name, factory)
    return reg


def test_the_scripted_test_provider_satisfies_the_protocol():
    assert isinstance(ScriptedProvider(), LLMProvider)
    assert not isinstance(object(), LLMProvider)


def test_the_shipped_registry_contains_all_four_providers_after_m1(tmp_path):
    reg = default_registry()
    assert set(reg.names()) == {"extractive", "mock", "ollama", "openai_compatible"}
    assert "extractive" in reg and "mock" in reg


def test_registering_and_building(tmp_path):
    provider = ScriptedProvider()
    reg = registry_with(lambda s: provider)
    assert reg.names() == ["extractive"] and "extractive" in reg
    assert reg.build(settings(tmp_path)) is provider


def test_registering_twice_replaces_and_names_are_validated():
    reg = ProviderRegistry()
    reg.register("mock", lambda s: ScriptedProvider(name="first"))
    reg.register("mock", lambda s: ScriptedProvider(name="second"))
    assert reg.names() == ["mock"]
    for bad in ("", None):
        with pytest.raises(ValueError):
            reg.register(bad, lambda s: ScriptedProvider())  # type: ignore[arg-type]


def test_an_unregistered_provider_lists_what_is_registered(tmp_path):
    reg = registry_with(lambda s: ScriptedProvider(), name="mock")
    with pytest.raises(ProviderNotConfigured, match="registered: mock"):
        reg.build(settings(tmp_path))  # the default provider is "extractive"


def test_a_factory_crash_is_wrapped_logged_and_never_leaks_its_message(tmp_path, caplog):
    def boom(s):
        raise RuntimeError("secret detail: sk-abc123")

    with caplog.at_level("ERROR"):
        with pytest.raises(ProviderNotConfigured) as caught:
            registry_with(boom).build(settings(tmp_path))
    assert "RuntimeError" in str(caught.value) and "sk-abc123" not in str(caught.value)
    assert "Building the extractive provider failed" in caplog.text


def test_a_provider_error_raised_by_a_factory_passes_through_unchanged(tmp_path):
    def down(s):
        raise ProviderUnavailable("server not reachable")

    with pytest.raises(ProviderUnavailable):
        registry_with(down).build(settings(tmp_path))


def test_the_url_privacy_rule_is_re_checked_when_a_provider_is_built(tmp_path):
    """Settings validate the URL, but settings can also be hand-made (model_construct)."""
    s = Settings.model_construct(
        **{
            **Settings(data_dir=tmp_path, _env_file=None).model_dump(),
            "llm_base_url": "http://10.0.0.5:11434",
        }
    )
    called = []
    reg = registry_with(lambda st: called.append(1) or ScriptedProvider())
    with pytest.raises(ProviderNotConfigured, match="llm_base_url.*GRAPHRAG_LLM_ALLOW_REMOTE"):
        reg.build(s)
    assert called == []  # the factory was never reached


# ----------------------------------------------------------------------------- runtime


def test_the_runtime_builds_lazily_once_and_caches(tmp_path):
    built = []

    def factory(s):
        built.append(1)
        return ScriptedProvider()

    runtime = LLMRuntime(settings(tmp_path), registry_with(factory))
    assert built == [] and runtime.describe()["built"] is False
    first = runtime.provider()
    assert runtime.provider() is first and built == [1]
    assert runtime.describe()["built"] is True


def test_concurrent_first_use_builds_exactly_one_provider(tmp_path):
    built, barrier = [], threading.Barrier(8)

    def factory(s):
        built.append(1)
        return ScriptedProvider()

    runtime = LLMRuntime(settings(tmp_path), registry_with(factory))
    results = []

    def use():
        barrier.wait(timeout=10)
        results.append(runtime.provider())

    threads = [threading.Thread(target=use) for _ in range(8)]
    [t.start() for t in threads]
    [t.join(10) for t in threads]
    assert len(built) == 1 and len({id(r) for r in results}) == 1


def test_a_failed_build_is_not_cached_so_a_later_fix_takes_effect(tmp_path):
    reg = ProviderRegistry()
    runtime = LLMRuntime(settings(tmp_path), reg)
    with pytest.raises(ProviderNotConfigured):
        runtime.provider()
    reg.register("extractive", lambda s: ScriptedProvider())
    assert runtime.provider().name == "scripted"


def test_close_closes_the_provider_once_and_survives_a_failing_close(tmp_path, caplog):
    provider = ScriptedProvider()
    provider.close_error = RuntimeError("close failed")
    runtime = LLMRuntime(settings(tmp_path), registry_with(lambda s: provider))
    runtime.provider()
    with caplog.at_level("ERROR"):
        runtime.close()
        runtime.close()  # idempotent
    assert provider.closed and "Closing the LLM provider failed" in caplog.text
    LLMRuntime(settings(tmp_path)).close()  # closing a runtime that never built anything


def test_diagnostics_contain_no_url_key_or_model(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_KEY", "sk-secret-value")
    s = settings(
        tmp_path,
        llm_provider="ollama",
        llm_model="private-model-name",
        llm_base_url="http://127.0.0.1:11434",
        llm_api_key_env="MY_KEY",
    )
    info = LLMRuntime(s).describe()
    assert set(info) == {
        "configured_provider",
        "registered",
        "built",
        "remote_allowed",
        "fallback_to_extractive",
    }
    rendered = str(info)
    for forbidden in ("sk-secret-value", "private-model-name", "11434", "MY_KEY"):
        assert forbidden not in rendered


def test_the_extractive_fallback_needs_that_provider_to_exist(tmp_path):
    runtime = LLMRuntime(
        settings(tmp_path, llm_provider="mock"), registry_with(lambda s: ScriptedProvider(), "mock")
    )
    with pytest.raises(ProviderNotConfigured, match="extractive provider is not available"):
        runtime.extractive()
    runtime.registry.register("extractive", lambda s: ScriptedProvider(name="extractive"))
    assert runtime.extractive().name == "extractive"
