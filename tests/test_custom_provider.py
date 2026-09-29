import pytest

from insight.llm.registry import PROVIDERS, _ensure_custom_registered


class _FakeCustomSettings:
    custom_base_url = "https://example.com/v1"
    custom_api_key = "sk-test"
    custom_models = "x-preview-f-free"
    custom_name = "oxalpha"
    custom_discovery = True


def test_discovery_toggle_respected(monkeypatch):
    settings_on = _FakeCustomSettings()
    monkeypatch.setattr(settings_on, "custom_discovery", True)
    _ensure_custom_registered(settings_on)
    assert "oxalpha" in PROVIDERS
    assert PROVIDERS["oxalpha"].supports_discovery is True
    del PROVIDERS["oxalpha"]

    settings_off = _FakeCustomSettings()
    monkeypatch.setattr(settings_off, "custom_discovery", False)
    _ensure_custom_registered(settings_off)
    assert "oxalpha" in PROVIDERS
    assert PROVIDERS["oxalpha"].supports_discovery is False
    curated = {m for m, _, _ in PROVIDERS["oxalpha"].curated_models}
    assert curated == {"x-preview-f-free"}
    del PROVIDERS["oxalpha"]
