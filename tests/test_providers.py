"""Tests for custom provider profile loading."""

import tomllib
from pathlib import Path

import pytest

from src.llm_transcribe.providers import (
    DEFAULT_PROVIDERS_FILE,
    ResolvedProvider,
    load_profiles,
)


def test_load_profiles_missing_file_returns_empty(temp_dir: Path) -> None:
    assert load_profiles(temp_dir / "nope.toml") == {}


def test_load_profiles_parses_providers_table(temp_dir: Path) -> None:
    path = temp_dir / "providers.toml"
    path.write_text(
        '[providers.rdsec]\n'
        'api_base = "https://x.example/v1"\n'
        'api_key = "k1"\n'
        'default_model = "gemini-3.5-transcribe"\n'
    )

    profiles = load_profiles(path)

    assert profiles["rdsec"]["api_base"] == "https://x.example/v1"
    assert profiles["rdsec"]["api_key"] == "k1"
    assert profiles["rdsec"]["default_model"] == "gemini-3.5-transcribe"


def test_load_profiles_invalid_toml_raises_decode_error(temp_dir: Path) -> None:
    path = temp_dir / "bad.toml"
    path.write_text("not [valid toml")

    with pytest.raises(tomllib.TOMLDecodeError):
        load_profiles(path)


def test_default_path_honors_xdg(monkeypatch: pytest.MonkeyPatch, temp_dir: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(temp_dir))
    monkeypatch.delenv("LLM_TRANSCRIBE_PROVIDERS_FILE", raising=False)

    from src.llm_transcribe.providers import _default_path

    assert _default_path() == temp_dir / "llm-transcribe" / "providers.toml"


def test_env_var_overrides_providers_file(
    monkeypatch: pytest.MonkeyPatch, temp_dir: Path
) -> None:
    path = temp_dir / "custom.toml"
    path.write_text('[providers.envfile]\napi_base = "https://env/v1"\n')
    monkeypatch.setenv("LLM_TRANSCRIBE_PROVIDERS_FILE", str(path))

    profiles = load_profiles(None)

    assert profiles["envfile"]["api_base"] == "https://env/v1"


def test_default_providers_file_is_expanded_string() -> None:
    assert isinstance(DEFAULT_PROVIDERS_FILE, str)
    assert "~" not in DEFAULT_PROVIDERS_FILE


def test_resolved_provider_is_custom() -> None:
    assert ResolvedProvider().is_custom is False
    assert ResolvedProvider(api_base="https://x/v1").is_custom is True
