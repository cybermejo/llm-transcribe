"""Tests for custom provider profile loading."""

import tomllib
from pathlib import Path

import pytest

from src.llm_transcribe.providers import (
    DEFAULT_PROVIDERS_FILE,
    ResolvedProvider,
    load_profiles,
    resolve_provider,
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


PROFILES = '''
[providers.rdsec]
api_base = "https://rdsec.example/v1"
key_env = "RDSEC_KEY"
default_model = "gemini-3.5-transcribe"

[providers.local]
api_base = "http://localhost:4000/v1"
api_key = "local-key"
'''


@pytest.fixture
def profiles_file(temp_dir: Path) -> Path:
    path = temp_dir / "providers.toml"
    path.write_text(PROFILES)
    return path


def test_resolve_cli_flag_wins_over_env_and_profile(
    profiles_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLM_TRANSCRIBE_API_BASE", "https://env.example/v1")
    resolved = resolve_provider("local", "https://cli.example/v1", None, profiles_file)

    assert resolved.api_base == "https://cli.example/v1"


def test_resolve_env_used_when_no_cli_flag(
    profiles_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLM_TRANSCRIBE_API_BASE", "https://env.example/v1")
    monkeypatch.setenv("LLM_TRANSCRIBE_API_KEY", "env-key")
    resolved = resolve_provider("local", None, None, profiles_file)

    assert (resolved.api_base, resolved.api_key) == (
        "https://env.example/v1",
        "env-key",
    )


def test_resolve_profile_used_when_no_cli_no_env(
    profiles_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RDSEC_KEY", "from-env-file")
    resolved = resolve_provider("rdsec", None, None, profiles_file)

    assert resolved.api_base == "https://rdsec.example/v1"
    assert resolved.api_key == "from-env-file"
    assert resolved.model == "gemini-3.5-transcribe"


def test_resolve_profile_explicit_key_beats_key_env(
    monkeypatch: pytest.MonkeyPatch, temp_dir: Path
) -> None:
    path = temp_dir / "p2.toml"
    path.write_text(
        '[providers.x]\napi_base = "https://x/v1"\n'
        'api_key = "inline"\nkey_env = "UNUSED"\n'
    )
    monkeypatch.setenv("UNUSED", "env-value")

    resolved = resolve_provider("x", None, None, path)

    assert resolved.api_key == "inline"


def test_resolve_empty_env_string_is_ignored(profiles_file: Path) -> None:
    resolved = resolve_provider(
        "rdsec",
        None,
        None,
        profiles_file,
        environ={"LLM_TRANSCRIBE_API_KEY": ""},
    )

    assert resolved.api_key is None


def test_resolve_cli_key_wins_over_profile(profiles_file: Path) -> None:
    resolved = resolve_provider("local", None, "cli-key", profiles_file)

    assert resolved.api_key == "cli-key"


def test_resolve_unknown_provider_lists_available_profiles(profiles_file: Path) -> None:
    with pytest.raises(ValueError) as error:
        resolve_provider("typo", None, None, profiles_file)

    assert "rdsec" in str(error.value) and "local" in str(error.value)


def test_resolve_no_provider_no_flags_returns_default(profiles_file: Path) -> None:
    resolved = resolve_provider(None, None, None, profiles_file)

    assert resolved.api_base is None and resolved.is_custom is False


def test_resolve_without_provider_uses_flags_and_env_without_loading_profiles(
    temp_dir: Path,
) -> None:
    broken_profiles = temp_dir / "broken.toml"
    broken_profiles.write_text("not valid toml")

    resolved = resolve_provider(
        None,
        "https://cli.example/v1",
        None,
        broken_profiles,
        environ={"LLM_TRANSCRIBE_API_KEY": "env-key"},
    )

    assert resolved.api_base == "https://cli.example/v1"
    assert resolved.api_key == "env-key"


def test_resolve_supplied_environment_is_used_for_profile_key_and_path(
    temp_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = temp_dir / "isolated.toml"
    path.write_text(
        '[providers.x]\napi_base = "https://x/v1"\nkey_env = "PROFILE_KEY"\n'
    )
    monkeypatch.setenv("PROFILE_KEY", "process-environment-key")
    monkeypatch.setenv("LLM_TRANSCRIBE_PROVIDERS_FILE", "/missing/from/process.toml")

    resolved = resolve_provider(
        "x",
        None,
        None,
        None,
        environ={
            "LLM_TRANSCRIBE_PROVIDERS_FILE": str(path),
            "PROFILE_KEY": "supplied-environment-key",
        },
    )

    assert resolved.api_key == "supplied-environment-key"
    assert resolved.api_base == "https://x/v1"
