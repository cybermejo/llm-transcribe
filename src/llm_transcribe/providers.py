"""Load named custom provider profiles from a TOML configuration file."""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel


def _default_path() -> Path:
    """Resolve the profile path using the current environment."""
    if providers_file := os.environ.get("LLM_TRANSCRIBE_PROVIDERS_FILE"):
        return Path(providers_file)
    if config_home := os.environ.get("XDG_CONFIG_HOME"):
        return Path(config_home) / "llm-transcribe" / "providers.toml"
    return Path.home() / ".config" / "llm-transcribe" / "providers.toml"


DEFAULT_PROVIDERS_FILE: str = str(_default_path())


def load_profiles(path: Path | None = None) -> dict[str, dict]:
    """Load custom provider profiles from a TOML file, if it exists."""
    profile_path = path if path is not None else _default_path()
    try:
        with profile_path.open("rb") as profile_file:
            data = tomllib.load(profile_file)
    except FileNotFoundError:
        return {}

    return data.get("providers", {})


def resolve_provider(
    provider_name: str | None,
    api_base: str | None,
    api_key: str | None,
    profiles_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> ResolvedProvider:
    """Resolve provider settings from CLI arguments, environment, and profiles."""
    env = os.environ if environ is None else environ

    if profiles_path is None:
        configured_path = env.get("LLM_TRANSCRIBE_PROVIDERS_FILE")
        config_home = env.get("XDG_CONFIG_HOME")
        if configured_path:
            profiles_path = Path(configured_path)
        elif config_home:
            profiles_path = Path(config_home) / "llm-transcribe" / "providers.toml"
        else:
            profiles_path = (
                Path.home() / ".config" / "llm-transcribe" / "providers.toml"
            )

    profiles = load_profiles(profiles_path) if provider_name is not None else {}
    profile: dict = {}
    if provider_name is not None:
        if provider_name not in profiles:
            available = ", ".join(sorted(profiles))
            raise ValueError(
                f"Unknown provider {provider_name!r}. Available providers: {available}"
            )
        profile = profiles[provider_name]

    env_api_base = env.get("LLM_TRANSCRIBE_API_BASE") or None
    env_api_key = env.get("LLM_TRANSCRIBE_API_KEY") or None
    profile_key = profile.get("api_key")
    if profile_key is None and (key_env := profile.get("key_env")):
        profile_key = env.get(key_env) or None

    return ResolvedProvider(
        api_base=(
            api_base
            if api_base is not None
            else env_api_base if env_api_base is not None else profile.get("api_base")
        ),
        api_key=(
            api_key
            if api_key is not None
            else env_api_key if env_api_key is not None else profile_key
        ),
        model=profile.get("default_model"),
    )


class ResolvedProvider(BaseModel):
    """Provider settings after resolving profile and CLI configuration."""

    api_base: str | None = None
    api_key: str | None = None
    model: str | None = None

    @property
    def is_custom(self) -> bool:
        return self.api_base is not None
