"""Load named custom provider profiles from a TOML configuration file."""

import os
import tomllib
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


class ResolvedProvider(BaseModel):
    """Provider settings after resolving profile and CLI configuration."""

    api_base: str | None = None
    api_key: str | None = None
    model: str | None = None

    @property
    def is_custom(self) -> bool:
        return self.api_base is not None
