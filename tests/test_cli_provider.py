"""CLI wiring tests for custom provider profiles."""

import struct
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from src.llm_transcribe.cli import app

runner = CliRunner()

PROFILES = '''
[providers.rdsec]
api_base = "https://rdsec.example/v1"
api_key = "profile-key"
default_model = "gemini-3.5-transcribe"
'''


@pytest.fixture
def providers_file(tmp_path: Path) -> Path:
    path = tmp_path / "providers.toml"
    path.write_text(PROFILES)
    return path


@pytest.fixture
def audio_file(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(struct.pack("<h", 0) * 80)
    return path


def _run(args, audio_file, providers_file, monkeypatch):
    monkeypatch.setenv("LLM_TRANSCRIBE_PROVIDERS_FILE", str(providers_file))
    with patch("src.llm_transcribe.cli.TranscriptionEngine") as engine:
        engine.return_value.test_setup.return_value = True
        result = runner.invoke(app, [*args, str(audio_file)])
    return result, engine


def test_provider_flag_resolves_profile(providers_file, audio_file, monkeypatch):
    monkeypatch.delenv("LLM_TRANSCRIBE_API_KEY", raising=False)
    result, engine = _run(["--test", "--provider", "rdsec"], audio_file, providers_file, monkeypatch)

    assert result.exit_code == 0, result.output
    assert engine.call_args.kwargs["api_base"] == "https://rdsec.example/v1"
    assert engine.call_args.kwargs["api_key"] == "profile-key"
    assert engine.call_args.args[0].api_base == "https://rdsec.example/v1"
    assert engine.call_args.args[0].api_key == "profile-key"
    assert engine.call_args.args[0].provider == "rdsec"


def test_builtin_model_is_used_without_profile_model(audio_file):
    with patch("src.llm_transcribe.cli.TranscriptionEngine") as engine:
        engine.return_value.test_setup.return_value = True
        result = runner.invoke(app, ["--test", str(audio_file)])

    assert result.exit_code == 0, result.output
    assert engine.call_args.args[1] == "gemini-2.5-flash"
    assert "Provider: direct" in result.output
    assert "API Base: default" in result.output


def test_profile_model_is_used_when_model_flag_is_absent(providers_file, audio_file, monkeypatch):
    result, engine = _run(["--test", "--provider", "rdsec"], audio_file, providers_file, monkeypatch)

    assert result.exit_code == 0, result.output
    assert engine.call_args.args[1] == "gemini-3.5-transcribe"


def test_cli_model_overrides_profile_default(providers_file, audio_file, monkeypatch):
    result, engine = _run(
        ["--test", "--provider", "rdsec", "--model", "gemini-2.5-flash"],
        audio_file,
        providers_file,
        monkeypatch,
    )

    assert result.exit_code == 0, result.output
    assert engine.call_args.args[1] == "gemini-2.5-flash"


def test_unknown_provider_reports_available_names(providers_file, audio_file, monkeypatch):
    result, _ = _run(["--test", "--provider", "typo"], audio_file, providers_file, monkeypatch)

    assert result.exit_code != 0
    assert "rdsec" in result.output


def test_provider_resolution_error_redacts_api_key(providers_file, audio_file, monkeypatch):
    secret = "invalid-provider-secret"
    monkeypatch.setenv("LLM_TRANSCRIBE_PROVIDERS_FILE", str(providers_file))
    monkeypatch.setenv("LLM_TRANSCRIBE_API_KEY", secret)
    result, _ = _run(["--test", "--provider", secret], audio_file, providers_file, monkeypatch)

    assert result.exit_code != 0
    assert secret not in result.output
    assert "Unknown provider" in result.output


def test_key_and_api_base_are_not_printed(providers_file, audio_file, monkeypatch):
    result, _ = _run(["--test", "--provider", "rdsec"], audio_file, providers_file, monkeypatch)

    assert result.exit_code == 0, result.output
    assert "profile-key" not in result.output
    assert "https://rdsec.example/v1" not in result.output
    assert "Provider: rdsec" in result.output


def test_api_flags_override_environment_and_profile(providers_file, audio_file, monkeypatch):
    monkeypatch.setenv("LLM_TRANSCRIBE_PROVIDERS_FILE", str(providers_file))
    monkeypatch.setenv("LLM_TRANSCRIBE_API_BASE", "https://env.example/v1")
    monkeypatch.setenv("LLM_TRANSCRIBE_API_KEY", "env-key")

    with patch("src.llm_transcribe.cli.TranscriptionEngine") as engine:
        engine.return_value.test_setup.return_value = True
        result = runner.invoke(
            app,
            [
                "--test",
                "--provider",
                "rdsec",
                "--api-base",
                "https://cli.example/v1",
                "--api-key",
                "cli-key",
                str(audio_file),
            ],
        )

    assert result.exit_code == 0, result.output
    assert engine.call_args.kwargs["api_base"] == "https://cli.example/v1"
    assert engine.call_args.kwargs["api_key"] == "cli-key"
    assert "https://cli.example/v1" not in result.output
    assert "cli-key" not in result.output


def test_provider_options_read_environment_aliases(providers_file, audio_file, monkeypatch):
    monkeypatch.setenv("LLM_TRANSCRIBE_API_BASE", "https://env.example/v1")
    monkeypatch.setenv("LLM_TRANSCRIBE_API_KEY", "env-secret")
    monkeypatch.setenv("LLM_TRANSCRIBE_REASONING_EFFORT", "high")
    monkeypatch.setenv("LLM_TRANSCRIBE_PROVIDERS_FILE", str(providers_file))

    with patch("src.llm_transcribe.cli.TranscriptionEngine") as engine:
        engine.return_value.test_setup.return_value = True
        result = runner.invoke(app, ["--test", "--provider", "rdsec", str(audio_file)])

    assert result.exit_code == 0, result.output
    assert engine.call_args.kwargs["api_base"] == "https://env.example/v1"
    assert engine.call_args.kwargs["api_key"] == "env-secret"
    assert engine.call_args.kwargs["reasoning_effort"] == "high"
    assert engine.call_args.args[0].reasoning_effort == "high"
    assert "env-secret" not in result.output
    assert "https://env.example/v1" not in result.output


def test_cli_reasoning_effort_overrides_environment(providers_file, audio_file, monkeypatch):
    monkeypatch.setenv("LLM_TRANSCRIBE_PROVIDERS_FILE", str(providers_file))
    monkeypatch.setenv("LLM_TRANSCRIBE_REASONING_EFFORT", "low")
    with patch("src.llm_transcribe.cli.TranscriptionEngine") as engine:
        engine.return_value.test_setup.return_value = True
        result = runner.invoke(
            app,
            ["--test", "--provider", "rdsec", "--reasoning-effort", "high", str(audio_file)],
        )

    assert result.exit_code == 0, result.output
    assert engine.call_args.kwargs["reasoning_effort"] == "high"


def test_provider_name_is_set_on_created_job(providers_file, audio_file, monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_TRANSCRIBE_PROVIDERS_FILE", str(providers_file))
    job = MagicMock(chunks=[], results=[])
    job.total_duration_seconds = 0
    with patch("src.llm_transcribe.cli.TranscriptionEngine") as engine:
        engine.return_value.create_job.return_value = job
        engine.return_value.prepare_job.return_value = job
        engine.return_value.process_job.return_value = job
        engine.return_value.llm_client.print_cost_summary.return_value = None
        with patch("src.llm_transcribe.cli.output_handler.export_job_results"):
            result = runner.invoke(
                app,
                ["--provider", "rdsec", "--yes", "--output", str(tmp_path / "out.txt"), str(audio_file)],
            )

    assert result.exit_code == 0, result.output
    assert job.provider_name == "rdsec"


def test_engine_error_redacts_api_key_and_base_url(providers_file, audio_file, monkeypatch):
    monkeypatch.setenv("LLM_TRANSCRIBE_PROVIDERS_FILE", str(providers_file))
    with patch("src.llm_transcribe.cli.TranscriptionEngine", side_effect=RuntimeError(
        "failed request to https://rdsec.example/v1 with profile-key"
    )):
        result = runner.invoke(app, ["--test", "--provider", "rdsec", str(audio_file)])

    assert result.exit_code != 0
    assert "profile-key" not in result.output
    assert "https://rdsec.example/v1" not in result.output
    assert "failed request to" in result.output


def test_test_setup_error_redacts_api_key(providers_file, audio_file, monkeypatch):
    monkeypatch.setenv("LLM_TRANSCRIBE_PROVIDERS_FILE", str(providers_file))
    with patch("src.llm_transcribe.cli.TranscriptionEngine") as engine:
        engine.return_value.test_setup.side_effect = RuntimeError(
            "provider setup failed with profile-key"
        )
        result = runner.invoke(app, ["--test", "--provider", "rdsec", str(audio_file)])

    assert result.exit_code != 0
    assert "profile-key" not in result.output
    assert "provider setup failed" in result.output


def test_logs_redact_api_key(providers_file, audio_file, monkeypatch, caplog):
    import logging

    monkeypatch.setenv("LLM_TRANSCRIBE_PROVIDERS_FILE", str(providers_file))
    with patch("src.llm_transcribe.cli.TranscriptionEngine") as engine:
        def emit_sensitive_logs():
            logger = logging.getLogger("llm_transcribe.test")
            logger.error("provider rejected profile-key")
            try:
                raise RuntimeError("request failed with profile-key")
            except RuntimeError:
                logger.exception("provider setup failed")
            return True

        engine.return_value.test_setup.side_effect = emit_sensitive_logs
        result = runner.invoke(app, ["--test", "--provider", "rdsec", "--verbose", str(audio_file)])

    assert result.exit_code == 0, result.output
    assert "profile-key" not in result.output
    assert "profile-key" not in caplog.text
    assert "provider rejected" in caplog.text


def test_setup_error_redacts_api_key_and_base_url(providers_file, audio_file, monkeypatch):
    monkeypatch.setenv("LLM_TRANSCRIBE_PROVIDERS_FILE", str(providers_file))
    with patch("src.llm_transcribe.cli.TranscriptionEngine") as engine:
        engine.return_value.test_setup.side_effect = RuntimeError(
            "setup failed for https://rdsec.example/v1 with profile-key"
        )
        result = runner.invoke(app, ["--test", "--provider", "rdsec", str(audio_file)])

    assert result.exit_code != 0
    assert "profile-key" not in result.output
    assert "https://rdsec.example/v1" not in result.output
    assert "setup failed for" in result.output


def test_job_error_redacts_api_key_and_base_url(providers_file, audio_file, monkeypatch):
    monkeypatch.setenv("LLM_TRANSCRIBE_PROVIDERS_FILE", str(providers_file))
    with patch("src.llm_transcribe.cli.TranscriptionEngine") as engine:
        engine.return_value.create_job.side_effect = RuntimeError(
            "request to https://rdsec.example/v1 failed with profile-key"
        )
        result = runner.invoke(app, ["--provider", "rdsec", "--yes", str(audio_file)])

    assert result.exit_code != 0
    assert "request to" in result.output
    assert "profile-key" not in result.output
    assert "https://rdsec.example/v1" not in result.output


def test_malformed_profiles_error_names_file_without_leaking_keys(audio_file, tmp_path, monkeypatch):
    path = tmp_path / "malformed-providers.toml"
    path.write_text('not [valid toml\napi_key = "must-not-leak"')
    monkeypatch.setenv("LLM_TRANSCRIBE_PROVIDERS_FILE", str(path))

    result = runner.invoke(app, ["--test", "--provider", "rdsec", str(audio_file)])

    assert result.exit_code != 0
    assert str(path).replace(" ", "") in result.output.replace("\n", "").replace(" ", "")
    assert "must-not-leak" not in result.output
