from pathlib import Path

from src.llm_transcribe.llm_client import LLMClient
from src.llm_transcribe.models import Config, TranscriptionJob
from src.llm_transcribe.transcriber import TranscriptionEngine


def test_config_accepts_provider_fields():
    config = Config(
        api_base="https://x/v1",
        api_key="k",
        provider="rdsec",
        reasoning_effort=None,
    )

    assert config.api_base == "https://x/v1"
    assert config.api_key == "k"
    assert config.provider == "rdsec"
    assert config.reasoning_effort is None


def test_engine_passes_provider_settings_to_llm_client():
    engine = TranscriptionEngine(
        Config(),
        model="gemini-3.5-transcribe",
        api_base="https://x/v1",
        api_key="k",
        reasoning_effort="high",
    )

    assert isinstance(engine.llm_client, LLMClient)
    assert engine.llm_client.api_base == "https://x/v1"
    assert engine.llm_client.api_key == "k"
    assert engine.llm_client.reasoning_effort == "high"
    assert engine.llm_client.effective_model == "openai/gemini-3.5-transcribe"


def test_engine_default_provider_is_unchanged():
    engine = TranscriptionEngine(Config(), model="gemini-2.5-flash")

    assert engine.llm_client.api_base is None
    assert engine.llm_client.effective_model == "gemini-2.5-flash"


def test_transcription_job_accepts_provider_name(tmp_path: Path):
    input_file = tmp_path / "input.wav"
    input_file.touch()

    job = TranscriptionJob(
        input_file=input_file,
        output_file=tmp_path / "output.txt",
        model="gemini-2.5-flash",
        config=Config(),
        provider_name="rdsec",
    )

    assert job.provider_name == "rdsec"
