"""Tests for custom endpoint routing in the LLM client."""

from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from src.llm_transcribe.llm_client import LLMClient


def test_plain_text_transcription_is_preserved_without_invented_metadata() -> None:
    client = LLMClient(model="gemini-3.5-transcribe")
    response_text = "Good morning. My name is Alex. The meeting begins at 9:00."

    lines = client.parse_transcription_response(response_text, chunk_start_seconds=120.0)

    assert len(lines) == 1
    assert lines[0].text == response_text
    assert lines[0].timestamp is None
    assert lines[0].speaker is None
    assert str(lines[0]) == response_text


def test_multiline_plain_text_is_preserved_as_one_unattributed_transcript() -> None:
    client = LLMClient(model="gemini-3.5-transcribe")
    response_text = "Good morning.\nThe meeting begins at nine."

    lines = client.parse_transcription_response(response_text, chunk_start_seconds=120.0)

    assert len(lines) == 1
    assert lines[0].text == response_text
    assert str(lines[0]) == response_text
    assert lines[0].timestamp is None
    assert lines[0].speaker is None


def test_connection_test_accepts_empty_response_as_reachable() -> None:
    client = LLMClient(model="m1", api_base="https://private.example/v1", api_key="private-key")
    with (
        patch("src.llm_transcribe.llm_client.completion") as mock_completion,
        patch("src.llm_transcribe.llm_client.completion_cost", return_value=0.0),
    ):
        mock_completion.return_value = _fake_response("")

        assert client.test_connection() is True

    assert client.request_count == 1


def test_connection_test_still_rejects_api_errors() -> None:
    client = LLMClient(model="m1")
    with patch(
        "src.llm_transcribe.llm_client.completion",
        side_effect=RuntimeError("unauthorized"),
    ):
        assert client.test_connection() is False


def test_effective_model_prefixes_bare_model_when_custom() -> None:
    client = LLMClient(
        model="gemini-3.5-transcribe",
        api_base="https://x/v1",
        api_key="k",
    )

    assert client.effective_model == "openai/gemini-3.5-transcribe"


def test_effective_model_no_prefix_without_api_base() -> None:
    client = LLMClient(model="gemini-2.5-flash")

    assert client.effective_model == "gemini-2.5-flash"


def test_effective_model_no_prefix_when_model_has_slash_prefix() -> None:
    client = LLMClient(
        model="openai/gpt-4o",
        api_base="https://x/v1",
        api_key="k",
    )

    assert client.effective_model == "openai/gpt-4o"


def test_effective_model_no_prefix_when_model_has_colon_prefix() -> None:
    client = LLMClient(
        model="anthropic:claude-3-5-sonnet",
        api_base="https://x/v1",
        api_key="k",
    )

    assert client.effective_model == "anthropic:claude-3-5-sonnet"


def test_completion_receives_custom_endpoint_key_and_effective_model() -> None:
    client = LLMClient(model="m1", api_base="https://x/v1", api_key="secret-key")
    with (
        patch("src.llm_transcribe.llm_client.completion") as mock_completion,
        patch("src.llm_transcribe.llm_client.completion_cost", return_value=0.0),
    ):
        mock_completion.return_value = _fake_response("ok")
        client._make_llm_call_with_retry([{"role": "user", "content": "hi"}])

    kwargs = mock_completion.call_args.kwargs
    assert kwargs["api_base"] == "https://x/v1"
    assert kwargs["api_key"] == "secret-key"
    assert kwargs["model"] == "openai/m1"


def test_completion_omits_api_fields_for_direct_provider() -> None:
    client = LLMClient(model="gemini-2.5-flash")
    with (
        patch("src.llm_transcribe.llm_client.completion") as mock_completion,
        patch("src.llm_transcribe.llm_client.completion_cost", return_value=0.0),
    ):
        mock_completion.return_value = _fake_response("ok")
        client._make_llm_call_with_retry([{"role": "user", "content": "hi"}])

    kwargs = mock_completion.call_args.kwargs
    assert "api_base" not in kwargs
    assert "api_key" not in kwargs
    assert kwargs["model"] == "gemini-2.5-flash"


def test_completion_omits_unset_api_key_for_custom_endpoint() -> None:
    client = LLMClient(model="m1", api_base="https://x/v1")
    with (
        patch("src.llm_transcribe.llm_client.completion") as mock_completion,
        patch("src.llm_transcribe.llm_client.completion_cost", return_value=0.0),
    ):
        mock_completion.return_value = _fake_response("ok")
        client._make_llm_call_with_retry([{"role": "user", "content": "hi"}])

    kwargs = mock_completion.call_args.kwargs
    assert kwargs["api_base"] == "https://x/v1"
    assert "api_key" not in kwargs


def test_reasoning_effort_is_omitted_by_default() -> None:
    client = LLMClient(model="m1", api_base="https://x/v1", api_key="k")
    with (
        patch("src.llm_transcribe.llm_client.completion") as mock_completion,
        patch("src.llm_transcribe.llm_client.completion_cost", return_value=0.0),
    ):
        mock_completion.return_value = _fake_response("ok")
        client._make_llm_call_with_retry([{"role": "user", "content": "hi"}])

    assert "reasoning_effort" not in mock_completion.call_args.kwargs


def test_reasoning_effort_is_included_when_configured() -> None:
    client = LLMClient(model="m1", reasoning_effort="low")
    with (
        patch("src.llm_transcribe.llm_client.completion") as mock_completion,
        patch("src.llm_transcribe.llm_client.completion_cost", return_value=0.0),
    ):
        mock_completion.return_value = _fake_response("ok")
        client._make_llm_call_with_retry([{"role": "user", "content": "hi"}])

    assert mock_completion.call_args.kwargs["reasoning_effort"] == "low"


def test_cost_lookup_failure_keeps_response_and_counts_request(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = LLMClient(model="unknown-model", api_base="https://secret.example/v1", api_key="secret-key")
    with (
        patch("src.llm_transcribe.llm_client.completion") as mock_completion,
        patch(
            "src.llm_transcribe.llm_client.completion_cost",
            side_effect=[1.25, Exception("unknown model: private detail")],
        ),
    ):
        mock_completion.return_value = _fake_response("ok")
        for _ in range(2):
            text = client._make_llm_call_with_retry([{"role": "user", "content": "hi"}])

    assert text == "ok"
    assert client.request_count == 2
    assert client.total_cost == 1.25
    assert client.cost_known is False
    warnings = [record for record in caplog.records if record.levelname == "WARNING"]
    assert len(warnings) == 1
    assert warnings[0].getMessage() == (
        "Cost tracking unavailable for this model "
        "(unknown to LiteLLM price table); showing n/a"
    )
    assert "private detail" not in warnings[0].getMessage()
    assert "secret-key" not in warnings[0].getMessage()
    assert "https://secret.example/v1" not in warnings[0].getMessage()

    summary = StringIO()
    with redirect_stdout(summary):
        client.print_cost_summary()
    assert "Total cost: n/a" in summary.getvalue()


def _fake_response(text: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = text
    return response
