"""Tests for custom endpoint routing in the LLM client."""

from unittest.mock import MagicMock, patch

from src.llm_transcribe.llm_client import LLMClient


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


def _fake_response(text: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = text
    return response
