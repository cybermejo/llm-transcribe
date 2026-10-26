# Final fix wave report: custom providers

**Date:** 2026-10-26
**Worktree:** `/Users/christianb/Documents/llm-transcribe/.worktrees/custom-providers`
**Status:** Implemented and verified

## Changes

### Plain-text transcription fallback

Some transcription models return speech as ordinary text rather than `[MM:SS] Speaker: text` lines. When the parser finds no structured lines in a non-empty response, it now creates one transcript line with `timestamp=None` and `speaker=None`. Formatting writes only the returned text; it does not invent a timestamp or speaker identity. This is documented in `README.md` and `docs/architecture.md`.

If the model returns structured lines, existing structured parsing remains unchanged. A wholly unparseable response is retained as one line rather than silently discarded. Optional metadata is supported by `TranscriptionLine`; formatting and context extraction handle missing timestamps safely.

### Connectivity-only `--test`

The shared LLM request path accepts an `allow_empty_response` option. Only the connection probe enables it: a successful completion response counts as reachability even if message content is empty. Completion/transport/auth/model errors still follow the existing error path and make the connection test fail.

### Exception log redaction

`_CredentialRedactionFilter` now formats exception details using a local `logging.Formatter`, rather than reading `self.formatter` from a plain `logging.Filter`. It retains the original log message and traceback while redacting the configured key and API base.

## TDD evidence

Regression tests were written before implementation. The initial focused run showed the reported defects:

```text
.venv/bin/pytest -q \
  tests/test_llm_client_provider.py::test_plain_text_transcription_is_preserved_without_invented_metadata \
  tests/test_llm_client_provider.py::test_connection_test_accepts_empty_response_as_reachable \
  tests/test_llm_client_provider.py::test_connection_test_still_rejects_api_errors \
  tests/test_cli_provider.py::test_exception_log_keeps_useful_message_and_redacts_key_and_base

3 failed, 1 passed
```

The failures were: plain text yielded zero lines; empty response made connection test return false after retries; exception filtering replaced useful diagnostics with `Sensitive log message suppressed`. The API-error regression already passed, confirming errors still fail.

After the changes, the focused regression run passed:

```text
5 passed, 9 warnings in 1.60s
```

The focused run covered plain-text retention, multiline plain-text retention, empty connectivity response, API-error failure, and exception-message/key/base redaction.

Full suite after the final implementation:

```text
.venv/bin/pytest -q
135 passed, 9 warnings in 1.03s
```

The warnings are existing Pydantic v1-style validator deprecations in `src/llm_transcribe/models.py` and Pydantic configuration deprecations. No test failures remain.

`git diff --check` completed without whitespace errors.

## Live RDSec results

The API key was read from `/Users/christianb/.pi/agent/models.json` in a Python process and passed only through the child-process environment variable `RDSEC_API_KEY`. It was not printed, put in the temporary TOML profile, or written to this report. The temporary profile and generated speech audio were held in a temporary directory. No home configuration was changed. No direct environment key was used; `LLM_TRANSCRIBE_API_KEY` and `LLM_TRANSCRIBE_API_BASE` were unset in the probe environment.

The live checks used `.venv/bin/llm-transcribe` with the RDSec profile and model `gemini-3.5-transcribe`:

```text
connectivity: exit=0, setup_passed=True
```

The probe received an accepted API response with empty content and reported `Setup test passed!`.

```text
transcription: exit=0, transcription_success=True
transcript_bytes=140
transcript_preserves_speech=True
transcript_content=Good morning. My name is Alex. Today I am checking live audio transcription. The weather is clear and the meeting begins at nine. Thank you.
```

The live transcription retained all recognizable returned speech as plain text. It did not add timestamps or speaker attribution. Cost remained `n/a` for this LiteLLM-unlisted model, as expected.

## Commands and output summary

- Initial regression run: `3 failed, 1 passed` (expected RED; reasons recorded above).
- Focused regression run: `5 passed, 9 warnings`.
- Full suite: `135 passed, 9 warnings`.
- `git diff --check`: no output, exit status 0.
- Live connection probe: exit status 0; `Setup test passed!`.
- Live speech transcription: exit status 0; output file had 140 bytes and preserved speech.

## Changed files

- `README.md`
- `docs/architecture.md`
- `src/llm_transcribe/cli.py`
- `src/llm_transcribe/llm_client.py`
- `src/llm_transcribe/models.py`
- `src/llm_transcribe/output.py`
- `tests/test_cli_provider.py`
- `tests/test_llm_client_provider.py`
- `.superpowers/sdd/2026-10-26-custom-providers/final-fix-report.md`
