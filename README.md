# LLM Transcribe

LLM Transcribe is a Python CLI tool that leverages state-of-the-art Large Language Models
to convert audio files into richly annotated transcriptions with timestamps,
speaker identification, and emotional context.

Built for developers and technical users who need high-quality transcription with minimal setup,
it supports all major LLM providers (Google Gemini, OpenAI, Anthropic) through a single interface,
handles arbitrarily long audio files—from quick voice notes to multi-hour meetings—and
produces structured output that captures not just what was said but how it was said,
including non-verbal cues like laughter, hesitation, or frustration.

## Installation

We recommend using [uv](https://docs.astral.sh/uv/)
```sh
uv tool install llm-transcribe
```

After installation, the `llm-transcribe` command will be available in your PATH.

## Usage

```sh
# Simplest way, result will be in audio.txt
llm-transcribe audio.wav

# Specify model and output filename
llm-transcribe -m gemini-2.5-flash -o audio.out.txt audio.wav

# Test setup without processing audio
llm-transcribe audio.wav --test

# Add context for better transcription
llm-transcribe -c "Meeting between John and Kate about Q3 revenue projections" audio.wav

# Get help
llm-transcribe --help
```

## Custom providers and models

Use `--api-base` and `--api-key` to send requests to an OpenAI-compatible endpoint:

```sh
llm-transcribe audio.wav -m gemini-3.5-transcribe --api-base https://api.rdsec.trendmicro.com/prod/aiendpoint/v1 --api-key YOUR_KEY
```

You can also save endpoint settings as a named profile in
`~/.config/llm-transcribe/providers.toml`:

```toml
[providers.rdsec]
api_base = "https://api.rdsec.trendmicro.com/prod/aiendpoint/v1"
# api_key = "YOUR_KEY" # optional; if set, this takes precedence over key_env
key_env = "RDSEC_API_KEY"
default_model = "gemini-3.5-transcribe"
```

Set `RDSEC_API_KEY` in your environment, then select the profile. The profile's `default_model` is used because this command omits `-m`:

```sh
export RDSEC_API_KEY=your_api_key_here
llm-transcribe audio.wav --provider rdsec
```

A profile may use `api_key` instead of `key_env`, or include both. If both are set, `api_key` takes precedence. For each setting, CLI flag > environment variable > profile.

| Environment variable | Purpose |
| --- | --- |
| `LLM_TRANSCRIBE_API_BASE` | API base used when `--api-base` is not set |
| `LLM_TRANSCRIBE_API_KEY` | API key used when `--api-key` is not set |
| `LLM_TRANSCRIBE_REASONING_EFFORT` | Reasoning effort used when `--reasoning-effort` is not set |
| `LLM_TRANSCRIBE_PROVIDERS_FILE` | Path to the provider profiles TOML file; otherwise the tool uses `~/.config/llm-transcribe/providers.toml`, or `$XDG_CONFIG_HOME/llm-transcribe/providers.toml` when `XDG_CONFIG_HOME` is set |

Use `--test` to check connectivity, authentication, and model access without transcribing audio:

```sh
llm-transcribe audio.wav --provider rdsec --test
```

The audio path is still required and must point to an existing file; the check itself sends a text-only request. For RDSec, verify that the model accepts audio separately by running a real transcription.

Models missing from LiteLLM's price table show `n/a` in the cost summary. Request counts are still reported.

Some transcription endpoints return recognizable speech as plain text instead of the requested timestamped speaker lines. When the entire response has no parseable timestamped lines, the tool preserves it as one transcript line with no timestamp or speaker label. This fallback does not infer timing or identify speakers. The plain text is written to the normal transcript output rather than being discarded or reported as an empty successful transcription.

## How it Works

1. **Audio Processing**: Audio file is split into 10-minute chunks with 1-minute overlap using a sliding-window approach.
   - Chunks are technically up to 11 minutes long for context preservation
   - Timing: [0:00-10:00], [9:00-20:00], [19:00-30:00], etc.
   - Supports all common audio formats (WAV, MP3, M4A, etc.)

2. **LLM Transcription**: Each chunk is transcribed using modern LLMs instead of traditional speech-to-text methods.
   - Audio converted to base64 and sent with carefully crafted prompts
   - Context from previous chunk's final minute maintains speaker consistency
   - Superior accuracy through understanding context and capturing emotional nuances
   - Output format: timestamped lines with speaker identification and non-verbal cues

3. **Context Management**: Maintains conversation flow across chunks.
   - Previous chunk's final minute provides context for the next chunk
   - Speaker names and roles preserved throughout entire transcription
   - Handles complex timestamp arithmetic (chunk-relative to absolute time)

4. **Production Ready**: Built on modern Python stack with robust error handling.
   - Retry logic for API failures and rate limiting
   - Cost tracking and provider abstraction via LiteLLM
   - Supports all major LLM providers through unified interface

## Output Format

The transcription output includes timestamps and speaker identification:

```
[01:23:45] John: We should start the meeting now.
[01:23:52] Sarah: I agree. Let me pull up the agenda.
[01:24:05] Speaker 1: Can everyone hear me clearly?
[01:24:08] John: Yes, you're coming through fine.
```

### Speaker Identification Priority

1. **Named speakers**: When identifiable by name (John, Sarah, etc.)
2. **Role-based**: When identifiable by role (Manager, Salesperson, etc.)
3. **Numerical fallback**: Unknown speakers labeled as "Speaker 1", "Speaker 2", etc.

## Supported Models

Works with any LLM provider supported by LiteLLM:
- **Gemini**: `gemini-2.5-flash`, `gemini-1.5-pro`
- **OpenAI**: `gpt-4o`, `gpt-4o-mini`
- **Anthropic**: `claude-3-5-sonnet`, `claude-3-5-haiku`
- **And many more**: See [LiteLLM documentation](https://docs.litellm.ai/docs/providers)

## Requirements

- Python 3.13
- FFmpeg (for audio processing)
- API key for your chosen LLM provider

## Configuration

Set your API key as an environment variable:

```sh
# For Gemini
export GOOGLE_API_KEY=your_api_key_here

# For OpenAI
export OPENAI_API_KEY=your_api_key_here

# For Anthropic
export ANTHROPIC_API_KEY=your_api_key_here
```
