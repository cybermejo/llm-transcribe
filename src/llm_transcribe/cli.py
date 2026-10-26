"""Command line interface for llm-transcribe."""

import logging
import os
import tomllib
from pathlib import Path
from typing import Optional

# Disable LiteLLM logging to prevent audio data leakage in verbose mode
import litellm
litellm.set_verbose = False

# Disable the main LiteLLM logger that outputs sensitive data
logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.prompt import Confirm

from .models import Config
from .output import OutputHandler
from .providers import resolve_provider
from .transcriber import TranscriptionEngine

# Create console for rich output
console = Console()


class _CredentialRedactionFilter(logging.Filter):
    """Redact credentials from log records before handlers emit them."""

    def __init__(self) -> None:
        super().__init__()
        self.secrets: tuple[str, ...] = ()

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
            if record.exc_info:
                formatter = self.formatter or logging.Formatter()
                message = "\n".join(
                    (message, formatter.formatException(record.exc_info))
                )
                record.exc_info = None
                record.exc_text = None
            for secret in self.secrets:
                message = message.replace(secret, "[redacted]")
            record.msg = message
            record.args = ()
        except Exception:
            record.msg = "Sensitive log message suppressed"
            record.args = ()
            record.exc_info = None
            record.exc_text = None
        return True


_credential_redaction_filter = _CredentialRedactionFilter()

# Create Typer app
app = typer.Typer(
    name="llm-transcribe",
    help="Audio transcription using Large Language Models",
    add_completion=False,
)

# Global variables
config = Config()
output_handler = OutputHandler()


def setup_logging(verbose: bool = False):
    """Setup logging with rich handler."""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Configure logging
    handler = RichHandler(console=console, show_path=False)
    handler.addFilter(_credential_redaction_filter)
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[handler],
    )
    
    # Prevent LiteLLM from logging sensitive audio data in verbose mode
    logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
    
    # Reduce noise from other external libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("pydub").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)


def progress_callback(current: int, total: int, message: str):
    """Callback for progress updates."""
    # This will be used by the progress bar context
    pass


def _safe_error_message(
    error: Exception, api_key: str | None, api_base: str | None
) -> str:
    """Hide provider credentials and endpoint details from user-facing errors."""
    message = str(error)
    for secret in (api_key, api_base):
        if secret:
            message = message.replace(secret, "[redacted]")
    return message


@app.command()
def main(
    audio_file: Path = typer.Argument(
        ...,
        help="Path to the audio file to transcribe",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    model: Optional[str] = typer.Option(
        None,
        "-m", "--model",
        help="LLM model to use for transcription"
    ),
    api_base: Optional[str] = typer.Option(
        None,
        "--api-base",
        envvar="LLM_TRANSCRIBE_API_BASE",
        help="OpenAI-compatible API base URL",
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        envvar="LLM_TRANSCRIBE_API_KEY",
        help="API key for the selected provider",
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        help="Named provider profile from providers.toml",
    ),
    reasoning_effort: Optional[str] = typer.Option(
        None,
        "--reasoning-effort",
        envvar="LLM_TRANSCRIBE_REASONING_EFFORT",
        help="Reasoning effort passed to the model (for example: low, medium, high)",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "-o", "--output",
        help="Output file path (default: input_file.txt)"
    ),
    chunk_duration: int = typer.Option(
        10,
        "--chunk-duration",
        help="Duration of each audio chunk in minutes"
    ),
    overlap_duration: int = typer.Option(
        1,
        "--overlap-duration",
        help="Overlap duration between chunks in minutes"
    ),
    export_formats: Optional[str] = typer.Option(
        None,
        "--export",
        help="Export formats: txt,json,report (comma-separated)"
    ),
    verbose: bool = typer.Option(
        False,
        "-v", "--verbose",
        help="Enable verbose logging"
    ),
    test: bool = typer.Option(
        False,
        "--test",
        help="Test setup and exit"
    ),
    yes: bool = typer.Option(
        False,
        "-y", "--yes",
        help="Skip confirmation prompts"
    ),
    context: Optional[str] = typer.Option(
        None,
        "-c", "--context",
        help="Extra context about the meeting (e.g., 'Meeting between John and Kate about Q3 revenue projections')"
    ),
    version: bool = typer.Option(
        False,
        "--version",
        help="Show version information"
    )
):
    """Transcribe audio files using Large Language Models."""
    
    # Handle version flag
    if version:
        from . import __version__
        console.print(f"llm-transcribe v{__version__}")
        raise typer.Exit()
    
    # Setup logging
    setup_logging(verbose)

    # Resolve profile and environment settings before selecting the model.
    try:
        resolved = resolve_provider(provider, api_base, api_key)
    except tomllib.TOMLDecodeError:
        profile_path = os.environ.get("LLM_TRANSCRIBE_PROVIDERS_FILE")
        if profile_path is None and os.environ.get("XDG_CONFIG_HOME"):
            profile_path = str(
                Path(os.environ["XDG_CONFIG_HOME"])
                / "llm-transcribe"
                / "providers.toml"
            )
        filename = profile_path or str(
            Path.home() / ".config" / "llm-transcribe" / "providers.toml"
        )
        console.print(f"[red]Error:[/red] Invalid provider TOML in {filename}")
        raise typer.Exit(1)
    except ValueError as e:
        message = _safe_error_message(
            e,
            api_key or os.environ.get("LLM_TRANSCRIBE_API_KEY"),
            api_base or os.environ.get("LLM_TRANSCRIBE_API_BASE"),
        )
        console.print(f"[red]Error:[/red] {message}")
        raise typer.Exit(1)

    model = model or resolved.model or "gemini-2.5-flash"
    config.api_base = resolved.api_base
    config.api_key = resolved.api_key
    config.provider = provider
    config.reasoning_effort = reasoning_effort
    _credential_redaction_filter.secrets = tuple(
        secret for secret in (resolved.api_key, resolved.api_base) if secret
    )
    for handler in logging.getLogger().handlers:
        handler.addFilter(_credential_redaction_filter)
    # Update config with command line options
    config.chunk_duration_minutes = chunk_duration
    config.overlap_duration_minutes = overlap_duration
    
    # Determine output file
    if output is None:
        output = audio_file.with_suffix('.txt')
    
    # Validate audio file extension
    file_extension = audio_file.suffix.lower().lstrip('.')
    if file_extension not in config.supported_formats:
        console.print(f"[red]Error:[/red] Unsupported audio format: {file_extension}")
        console.print(f"Supported formats: {', '.join(config.supported_formats)}")
        raise typer.Exit(1)
    
    # Parse export formats
    export_format_list = ['txt']  # Default
    if export_formats:
        export_format_list = [fmt.strip() for fmt in export_formats.split(',')]
    
    # Show configuration
    console.print(f"[blue]Input:[/blue] {audio_file}")
    console.print(f"[blue]Output:[/blue] {output}")
    console.print(f"[blue]Model:[/blue] {model}")
    console.print(f"[blue]Provider:[/blue] {provider or 'direct'}")
    console.print(f"[blue]API Base:[/blue] {'custom' if resolved.api_base else 'default'}")
    console.print(f"[blue]Chunk Duration:[/blue] {chunk_duration} minutes")
    console.print(f"[blue]Overlap Duration:[/blue] {overlap_duration} minutes")
    console.print(f"[blue]Export Formats:[/blue] {', '.join(export_format_list)}")
    if context:
        console.print(f"[blue]Context:[/blue] {context}")
    
    # Create transcription engine
    try:
        engine = TranscriptionEngine(
            config,
            model,
            api_base=resolved.api_base,
            api_key=resolved.api_key,
            reasoning_effort=reasoning_effort,
        )
    except Exception as e:
        message = _safe_error_message(e, resolved.api_key, resolved.api_base)
        console.print(f"[red]Error creating transcription engine:[/red] {message}")
        raise typer.Exit(1)
    
    # Test setup if requested
    if test:
        console.print("[yellow]Testing setup...[/yellow]")
        try:
            setup_passed = engine.test_setup()
        except Exception as e:
            message = _safe_error_message(e, resolved.api_key, resolved.api_base)
            console.print(f"[red]Setup test failed:[/red] {message}")
            raise typer.Exit(1)
        if setup_passed:
            console.print("[green]Setup test passed![/green]")
            raise typer.Exit(0)
        else:
            console.print("[red]Setup test failed![/red]")
            raise typer.Exit(1)
    
    # Check if output file exists
    if output.exists() and not yes:
        if not Confirm.ask(f"Output file {output} already exists. Overwrite?"):
            console.print("[yellow]Aborted.[/yellow]")
            raise typer.Exit(0)
    
    # Create and prepare job
    try:
        console.print("[yellow]Preparing transcription job...[/yellow]")
        job = engine.create_job(audio_file, output, model, context)
        job.provider_name = provider
        job = engine.prepare_job(job)
        
        console.print(f"[green]Job prepared:[/green] {len(job.chunks)} chunks to process")
        
        # Estimate processing time
        audio_duration = job.chunks[-1].end_time_seconds if job.chunks else 0
        estimated_time = audio_duration * 0.3  # Rough estimate: 30% of audio duration
        console.print(f"[blue]Estimated processing time:[/blue] {estimated_time:.0f} seconds")
        
    except Exception as e:
        message = _safe_error_message(e, resolved.api_key, resolved.api_base)
        console.print(f"[red]Error preparing job:[/red] {message}")
        raise typer.Exit(1)
    
    # Process job with progress bar
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            
            task = progress.add_task("Transcribing...", total=len(job.chunks))
            
            def update_progress(current: int, total: int, message: str):
                progress.update(task, completed=current, description=message)
            
            # Process the job
            job = engine.process_job(job, update_progress)
            
            # Complete the progress bar
            progress.update(task, completed=len(job.chunks), description="Transcription complete!")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Transcription interrupted by user.[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        message = _safe_error_message(e, resolved.api_key, resolved.api_base)
        console.print(f"\n[red]Error during transcription:[/red] {message}")
        raise typer.Exit(1)
    
    # Export results
    try:
        console.print("[yellow]Exporting results...[/yellow]")
        output_handler.export_job_results(job, export_format_list)
        
        # Show completion message
        console.print(f"[green]Transcription completed successfully![/green]")
        console.print(f"[blue]Output written to:[/blue] {output}")
        
        # Show summary
        total_lines = sum(len(result.lines) for result in job.results)
        console.print(f"[blue]Total lines:[/blue] {total_lines}")
        console.print(f"[blue]Processing time:[/blue] {job.total_duration_seconds:.2f} seconds")
        
        # Show report file if generated
        if 'report' in export_format_list:
            report_file = output.with_suffix('.report.txt')
            console.print(f"[blue]Summary report:[/blue] {report_file}")
        
        # Show cost summary
        engine.llm_client.print_cost_summary()
        
    except Exception as e:
        message = _safe_error_message(e, resolved.api_key, resolved.api_base)
        console.print(f"[red]Error exporting results:[/red] {message}")
        raise typer.Exit(1)




if __name__ == "__main__":
    app()