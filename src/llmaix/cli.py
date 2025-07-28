# src/llmaix/cli.py
"""
Command‑line interface for LLMAIx.

Highlights
----------
* `preprocess` command now wraps the new `DocumentPreprocessor` class.
* Supports all modern flags: --mode, --ocr-engine, --force-ocr, --enable-picture-description, etc.
* Optional file output via `-o/--output`.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import click
from dotenv import load_dotenv

from .__version__ import __version__
from .extract import extract_info
from .preprocess import DocumentPreprocessor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_commit_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode("utf-8")
            .strip()
        )
    except Exception as e:  # pragma: no cover
        return f"n/a ({e})"


def _get_version() -> str:
    return f"{__version__} ({_get_commit_hash()})"


# ---------------------------------------------------------------------------
# CLI setup
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(_get_version(), message="%(prog)s %(version)s")
def main() -> None:
    """LLMAIx command‑line interface."""
    pass  # noqa: D401


# ---------------------------------------------------------------------------
# Preprocess command
# ---------------------------------------------------------------------------


@main.command()
@click.argument("filename", type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False),
    help="Write result to file instead of stdout",
)
@click.option(
    "--mode",
    type=click.Choice(["fast", "advanced"]),
    default="fast",
    show_default=True,
    help="Extraction mode (fast = lightweight; advanced = Docling enrichment)",
)
@click.option(
    "--ocr-engine",
    type=click.Choice(["ocrmypdf", "paddleocr", "surya"]),
    default="ocrmypdf",
    show_default=True,
    help="OCR backend to use when OCR is needed / forced",
)
@click.option("--force-ocr", is_flag=True, help="Force OCR even if text is detected")
@click.option(
    "--enable-picture-description", is_flag=True, help="Generate captions for images"
)
@click.option("--enable-formula", is_flag=True, help="Enrich LaTeX formulas")
@click.option("--enable-code", is_flag=True, help="Detect code blocks")
@click.option(
    "--output-format",
    type=click.Choice(["markdown", "text"]),
    default="markdown",
    show_default=True,
    help="Output format",
)
# VLM / LLM integration ----------------------------------------------------
@click.option(
    "--use-local-vlm",
    is_flag=True,
    help="Use a local HuggingFace VLM for picture description",
)
@click.option("--local-vlm-repo-id", type=str, help="Repo ID for local VLM model")
@click.option("--llm-model", type=str, help="Remote VLM model name / ID")
@click.option("--base-url", type=str, help="Remote API base URL")
@click.option("--api-key", type=str, help="API key for remote API", hide_input=True)
@click.option("--vlm-prompt", type=str, help="Prompt for VLM extraction")
# misc ---------------------------------------------------------------------
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose diagnostics")
def preprocess(
    filename: str,
    output: str | None,
    mode: str,
    ocr_engine: str,
    force_ocr: bool,
    enable_picture_description: bool,
    enable_formula: bool,
    enable_code: bool,
    output_format: str,
    use_local_vlm: bool,
    local_vlm_repo_id: str | None,
    llm_model: str | None,
    base_url: str | None,
    api_key: str | None,
    verbose: bool,
    vlm_prompt: str | None,
) -> None:
    """
    Extract text/markdown from *FILENAME* and optionally write to *OUTPUT*.

    Examples
    --------
    \b
    # quick extraction
    llmaix preprocess report.pdf

    # enforce OCR
    llmaix preprocess scan.pdf --force-ocr --ocr-engine paddleocr -o scan.md

    # advanced mode with picture captions via a local VLM
    llmaix preprocess brochure.pdf --mode advanced --enable-picture-description \\
        --use-local-vlm --local-vlm-repo-id HuggingFaceTB/SmolVLM-256M-Instruct
    """
    load_dotenv()

    if verbose:
        click.echo(
            f"[INFO] Starting preprocessing for {filename} (mode={mode})", err=True
        )

    client = None
    if base_url and api_key:
        # very lightweight dummy client object
        class _Client:
            pass

        client = _Client()
        client.base_url = base_url
        client.api_key = api_key

    processor = DocumentPreprocessor(
        mode=mode,
        ocr_engine=ocr_engine,
        enable_picture_description=enable_picture_description,
        enable_formula=enable_formula,
        enable_code=enable_code,
        output_format=output_format,
        llm_client=client,
        llm_model=llm_model,
        use_local_vlm=use_local_vlm,
        local_vlm_repo_id=local_vlm_repo_id,
        force_ocr=force_ocr,
        vlm_prompt=vlm_prompt,
    )

    result = processor.process(Path(filename))

    if output:
        Path(output).write_text(result, encoding="utf-8")
        click.echo(f"[OK] Written to {output}")
    else:
        click.echo(result)


# ---------------------------------------------------------------------------
# Extract‑info command (unchanged)
# ---------------------------------------------------------------------------


@main.command()
@click.option("--input", "-i", type=str, help="Input text to analyse")
def extract(input: str | None) -> None:  # noqa: D401
    """Extract structured information from free‑text *INPUT*."""
    load_dotenv()
    result = extract_info(prompt=input)
    click.echo(result)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
