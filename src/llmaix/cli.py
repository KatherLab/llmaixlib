# src/llmaix/cli.py
from __future__ import annotations

import subprocess
from pathlib import Path

import click
from dotenv import load_dotenv

from .__version__ import __version__
from .extract import extract_info
from .preprocess import DocumentPreprocessor


def _get_commit_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode("utf-8")
            .strip()
        )
    except Exception as e:
        return f"n/a ({e})"


def _get_version() -> str:
    return f"{__version__} ({_get_commit_hash()})"


@click.group()
@click.version_option(_get_version(), message="%(prog)s %(version)s")
def main() -> None:
    """LLMAIx CLI: preprocess documents or extract structured info from text."""
    pass


@main.command()
@click.argument("filename", type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False),
    help="Write result to file (defaults to stdout)",
)
@click.option(
    "--mode",
    type=click.Choice(["fast", "advanced"]),
    default="fast",
    show_default=True,
    help="Preprocessing mode",
)
@click.option(
    "--ocr-engine",
    type=click.Choice(["ocrmypdf", "paddleocr", "marker"]),
    default="ocrmypdf",
    show_default=True,
    help="OCR backend for scans",
)
@click.option(
    "--force-ocr", is_flag=True, help="Force OCR even if text layer is present"
)
@click.option(
    "--enable-picture-description", is_flag=True, help="Generate captions for images"
)
@click.option("--enable-formula", is_flag=True, help="Enrich LaTeX formulas")
@click.option("--enable-code", is_flag=True, help="Detect and preserve code blocks")
@click.option(
    "--output-format",
    type=click.Choice(["markdown", "text"]),
    default="markdown",
    show_default=True,
    help="Output format",
)
@click.option(
    "--use-local-vlm",
    is_flag=True,
    help="Use a local vision-language model for captions",
)
@click.option("--local-vlm-repo-id", type=str, help="Repo ID of the local VLM model")
@click.option(
    "--llm-model", type=str, help="Identifier of the LLM model (for remote calls)"
)
@click.option("--base-url", type=str, help="Base URL of the remote API")
@click.option(
    "--api-key", type=str, hide_input=True, help="API key for remote LLM endpoint"
)
@click.option(
    "--vlm-prompt", type=str, help="Prompt used for VLM-driven picture descriptions"
)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
def preprocess(
    filename,
    output,
    mode,
    ocr_engine,
    force_ocr,
    enable_picture_description,
    enable_formula,
    enable_code,
    output_format,
    use_local_vlm,
    local_vlm_repo_id,
    llm_model,
    base_url,
    api_key,
    vlm_prompt,
    verbose,
):
    """
    Extract text or Markdown from a document.

    Automatically uses OCR when needed and supports optional layout enrichment.
    """
    load_dotenv()
    if verbose:
        click.echo(f"[INFO] Preprocessing '{filename}' (mode={mode})", err=True)

    client = None
    if base_url and api_key:

        class _Client:
            pass

        client = _Client()
        client.base_url = base_url
        client.api_key = api_key

    proc = DocumentPreprocessor(
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
    result = proc.process(Path(filename))

    if output:
        Path(output).write_text(result, encoding="utf-8")
        click.echo(f"[OK] Output written to {output}")
    else:
        click.echo(result)


@main.command()
@click.option(
    "--input", "-i", type=str, required=True, help="Free-form text to analyse"
)
@click.option("--llm-model", type=str, help="LLM model identifier (e.g. 'gpt-4')")
@click.option("--base-url", type=str, help="Remote LLM API base URL")
@click.option("--api-key", type=str, hide_input=True, help="API key for LLM API")
@click.option(
    "--json-schema",
    type=str,
    help="JSON schema (file path or literal) for structured validation",
)
@click.option(
    "--pydantic-model", type=str, help="Pydantic model, specified as module.ClassName"
)
@click.option(
    "--temperature",
    type=float,
    default=None,
    show_default=True,
    help="Sampling temperature for LLM request",
)
@click.option(
    "--max-tokens",
    type=int,
    default=None,
    show_default=True,
    help="Max tokens for LLM response",
)
@click.option(
    "--include-full",
    is_flag=True,
    help="Return full completion result instead of parsed content",
)
def extract(
    input,
    llm_model,
    base_url,
    api_key,
    json_schema,
    pydantic_model,
    temperature,
    max_tokens,
    include_full,
):
    """
    Extract structured information from INPUT using an LLM.
    """
    load_dotenv()

    # --- Patch: detect if json_schema is a file, load content
    schema = None
    if json_schema:
        import json as _json
        import os

        if os.path.isfile(json_schema):
            with open(json_schema, "r", encoding="utf-8") as f:
                schema = f.read()
                # Also accept JSON (dict) object if needed:
                try:
                    schema = _json.loads(schema)
                except Exception:
                    pass
        else:
            # Accept as literal string, or already json
            try:
                schema = _json.loads(json_schema)
            except Exception:
                schema = json_schema
    # --- End patch

    result = extract_info(
        prompt=input,
        llm_model=llm_model,
        base_url=base_url,
        api_key=api_key,
        json_schema=schema,
        pydantic_model=_resolve_pydantic(pydantic_model),
        temperature=temperature,
        max_completion_tokens=max_tokens,
        include_full_completion_result=include_full,
    )
    click.echo(result)


def _resolve_pydantic(name: str | None):
    if not name:
        return None
    module_name, class_name = name.rsplit(".", 1)
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


if __name__ == "__main__":
    main()
