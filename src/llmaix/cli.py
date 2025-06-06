# src/llmaix/cli.py
import subprocess

import click
from dotenv import load_dotenv

from .__version__ import __version__
from .preprocess import preprocess_file
from .extract import extract_info


def get_commit_hash():
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode("utf-8")
            .strip()
        )
    except Exception as e:
        return "Could not retrieve commit hash: " + str(e)


def get_version():
    commit_hash = get_commit_hash()
    return f"{__version__} ({commit_hash})"


@click.group()
@click.version_option(get_version(), message="%(prog)s %(version)s")
def main():
    """LLMAIx CLI"""
    pass


@main.command()
@click.argument("filename", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), help="Output file")
@click.option(
    "--pdf-backend",
    type=click.Choice(["pymupdf4llm", "markitdown", "ocr_backend", "docling"]),
    default="markitdown",
    help="PDF backend",
)
@click.option(
    "--ocr-backend",
    type=click.Choice(["ocrmypdf", "surya-ocr", "doclingvlm"]),
    default="ocrmypdf",
    help="OCR backend to use",
)
@click.option("--use-ocr", is_flag=True, help="Use OCR for preprocessing")
@click.option("--force-ocr", is_flag=True, help="Force OCR for preprocessing")
@click.option(
    "--ocr-languages",
    multiple=True,
    help="Languages for OCR. Currently tesseract only.",
)
@click.option("--llm-model", type=str, help="LLM model to use for preprocessing")
@click.option("--base-url", type=str, help="Base URL for the API")
@click.option("--api-key", type=str, help="API key for authentication", hide_input=True)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose mode")
def preprocess(
    filename,
    output,
    pdf_backend,
    ocr_backend,
    use_ocr,
    force_ocr,
    ocr_languages,
    llm_model,
    base_url,
    api_key,
    verbose,
):
    """Preprocesses a file"""
    load_dotenv()
    result = preprocess_file(
        filename,
        output=output,
        verbose=verbose,
        pdf_backend=pdf_backend,
        base_url=base_url,
        api_key=api_key,
        llm_model=llm_model,
        ocr_backend=ocr_backend,
        use_ocr=use_ocr,
        force_ocr=force_ocr,
        ocr_languages=ocr_languages,
    )
    click.echo(result)


@main.command()
@click.option("--input", "-i", type=str, help="Input text")
# @click.option('--output', '-o', type=str, help='Output file')
def extract(input):
    """Extracts information from a file"""
    load_dotenv()
    result = extract_info(prompt=input)
    click.echo(result)


if __name__ == "__main__":
    main()
