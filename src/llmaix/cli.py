# src/llmaix/cli.py
import click
from dotenv import load_dotenv
from .preprocess import preprocess_file
from .extract import extract_info


@click.group()
def main():
    """LLMAIx CLI"""
    pass


@main.command()
@click.argument("filename", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), help="Output file")
@click.option("--base-url", type=str, help="Base URL for the API")
@click.option("--api-key", type=str, help="API key for authentication", hide_input=True)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose mode")
def preprocess(filename, output, base_url, api_key, verbose):
    """Preprocesses a file"""
    load_dotenv()
    result = preprocess_file(
        filename, output=output, verbose=verbose, base_url=base_url, api_key=api_key
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
