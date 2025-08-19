# tests/test_preprocess.py
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from llmaix.preprocess import DocumentPreprocessor

PDF_WITH_TEXT = Path("tests/testfiles/9874562_text.pdf")
PDF_NO_TEXT = Path("tests/testfiles/9874562_notext.pdf")
PDF_MISLEADING_TEXT = Path("tests/testfiles/9874562_misleading_text.pdf")
DOCX_FILE = Path("tests/testfiles/9874562.docx")
TXT_FILE = Path("tests/testfiles/9874562.txt")
IMG_FILE = Path("tests/testfiles/9874562.png")


def run_preprocess(source, **kwargs):
    """
    Convenience wrapper so tests can call the pipeline in one line.

    `source` is passed to `DocumentPreprocessor.process`, every other keyword
    argument goes into the constructor.
    """
    proc = DocumentPreprocessor(**{k: v for k, v in kwargs.items()})
    return proc.process(source)


@pytest.mark.parametrize("mode", ["fast", "advanced"])
def test_preprocess_pdf_with_text(mode):
    result = run_preprocess(PDF_WITH_TEXT, mode=mode)
    assert "Medical History" in result


@pytest.mark.parametrize("ocr_engine", ["ocrmypdf", "paddleocr", "marker"])
@pytest.mark.parametrize("mode", ["fast", "advanced"])
def test_preprocess_pdf_needs_ocr(ocr_engine, mode):
    result = run_preprocess(PDF_NO_TEXT, mode=mode, ocr_engine=ocr_engine)
    assert "Medical History" in result


def test_preprocess_pdf_with_force_ocr():
    # text layer present → direct extraction
    result = run_preprocess(PDF_WITH_TEXT, mode="fast", ocr_engine="ocrmypdf")
    assert "Medical History" in result

    # force OCR regardless
    result2 = run_preprocess(
        PDF_WITH_TEXT, mode="fast", ocr_engine="ocrmypdf", force_ocr=True
    )
    assert "Medical History" in result2


def test_preprocess_pdf_misleading_text_and_force_ocr():
    result = run_preprocess(
        PDF_MISLEADING_TEXT, mode="fast", ocr_engine="ocrmypdf", force_ocr=True
    )
    assert "Medical History" in result


@pytest.mark.parametrize(
    "file_path,expected",
    [
        (DOCX_FILE, "Medical History"),
        (TXT_FILE, "Medical History"),
        (IMG_FILE, "Medical History"),
    ],
)
def test_preprocess_other_formats(file_path, expected):
    result = run_preprocess(file_path, mode="advanced")
    assert expected in result


def test_preprocess_image_format():
    result = run_preprocess(IMG_FILE, mode="fast")
    assert "Medical History" in result or "image description" in result.lower()


def test_preprocess_pdf_with_local_vlm():
    result = run_preprocess(
        PDF_NO_TEXT,
        mode="advanced",
        use_local_vlm=True,
        local_vlm_repo_id="HuggingFaceTB/SmolVLM-256M-Instruct",
        enable_picture_description=True,
    )
    assert result.strip()
    assert "Ashley Park" in result or "image description" in result.lower()


def test_preprocess_pdf_with_remote_vlm():
    # Try to load environment variables from .env file
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_api_base = os.getenv("OPENAI_API_BASE")
    openai_model = os.getenv("OPENAI_MODEL")

    if openai_api_key and openai_api_base and openai_model:

        class DummyClient:
            base_url = openai_api_base + "/chat/completions"
            api_key = openai_api_key

        result = run_preprocess(
            PDF_NO_TEXT,
            mode="advanced",
            llm_client=DummyClient(),
            llm_model=openai_model,
            enable_picture_description=True,
        )
        assert "Ashley Park" in result
    else:
        pytest.skip("Skipping remote VLM test due to missing environment variables.")


def test_preprocess_pdf_as_bytes():
    with open(PDF_WITH_TEXT, "rb") as f:
        result = run_preprocess(f.read(), mode="fast")
    assert "Medical History" in result


@pytest.mark.parametrize("file_path", [PDF_WITH_TEXT, PDF_NO_TEXT, DOCX_FILE])
def test_output_text_format(file_path):
    result = run_preprocess(file_path, mode="advanced", output_format="text")
    assert ("#" not in result and "|" not in result) or ("Medical History" in result)
