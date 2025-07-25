# tests/test_preprocess.py

import pytest
from llmaix.preprocess import preprocess_file
from pathlib import Path

PDF_WITH_TEXT = Path("tests/testfiles/9874562_text.pdf")
PDF_NO_TEXT = Path("tests/testfiles/9874562_notext.pdf")
PDF_MISLEADING_TEXT = Path("tests/testfiles/9874562_misleading_text.pdf")
DOCX_FILE = Path("tests/testfiles/9874562.docx")
TXT_FILE = Path("tests/testfiles/9874562.txt")
IMG_FILE = Path("tests/testfiles/9874562.png")


@pytest.mark.parametrize("mode", ["fast", "advanced"])
def test_preprocess_pdf_with_text(mode):
    result = preprocess_file(PDF_WITH_TEXT, mode=mode)
    assert "Medical History" in result


@pytest.mark.parametrize("ocr_engine", ["ocrmypdf", "paddleocr", "surya"])
@pytest.mark.parametrize("mode", ["fast", "advanced"])
def test_preprocess_pdf_needs_ocr(ocr_engine, mode):
    result = preprocess_file(PDF_NO_TEXT, mode=mode, ocr_engine=ocr_engine)
    assert "Medical History" in result


def test_preprocess_pdf_with_force_ocr():
    # Even if text exists, should OCR
    result = preprocess_file(PDF_WITH_TEXT, mode="fast", ocr_engine="ocrmypdf")
    # Should use direct extraction (not OCR), still succeeds
    assert "Medical History" in result

    # Now force OCR
    result2 = preprocess_file(PDF_WITH_TEXT, mode="fast", ocr_engine="ocrmypdf")
    assert "Medical History" in result2


def test_preprocess_pdf_misleading_text_and_force_ocr():
    # Should fall back to OCR and still work
    result = preprocess_file(PDF_MISLEADING_TEXT, mode="fast", ocr_engine="ocrmypdf", force_ocr=True)
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
    # Will work in advanced mode (Docling)
    result = preprocess_file(file_path, mode="advanced")
    assert expected in result


def test_preprocess_pdf_with_local_vlm(monkeypatch):
    # Fake a local VLM model (simulate as if a local model is available)
    result = preprocess_file(
        PDF_NO_TEXT,
        mode="advanced",
        use_local_vlm=True,
        local_vlm_repo_id="HuggingFaceTB/SmolVLM-256M-Instruct",
        enable_picture_description=True,
    )
    assert "Medical History" in result or "image description" in result.lower()


def test_preprocess_pdf_with_remote_vlm(monkeypatch):
    # Simulate a remote API; you can provide a mock client with base_url/api_key attributes
    class DummyClient:
        base_url = "https://dummy"
        api_key = "sk-test"

    # In actual use, this will need a real remote endpoint
    with pytest.raises(Exception):
        preprocess_file(
            PDF_NO_TEXT,
            mode="advanced",
            llm_client=DummyClient(),
            llm_model="gpt-4v",
            enable_picture_description=True,
        )


def test_preprocess_pdf_as_bytes():
    with open(PDF_WITH_TEXT, "rb") as f:
        result = preprocess_file(f.read(), mode="fast")
    assert "Medical History" in result


@pytest.mark.parametrize("file_path", [PDF_WITH_TEXT, PDF_NO_TEXT, DOCX_FILE])
def test_output_text_format(file_path):
    result = preprocess_file(file_path, mode="advanced", output_format="text")
    # Should have no markdown symbols (naive test)
    assert "#" not in result and "|" not in result or "Medical History" in result
