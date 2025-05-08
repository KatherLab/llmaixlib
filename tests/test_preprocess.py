# tests/test_preprocess.py
from llmaix.preprocess import preprocess_file
from pathlib import Path
import pytest

def test_preprocess_pdf_with_text(tmp_path):
    # Create a temporary file
    file_path = Path("tests") / Path("testfiles") / "9874562_text.pdf"

    # Call the preprocess_file function
    result = preprocess_file(file_path)

    # Assert that the result is as expected
    assert "Re: Medical History and Clinical Course of Patient Ashley Park" in result


def test_preprocess_pdf_with_text_and_ocr(tmp_path, ocr_backend):

    file_path = Path("tests") / Path("testfiles") / "9874562_notext.pdf"

    result = preprocess_file(file_path, use_ocr=True, ocr_backend=ocr_backend)

    assert "Re: Medical History and Clinical Course of Patient Ashley Park" in result


def test_preprocess_pdf_without_text_without_ocr(tmp_path, ocr_backend):
    # Create a temporary file
    file_path = Path("tests") / Path("testfiles") / "9874562_notext.pdf"

    # Expect an ValueError to be raised
    try:
        preprocess_file(file_path, ocr_backend=ocr_backend)
    except ValueError as e:
        assert str(e) == f"PDF {file_path} is empty and no OCR was requested."
    else:
        assert False, "Expected ValueError not raised"


def test_preprocess_pdf_with_misleading_text_with_ocr(tmp_path, ocr_backend):
    # Create a temporary file
    file_path = Path("tests") / Path("testfiles") / "9874562_misleading_text.pdf"

    # Call the preprocess_file function
    try:
        preprocess_file(file_path, use_ocr=True, ocr_backend=ocr_backend)
    except RuntimeError as e:
        assert "An error occurred while processing the PDF with ocrmypdf: " in str(e)


def test_preprocess_pdf_with_misleading_text_force_ocr(tmp_path, ocr_backend):
    # Create a temporary file
    file_path = Path("tests") / Path("testfiles") / "9874562_misleading_text.pdf"

    # Call the preprocess_file function
    result = preprocess_file(file_path, use_ocr=True, force_ocr=True, ocr_backend=ocr_backend)

    # Assert that the result is as expected
    assert "Re: Medical History and Clinical Course of Patient Ashley Park" in result
