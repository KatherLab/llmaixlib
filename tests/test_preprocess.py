# tests/test_preprocess.py
import os

from llmaix.preprocess import preprocess_file
from pathlib import Path


def test_preprocess_pdf_with_text(tmp_path, pdf_backend):
    # Create a temporary file
    file_path = Path("tests") / Path("testfiles") / "9874562_text.pdf"

    # Call the preprocess_file function
    result = preprocess_file(file_path, pdf_backend=pdf_backend)

    # Assert that the result is as expected
    assert (
        "Re: Medical History and Clinical Course of Patient Ashley Park"
        in result.replace(os.linesep, " ")
    )


def test_preprocess_pdf_with_text_and_ocr(tmp_path, ocr_backend, pdf_backend):
    file_path = Path("tests") / Path("testfiles") / "9874562_text.pdf"

    if ocr_backend == "ocrmypdf" and pdf_backend == "ocr_backend":
        try:
            preprocess_file(
                file_path,
                use_ocr=True,
                ocr_backend=ocr_backend,
                pdf_backend=pdf_backend,
            )
        except ValueError as e:
            assert (
                "Cannot use the pdf_backend 'ocr_backend' with ocrmypdf (tesseract)."
                in str(e)
            )
    else:
        result = preprocess_file(
            file_path, use_ocr=True, ocr_backend=ocr_backend, pdf_backend=pdf_backend
        )

        assert (
            "Re: Medical History and Clinical Course of Patient Ashley Park"
            in result.replace(os.linesep, " ")
        )


def test_preprocess_pdf_without_text_and_ocr(tmp_path, ocr_backend, pdf_backend):
    # Create a temporary file
    file_path = Path("tests") / Path("testfiles") / "9874562_notext.pdf"

    # Call the preprocess_file function
    if ocr_backend == "ocrmypdf" and pdf_backend == "ocr_backend":
        try:
            preprocess_file(
                file_path,
                use_ocr=True,
                ocr_backend=ocr_backend,
                pdf_backend=pdf_backend,
            )
            assert False, "Expected ValueError not raised"
        except ValueError as e:
            assert (
                "Cannot use the pdf_backend 'ocr_backend' with ocrmypdf (tesseract)."
                in str(e)
            )
    else:
        result = preprocess_file(
            file_path, use_ocr=True, ocr_backend=ocr_backend, pdf_backend=pdf_backend
        )

        # Assert that the result is as expected
        assert (
            "Re: Medical History and Clinical Course of Patient Ashley Park"
            in result.replace(os.linesep, " ")
        )


def test_preprocess_pdf_without_text_without_ocr(tmp_path, ocr_backend, pdf_backend):
    # Create a temporary file
    file_path = Path("tests") / Path("testfiles") / "9874562_notext.pdf"

    # Expect an ValueError to be raised
    try:
        preprocess_file(file_path, ocr_backend=ocr_backend, pdf_backend=pdf_backend)
        assert False, "Expected ValueError not raised"
    except ValueError as e:
        assert str(e) == f"PDF {file_path} is empty and no OCR was requested."


def test_preprocess_pdf_with_misleading_text_with_ocr(
    tmp_path, ocr_backend, pdf_backend
):
    # Create a temporary file
    file_path = Path("tests") / Path("testfiles") / "9874562_misleading_text.pdf"

    # Call the preprocess_file function
    if ocr_backend == "ocrmypdf" and pdf_backend == "ocr_backend":
        try:
            preprocess_file(
                file_path,
                use_ocr=True,
                ocr_backend=ocr_backend,
                pdf_backend=pdf_backend,
            )
            assert False, "Expected ValueError not raised"
        except ValueError as e:
            assert (
                "Cannot use the pdf_backend 'ocr_backend' with ocrmypdf (tesseract)."
                in str(e)
            )
    else:
        try:
            preprocess_file(
                file_path,
                use_ocr=True,
                ocr_backend=ocr_backend,
                pdf_backend=pdf_backend,
            )
        except RuntimeError as e:
            assert "An error occurred while processing the PDF with ocrmypdf: " in str(
                e
            )


def test_preprocess_pdf_with_misleading_text_force_ocr(
    tmp_path, ocr_backend, pdf_backend
):
    # Create a temporary file
    file_path = Path("tests") / Path("testfiles") / "9874562_misleading_text.pdf"

    if ocr_backend == "ocrmypdf" and pdf_backend == "ocr_backend":
        try:
            preprocess_file(
                file_path,
                use_ocr=True,
                ocr_backend=ocr_backend,
                pdf_backend=pdf_backend,
            )
            assert False, "Expected ValueError not raised"
        except ValueError as e:
            assert (
                "Cannot use the pdf_backend 'ocr_backend' with ocrmypdf (tesseract)."
                in str(e)
            )
    else:
        # Call the preprocess_file function
        result = preprocess_file(
            file_path,
            use_ocr=True,
            force_ocr=True,
            ocr_backend=ocr_backend,
            pdf_backend=pdf_backend,
        )

        # Assert that the result is as expected
        assert (
            "Re: Medical History and Clinical Course of Patient Ashley Park"
            in result.replace(os.linesep, " ")
        )
