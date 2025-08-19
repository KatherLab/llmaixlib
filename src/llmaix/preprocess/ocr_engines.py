"""
Wrappers around different OCR back‑ends used by the preprocessing pipeline.

Exports
-------
* run_tesseract_ocr
* run_paddleocr
* run_marker

Every function returns **pure text** (`str`), never a tuple. Any paths to
intermediate OCR PDFs are handled internally and, if needed, by the caller.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from llmaix.preprocess.mime_detect import detect_mime

# ---------------------------------------------------------------------------
# Tesseract via ocrmypdf
# ---------------------------------------------------------------------------


def run_tesseract_ocr(
    file_path: Path,
    languages: list[str] | None = None,
    force_ocr: bool = False,
    output_path: Path | None = None,
) -> str:
    """
    Accepts PDF or image. If image, auto-converts to 1-page PDF for OCRmyPDF.
    Now detects file type using MIME detection, not file extension.
    """
    import ocrmypdf
    import pymupdf4llm
    from PIL import Image

    # --- Use MIME detection to determine file type ---
    mime = detect_mime(file_path)
    if mime is None:
        raise ValueError(f"Could not determine MIME type for file: {file_path}")

    # Common image MIME types
    IMAGE_MIME_PREFIXES = ("image/",)
    PDF_MIME = "application/pdf"

    # Convert image to PDF if needed
    if mime.startswith(IMAGE_MIME_PREFIXES):
        with Image.open(file_path) as im:
            im = im.convert("RGB")
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                im.save(tmp, "PDF")
                pdf_path = Path(tmp.name)
    elif mime == PDF_MIME:
        pdf_path = file_path
    else:
        raise ValueError(f"Unsupported file type: {mime}")

    kwargs = {"force_ocr": force_ocr}
    if languages:
        kwargs["language"] = "+".join(languages)

    if output_path:
        ocrmypdf.ocr(str(pdf_path), str(output_path), **kwargs)
        result_path = output_path
    else:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            temp_output = Path(tmp.name)
        try:
            ocrmypdf.ocr(str(pdf_path), str(temp_output), **kwargs)
            result_path = temp_output
        finally:
            # If pdf_path was temporary, delete it
            if pdf_path != file_path and pdf_path.exists():
                pdf_path.unlink()

    # Use pymupdf4llm to extract markdown text
    try:
        return pymupdf4llm.to_markdown(result_path)
    finally:
        if not output_path and result_path.exists():
            result_path.unlink()


# ---------------------------------------------------------------------------
# PaddleOCR PP‑Structure
# ---------------------------------------------------------------------------


def run_paddleocr(
    file_path: Path,
    languages: list[str] | None = None,
    max_image_dim: int = 800,
) -> str:
    import warnings
    from pathlib import Path as _P

    import numpy as np
    from PIL import Image

    from .mime_detect import detect_mime

    mime = detect_mime(file_path)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="invalid escape sequence '\\\\W'",
                category=SyntaxWarning,
                module="paddlex",
            )
            import fitz
            from paddleocr import PPStructureV3

            # Same pipeline arguments as in your standalone script
            pipeline = PPStructureV3(
                text_recognition_model_name="PP-OCRv5_server_rec",
                text_detection_model_name="PP-OCRv5_server_det",
                use_doc_orientation_classify=True,
                use_textline_orientation=True,
                use_doc_unwarping=False,
                use_table_recognition=True,
                text_det_limit_side_len=2048,
                text_det_box_thresh=0.5,
                device="gpu",
                precision="fp16",
            )

            markdown_list: list[str] = []

            if mime == "application/pdf":
                with fitz.open(_P(file_path)) as doc:
                    for page in doc:
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                        img = Image.frombytes(
                            "RGB", (pix.width, pix.height), pix.samples
                        )
                        if max(img.size) > max_image_dim:
                            img.thumbnail(
                                (max_image_dim, max_image_dim), Image.Resampling.LANCZOS
                            )
                        output = pipeline.predict(np.array(img), use_table_orientation_classify=True)
                        markdown_list.extend([res.markdown for res in output])
            elif mime and mime.startswith("image/"):
                with Image.open(file_path) as img:
                    img = img.convert("RGB")
                    if max(img.size) > max_image_dim:
                        img.thumbnail(
                            (max_image_dim, max_image_dim), Image.Resampling.LANCZOS
                        )
                    output = pipeline.predict(np.array(img), use_table_orientation_classify=True)
                    markdown_list.extend([res.markdown for res in output])
            else:
                raise ValueError(f"Unsupported file type: {file_path} ({mime})")

            # Use the built-in concatenation like in your script
            return pipeline.concatenate_markdown_pages(markdown_list)

    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "PaddleOCR (paddleocr) not installed. Install with `pip install paddleocr`."
        ) from e
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"PaddleOCR failed on {file_path}: {e}") from e


# ---------------------------------------------------------------------------
# Marker
# ---------------------------------------------------------------------------


def run_marker(
    file_path: Path,
    languages: list[str] | None = None,  # kept for API parity (unused by Marker)
    max_image_dim: int = 800,            # kept for API parity (unused by Marker)
) -> str:
    """
    Accepts a PDF path and returns Markdown extracted by Marker.
    Note: Marker works on PDFs directly; images are not supported here.
    """
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "Marker is not installed. Install with `pip install marker-pdf`."
        ) from e

    # Only PDFs are supported for Marker in this function
    if file_path.suffix.lower() != ".pdf":
        raise ValueError(f"Unsupported file type for run_marker: {file_path.suffix}. Expected a PDF.")

    # Cache models/converter across calls
    if not hasattr(run_marker, "_converter"):
        model_dict = create_model_dict()
        run_marker._converter = PdfConverter(artifact_dict=model_dict)  # type: ignore[attr-defined]

    converter = run_marker._converter  # type: ignore[attr-defined]

    try:
        rendered = converter(str(file_path))
        # Marker returns a single Markdown string
        return getattr(rendered, "markdown", str(rendered))
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Marker failed on {file_path}: {e}") from e
