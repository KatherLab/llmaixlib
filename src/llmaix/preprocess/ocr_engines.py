"""
Wrappers around different OCR back-ends used by the preprocessing pipeline.

Exports
-------
* run_tesseract_ocr
* run_paddleocr
* run_marker
* reset_paddle_cache
* reset_marker_converter

Every function returns **pure text** (str), never a tuple. Any paths to
intermediate OCR PDFs are handled internally and, if needed, by the caller.
"""

from __future__ import annotations

import tempfile
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple

from llmaix.preprocess.mime_detect import detect_mime


# =============================================================================
# Tesseract via OCRmyPDF  (no caching here)
# =============================================================================


def run_tesseract_ocr(
    file_path: Path,
    languages: list[str] | None = None,
    force_ocr: bool = False,
    output_path: Path | None = None,
) -> str:
    """
    Accepts PDF or image. If image, auto-converts to 1-page PDF for OCRmyPDF.
    Uses MIME detection, not file extension.
    """
    import ocrmypdf
    import pymupdf4llm
    from PIL import Image

    mime = detect_mime(file_path)
    if mime is None:
        raise ValueError(f"Could not determine MIME type for file: {file_path}")

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
            if pdf_path != file_path and pdf_path.exists():
                pdf_path.unlink()

    try:
        return pymupdf4llm.to_markdown(result_path)
    finally:
        if not output_path and result_path.exists():
            result_path.unlink()


# =============================================================================
# PaddleOCR PP-StructureV3  (thread-safe, per-config singleton cache)
# =============================================================================

# Internal cache keyed by a small, hashable config tuple
_PPSTRUCTURE_CACHE: Dict[Tuple, "PPStructureV3"] = {}
_PPSTRUCTURE_LOCK = threading.Lock()


def _pp_key(
    *,
    text_recognition_model_name: str,
    text_detection_model_name: str,
    use_doc_orientation_classify: bool,
    use_textline_orientation: bool,
    use_doc_unwarping: bool,
    use_table_recognition: bool,
    text_det_limit_side_len: int,
    text_det_box_thresh: float,
    device: str,
    precision: str,
) -> Tuple:
    return (
        text_recognition_model_name,
        text_detection_model_name,
        use_doc_orientation_classify,
        use_textline_orientation,
        use_doc_unwarping,
        use_table_recognition,
        int(text_det_limit_side_len),
        float(text_det_box_thresh),
        device,
        precision,
    )


def _get_ppstructure(**cfg) -> "PPStructureV3":
    key = _pp_key(**cfg)
    pipeline = _PPSTRUCTURE_CACHE.get(key)
    if pipeline is not None:
        return pipeline

    with _PPSTRUCTURE_LOCK:
        pipeline = _PPSTRUCTURE_CACHE.get(key)
        if pipeline is not None:
            return pipeline

        # Import inside to keep failures local and avoid heavy imports on module load
        from paddleocr import PPStructureV3  # type: ignore

        pipeline = PPStructureV3(
            text_recognition_model_name=cfg["text_recognition_model_name"],
            text_detection_model_name=cfg["text_detection_model_name"],
            use_doc_orientation_classify=cfg["use_doc_orientation_classify"],
            use_textline_orientation=cfg["use_textline_orientation"],
            use_doc_unwarping=cfg["use_doc_unwarping"],
            use_table_recognition=cfg["use_table_recognition"],
            text_det_limit_side_len=cfg["text_det_limit_side_len"],
            text_det_box_thresh=cfg["text_det_box_thresh"],
            device=cfg["device"],
            precision=cfg["precision"],
        )
        _PPSTRUCTURE_CACHE[key] = pipeline
        return pipeline


def reset_paddle_cache() -> None:
    """Clear all cached PP-StructureV3 instances (useful after a task)."""
    with _PPSTRUCTURE_LOCK:
        _PPSTRUCTURE_CACHE.clear()


def run_paddleocr(
    file_path: Path,
    languages: list[str] | None = None,  # kept for API parity (unused)
    max_image_dim: int = 800,
    *,
    # Cache-keyable constructor options (defaults match your current code)
    text_recognition_model_name: str = "PP-OCRv5_server_rec",
    text_detection_model_name: str = "PP-OCRv5_server_det",
    use_doc_orientation_classify: bool = True,
    use_textline_orientation: bool = True,
    use_doc_unwarping: bool = False,
    use_table_recognition: bool = True,
    text_det_limit_side_len: int = 2048,
    text_det_box_thresh: float = 0.5,
    device: str = "cpu",     # change to "cpu" if needed
    precision: str = "fp16", # "fp32" on CPU or for some GPU setups
) -> str:
    import warnings
    from pathlib import Path as _P

    import numpy as np
    from PIL import Image

    mime = detect_mime(file_path)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="invalid escape sequence '\\\\W'",
                category=SyntaxWarning,
                module="paddlex",
            )
            import fitz  # PyMuPDF

            pipeline = _get_ppstructure(
                text_recognition_model_name=text_recognition_model_name,
                text_detection_model_name=text_detection_model_name,
                use_doc_orientation_classify=use_doc_orientation_classify,
                use_textline_orientation=use_textline_orientation,
                use_doc_unwarping=use_doc_unwarping,
                use_table_recognition=use_table_recognition,
                text_det_limit_side_len=text_det_limit_side_len,
                text_det_box_thresh=text_det_box_thresh,
                device=device,
                precision=precision,
            )

            markdown_list: list[str] = []

            if mime == "application/pdf":
                with fitz.open(_P(file_path)) as doc:
                    for page in doc:
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                        if max(img.size) > max_image_dim:
                            img.thumbnail((max_image_dim, max_image_dim), Image.Resampling.LANCZOS)
                        output = pipeline.predict(np.array(img), use_table_orientation_classify=True)
                        markdown_list.extend([res.markdown for res in output])

            elif mime and mime.startswith("image/"):
                with Image.open(file_path) as img:
                    img = img.convert("RGB")
                    if max(img.size) > max_image_dim:
                        img.thumbnail((max_image_dim, max_image_dim), Image.Resampling.LANCZOS)
                    output = pipeline.predict(np.array(img), use_table_orientation_classify=True)
                    markdown_list.extend([res.markdown for res in output])

            else:
                raise ValueError(f"Unsupported file type: {file_path} ({mime})")

            return pipeline.concatenate_markdown_pages(markdown_list)

    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "PaddleOCR (paddleocr) not installed. Install with `pip install paddleocr`."
        ) from e
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"PaddleOCR failed on {file_path}: {e}") from e


# =============================================================================
# Marker  (thread-safe singleton)
# =============================================================================

_marker_lock = threading.Lock()
_marker_converter: Optional["PdfConverter"] = None


def _get_marker_converter() -> "PdfConverter":
    global _marker_converter
    if _marker_converter is not None:
        return _marker_converter

    with _marker_lock:
        if _marker_converter is not None:
            return _marker_converter

        # Import inside to avoid heavy imports on module load
        from marker.converters.pdf import PdfConverter  # type: ignore
        from marker.models import create_model_dict    # type: ignore

        model_dict = create_model_dict()
        _marker_converter = PdfConverter(artifact_dict=model_dict)
        return _marker_converter


def reset_marker_converter() -> None:
    """Drop the cached Marker PdfConverter (use after a task to free VRAM)."""
    global _marker_converter
    with _marker_lock:
        _marker_converter = None


def run_marker(
    file_path: Path,
    languages: list[str] | None = None,  # kept for API parity (unused by Marker)
    max_image_dim: int = 800,            # kept for API parity (unused by Marker)
) -> str:
    """
    Accepts a PDF path and returns Markdown extracted by Marker.
    Note: Marker works on PDFs directly; images are not supported here.
    """
    if file_path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Unsupported file type for run_marker: {file_path.suffix}. Expected a PDF."
        )

    try:
        converter = _get_marker_converter()
        rendered = converter(str(file_path))
        return getattr(rendered, "markdown", str(rendered))
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "Marker is not installed. Install with `pip install marker-pdf`."
        ) from e
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Marker failed on {file_path}: {e}") from e
