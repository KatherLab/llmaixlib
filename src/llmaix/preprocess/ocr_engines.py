"""
Wrappers around different OCR back‑ends used by the preprocessing pipeline.

Exports
-------
* run_tesseract_ocr
* run_paddleocr
* run_suryaocr

Every function returns **pure text** (`str`), never a tuple. Any paths to
intermediate OCR PDFs are handled internally and, if needed, by the caller.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

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
    """
    import ocrmypdf
    import pymupdf4llm
    from PIL import Image

    # Detect if file is an image
    if file_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"]:
        with Image.open(file_path) as im:
            im = im.convert("RGB")
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                im.save(tmp, "PDF")
                pdf_path = Path(tmp.name)
    else:
        pdf_path = file_path

    kwargs = {"force_ocr": force_ocr}
    if languages:
        kwargs["language"] = "+".join(languages)

    if output_path:
        ocrmypdf.ocr(str(pdf_path), str(output_path), **kwargs)
        return pymupdf4llm.to_markdown(output_path)
    else:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            temp_output = Path(tmp.name)

        try:
            ocrmypdf.ocr(str(pdf_path), str(temp_output), **kwargs)
            return pymupdf4llm.to_markdown(temp_output)
        finally:
            if temp_output.exists():
                temp_output.unlink()
        # If pdf_path was temporary, delete it
        if pdf_path != file_path and pdf_path.exists():
            pdf_path.unlink()


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

            pipeline = PPStructureV3(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
            )
            results: list[str] = []

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
                        output = pipeline.predict(np.array(img))
                        for res in output:
                            md = (
                                (
                                    res.get("markdown_texts")
                                    or res.get("markdown")
                                    or str(res)
                                )
                                if isinstance(res, dict)
                                else str(res)
                            )
                            results.append(md)
            elif mime and mime.startswith("image/"):
                with Image.open(file_path) as img:
                    img = img.convert("RGB")
                    if max(img.size) > max_image_dim:
                        img.thumbnail(
                            (max_image_dim, max_image_dim), Image.Resampling.LANCZOS
                        )
                    output = pipeline.predict(np.array(img))
                    for res in output:
                        md = (
                            (
                                res.get("markdown_texts")
                                or res.get("markdown")
                                or str(res)
                            )
                            if isinstance(res, dict)
                            else str(res)
                        )
                        results.append(md)
            else:
                raise ValueError(f"Unsupported file type: {file_path} ({mime})")
            return "\n\n".join(results)
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "PaddleOCR (paddleocr) not installed. Install with `pip install paddleocr`."
        ) from e
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"PaddleOCR failed on {file_path}: {e}") from e


# ---------------------------------------------------------------------------
# Surya‑OCR
# ---------------------------------------------------------------------------


def run_suryaocr(
    file_path: Path,
    languages: list[str] | None = None,
    max_image_dim: int = 800,
) -> str:
    """
    Accepts PDF or image. Processes accordingly.
    """
    import fitz
    from PIL import Image
    from surya.detection import DetectionPredictor
    from surya.recognition import RecognitionPredictor

    # cache models
    if not hasattr(run_suryaocr, "_recog"):
        run_suryaocr._recog = RecognitionPredictor()  # type: ignore[attr-defined]
        run_suryaocr._detect = DetectionPredictor()  # type: ignore[attr-defined]

    recog = run_suryaocr._recog  # type: ignore[attr-defined]
    detect = run_suryaocr._detect  # type: ignore[attr-defined]

    images = []
    # PDF branch
    if file_path.suffix.lower() == ".pdf":
        with fitz.open(file_path) as doc:
            for page in doc:
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                if max(img.size) > max_image_dim:
                    img.thumbnail(
                        (max_image_dim, max_image_dim), Image.Resampling.LANCZOS
                    )
                images.append(img)
    # Image branch
    elif file_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"]:
        with Image.open(file_path) as img:
            img = img.convert("RGB")
            if max(img.size) > max_image_dim:
                img.thumbnail((max_image_dim, max_image_dim), Image.Resampling.LANCZOS)
            images.append(img)
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")

    predictions = recog(images, det_predictor=detect)
    lines: list[str] = []
    for page_pred in predictions:
        if hasattr(page_pred, "text_lines"):
            lines.extend(line.text for line in page_pred.text_lines)
        lines.append("")  # page break

    return "\n".join(lines).strip()
