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
    pdf_path: Path,
    languages: list[str] | None = None,
    force_ocr: bool = False,
    output_path: Path | None = None,
) -> str:
    """
    Run Tesseract OCR using *ocrmypdf* and return extracted Markdown.

    Parameters
    ----------
    pdf_path:
        Source PDF file (will not be modified).
    languages:
        List of Tesseract language codes (e.g. ``['eng', 'deu']``).
    force_ocr:
        Passes ``--force-ocr`` to ocrmypdf even if text is already present.
    output_path:
        Optional path to write the OCR'd PDF. If not provided, a temporary file is used.
    """
    import ocrmypdf
    import pymupdf4llm

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


# ---------------------------------------------------------------------------
# PaddleOCR PP‑Structure
# ---------------------------------------------------------------------------


def run_paddleocr(
    pdf_path: Path,
    languages: list[str] | None = None,
    max_image_dim: int = 800,
) -> str:
    """
    Perform advanced OCR with PaddleOCR PP‑StructureV3.

    Returns Markdown combining text, table, and formula layout where possible.
    """
    import warnings
    from pathlib import Path as _P

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="invalid escape sequence '\\\\W'",
                category=SyntaxWarning,
                module="paddlex",
            )
            import fitz
            import numpy as np
            from paddleocr import PPStructureV3
            from PIL import Image

            pipeline = PPStructureV3(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
            )
            results: list[str] = []
            with fitz.open(_P(pdf_path)) as doc:
                for page in doc:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                    if max(img.size) > max_image_dim:
                        img.thumbnail(
                            (max_image_dim, max_image_dim), Image.Resampling.LANCZOS
                        )
                    output = pipeline.predict(np.array(img))
                    for res in output:
                        if isinstance(res, dict):
                            md = (
                                res.get("markdown_texts")
                                or res.get("markdown")
                                or str(res)
                            )
                        elif hasattr(res, "markdown_texts") and res.markdown_texts:
                            md = res.markdown_texts
                        elif hasattr(res, "markdown") and isinstance(
                            res.markdown, dict
                        ):
                            md = (
                                res.markdown.get("markdown_texts")
                                or res.markdown.get("markdown")
                                or str(res.markdown)
                            )
                        else:
                            md = str(res)
                        results.append(md)
            return "\n\n".join(results)
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "PaddleOCR (paddleocr) not installed. Install with `pip install paddleocr`."
        ) from e
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"PaddleOCR failed on {pdf_path}: {e}") from e


# ---------------------------------------------------------------------------
# Surya‑OCR
# ---------------------------------------------------------------------------


def run_suryaocr(
    pdf_path: Path,
    languages: list[str] | None = None,  # auto‑detect; kept for API symmetry
    max_image_dim: int = 800,
) -> str:
    """
    Run Surya‑OCR v0.14+ and return plain text (one line per OCR line).
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
    with fitz.open(pdf_path) as doc:
        for page in doc:
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            if max(img.size) > max_image_dim:
                img.thumbnail((max_image_dim, max_image_dim), Image.Resampling.LANCZOS)
            images.append(img)

    predictions = recog(images, det_predictor=detect)
    lines: list[str] = []
    for page_pred in predictions:
        if hasattr(page_pred, "text_lines"):
            lines.extend(line.text for line in page_pred.text_lines)
        lines.append("")  # page break

    return "\n".join(lines).strip()
