from pathlib import Path
from typing import Optional, Tuple, List


def run_tesseract_ocr(
    pdf_path: Path, languages: Optional[List[str]] = None, force_ocr: bool = False
) -> Tuple[str, Path]:
    """
    Use Tesseract OCR via ocrmypdf, then extract text with pymupdf4llm.

    Args:
        pdf_path: Path to the input PDF.
        languages: List of language codes (e.g., ['eng', 'deu']).
        force: If True, force OCR even if text is present.

    Returns:
        (extracted_text, path_to_ocr_pdf)
    """
    import ocrmypdf
    import pymupdf4llm

    tmp = pdf_path.with_name(pdf_path.stem + "_ocr.pdf")
    kwargs = {}
    # TODO: Pass languages
    if languages:
        kwargs["language"] = languages
    if force_ocr:
        kwargs["force_ocr"] = True
    ocrmypdf.ocr(str(pdf_path), str(tmp), force_ocr=force_ocr)
    text = pymupdf4llm.to_markdown(tmp)
    return text, tmp


def run_paddleocr(
    pdf_path: Path,
    languages: Optional[list] = None,
    max_image_dim: int = 800
) -> Tuple[str, Optional[Path]]:
    """
    Use PaddleOCR PPStructureV3 for advanced OCR with layout/table detection.

    Raises an exception if anything goes wrong (missing dependency, error in pipeline, etc).
    """
    import warnings

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="invalid escape sequence '\\W'",
                category=SyntaxWarning,
                module="paddlex"
            )
            from paddleocr import PPStructureV3
            import numpy as np
            import fitz
            from PIL import Image

            pipeline = PPStructureV3(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
            )

            results = []
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                    if max(img.size) > max_image_dim:
                        img.thumbnail((max_image_dim, max_image_dim), Image.Resampling.LANCZOS)
                    output = pipeline.predict(np.array(img))
                    for res in output:
                        # Robust markdown extraction
                        if isinstance(res, dict):
                            md = res.get("markdown_texts") or res.get("markdown") or str(res)
                        elif hasattr(res, "markdown_texts") and res.markdown_texts:
                            md = res.markdown_texts
                        elif hasattr(res, "markdown") and isinstance(res.markdown, dict):
                            md = res.markdown.get("markdown_texts") or res.markdown.get("markdown") or str(res.markdown)
                        else:
                            md = str(res)
                        results.append(md)
            return "\n\n".join(results), None

    except ImportError as e:
        raise RuntimeError(
            "PaddleOCR (paddleocr) or one of its dependencies is not installed. "
            "Install with `pip install paddleocr`.\nOriginal error: " + str(e)
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"PaddleOCR failed on {pdf_path}: {e}"
        ) from e


def run_suryaocr(
    pdf_path: Path,
    languages: Optional[List[str]] = None,
    max_image_dim: int = 800
) -> Tuple[str, Optional[Path]]:
    """
    Use Surya-OCR v0.14.x+ for text recognition and line layout.

    Args:
        pdf_path: Path to PDF file.
        languages: Ignored (Surya auto-detects).
        max_image_dim: Maximum width or height for images (resized for speed/memory).

    Returns:
        (extracted_text, None)
    """
    from surya.recognition import RecognitionPredictor
    from surya.detection import DetectionPredictor
    import fitz
    from PIL import Image

    # Use cached predictors to avoid re-loading models
    if not hasattr(run_suryaocr, "_recognition_predictor"):
        run_suryaocr._recognition_predictor = RecognitionPredictor()
    if not hasattr(run_suryaocr, "_detection_predictor"):
        run_suryaocr._detection_predictor = DetectionPredictor()

    recog = run_suryaocr._recognition_predictor
    detect = run_suryaocr._detection_predictor

    with fitz.open(pdf_path) as doc:
        images = []
        for page in doc:
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            if max(img.size) > max_image_dim:
                img.thumbnail((max_image_dim, max_image_dim), Image.Resampling.LANCZOS)
            images.append(img)
    # Surya: batch prediction (returns List[OCRResult])
    predictions = recog(images, det_predictor=detect)

    # Each OCRResult object should have .text_lines (list of line objects/dicts)
    text = ""
    for page_pred in predictions:
        if hasattr(page_pred, "text_lines"):
            for line in page_pred.text_lines:
                value = getattr(line, "text", None)
                if value is None and isinstance(line, dict):
                    value = line.get("text", "")
                if value:
                    text += value + "\n"
        text += "\n"
    return text.strip(), None

