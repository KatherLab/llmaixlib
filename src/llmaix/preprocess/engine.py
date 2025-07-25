from pathlib import Path
from typing import Optional, Union
from tempfile import NamedTemporaryFile
from .ocr_engines import run_tesseract_ocr, run_paddleocr, run_suryaocr
from .backends import extract_pymupdf, extract_docling
from .utils import string_is_empty_or_garbage


class DocumentPreprocessor:
    """
    See preprocess_file for usage and documentation.
    """

    VALID_MODES = {"fast", "advanced"}

    def __init__(
        self,
        mode: str = "fast",
        ocr_engine: Optional[str] = None,
        enable_picture_description: bool = False,
        enable_formula: bool = False,
        enable_code: bool = False,
        output_format: str = "markdown",
        llm_client=None,
        llm_model: Optional[str] = None,
        use_local_vlm: bool = False,
        local_vlm_repo_id: Optional[str] = None,
        ocr_model_paths: Optional[dict] = None,
        force_ocr: bool = False,
    ):
        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode: {mode}. Valid options are {self.VALID_MODES}")

        self.mode = mode
        self.ocr_engine = ocr_engine or ("ocrmypdf" if mode == "fast" else "paddleocr")
        self.enrich = {
            "picture": enable_picture_description,
            "formula": enable_formula,
            "code": enable_code,
        }
        self.format = output_format
        self.client = llm_client
        self.llm_model = llm_model
        self.use_local_vlm = use_local_vlm
        self.local_vlm_repo_id = local_vlm_repo_id
        self.ocr_model_paths = ocr_model_paths
        self.force_ocr = force_ocr

    def process(self, source: Union[Path, bytes]) -> str:
        """
        Extract text from a document path or bytes, using fast or advanced mode.
        """
        if isinstance(source, bytes):
            with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(source)
                tmp_path = Path(tmp.name)
        else:
            tmp_path = Path(source)

        if tmp_path.suffix.lower() == ".txt":
            return tmp_path.read_text(errors="ignore")

        if tmp_path.suffix.lower() == ".pdf":
            if self.mode == "fast":
                text = "" if self.force_ocr else extract_pymupdf(tmp_path)
                if self.force_ocr or string_is_empty_or_garbage(text):
                    text, _ = self._ocr_and_extract(tmp_path)
            else:
                use_vlm = bool(self.client and self.llm_model) or self.use_local_vlm
                text, _ = extract_docling(
                    tmp_path,
                    self.enrich,
                    use_vlm,
                    self.client,
                    self.llm_model,
                    use_local_vlm=self.use_local_vlm,
                    local_vlm_repo_id=self.local_vlm_repo_id,
                    ocr_model_paths=self.ocr_model_paths,
                )
                if (self.force_ocr or string_is_empty_or_garbage(text)) and not use_vlm:
                    text, _ = self._ocr_and_extract(tmp_path)
            return text
        else:
            if self.mode == "fast" and not self.force_ocr:
                try:
                    return tmp_path.read_text(errors="ignore")
                except Exception:
                    return ""
            else:
                # For other doc formats supported by docling (e.g., docx)
                text, _ = extract_docling(
                    tmp_path,
                    self.enrich,
                    bool(self.client and self.llm_model) or self.use_local_vlm,
                    self.client,
                    self.llm_model,
                    use_local_vlm=self.use_local_vlm,
                    local_vlm_repo_id=self.local_vlm_repo_id,
                    ocr_model_paths=self.ocr_model_paths,
                )
                if self.force_ocr or string_is_empty_or_garbage(text):
                    text, _ = self._ocr_and_extract(tmp_path)
                return text

    def _ocr_and_extract(self, path: Path):
        if self.ocr_engine == "ocrmypdf":
            return run_tesseract_ocr(path, force_ocr=self.force_ocr)
        elif self.ocr_engine == "paddleocr":
            return run_paddleocr(path)
        elif self.ocr_engine == "surya":
            return run_suryaocr(path)
        else:
            raise ValueError(f"Unknown OCR engine: {self.ocr_engine}")


def preprocess_file(
    path: Union[Path, bytes],
    *,
    mode: str = "fast",
    ocr_engine: Optional[str] = None,
    enable_picture_description: bool = False,
    enable_formula: bool = False,
    enable_code: bool = False,
    output_format: str = "markdown",
    llm_client=None,
    llm_model: Optional[str] = None,
    use_local_vlm: bool = False,
    local_vlm_repo_id: Optional[str] = None,
    ocr_model_paths: Optional[dict] = None,
    force_ocr: bool = False,
) -> str:
    """
    Preprocess a document for LLM input using PDF/Docx/Text/Image file.
    Args:
      path: Input file path or bytes.
      mode: "fast" (speed, direct extraction, Tesseract fallback), or
            "advanced" (layout-aware Docling, VLM, tables, enrichments).
      ocr_engine: One of "ocrmypdf", "paddleocr", "surya" (default per mode).
      enable_picture_description: Enable vision captioning for images/figures (Docling).
      enable_formula: Enable formula enrichment (LaTeX in output) (Docling).
      enable_code: Enable code block detection (Docling).
      output_format: "markdown" (default, for LLMs), or "text".
      llm_client: For remote VLM image captioning, client with .base_url and .api_key.
      llm_model: For remote VLM, model name/id.
      use_local_vlm: If True, use a local HuggingFace VLM (e.g. SmolVLM).
      local_vlm_repo_id: If use_local_vlm, provide model repo_id.
      ocr_model_paths: For custom PaddleOCR v5 via RapidOCR.
      force_ocr: If True, force OCR processing regardless of file type or content.

    Returns:
      Extracted markdown (default) or plain text from the document.
    """
    return DocumentPreprocessor(
        mode=mode,
        ocr_engine=ocr_engine,
        enable_picture_description=enable_picture_description,
        enable_formula=enable_formula,
        enable_code=enable_code,
        output_format=output_format,
        llm_client=llm_client,
        llm_model=llm_model,
        use_local_vlm=use_local_vlm,
        local_vlm_repo_id=local_vlm_repo_id,
        ocr_model_paths=ocr_model_paths,
        force_ocr=force_ocr,
    ).process(path)