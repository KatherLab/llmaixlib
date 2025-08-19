"""User-facing API (`preprocess_file`) and core orchestration."""

from __future__ import annotations

import atexit
import os
import tempfile
from pathlib import Path
from typing import Callable
from urllib.parse import urljoin

from pydantic import AnyUrl

from .backends import extract_docling, extract_pymupdf
from .document import Document
from .mime_detect import detect_mime
from .ocr_engines import run_paddleocr, run_marker, run_tesseract_ocr
from .utils import string_is_empty_or_garbage

# --------------------------------------------------------------------------------------
# Plugin registry (lightweight –for future formats)
# --------------------------------------------------------------------------------------
_BACKENDS: dict[str, Callable[[Document, "DocumentPreprocessor"], str]] = {}


def register_backend(mime: str):
    def _decorator(fn: Callable[[Document, "DocumentPreprocessor"], str]):
        _BACKENDS[mime] = fn
        return fn

    return _decorator


class DocumentPreprocessor:
    """High-level orchestrator. Instantiate once per *job*."""

    VALID_MODES = {"fast", "advanced"}

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------
    def __init__(
        self,
        *,
        mode: str = "fast",
        ocr_engine: str | None = None,
        enable_picture_description: bool = False,
        enable_formula: bool = False,
        enable_code: bool = False,
        output_format: str = "markdown",
        llm_client=None,
        llm_model: str | None = None,
        use_local_vlm: bool = False,
        local_vlm_repo_id: str | None = None,
        ocr_model_paths: dict | None = None,
        docling_ocr_engine: str = "rapidocr",
        force_ocr: bool = False,
        vlm_prompt: str | None = None,
        max_image_dim: int = 800,  # max image dimension for VLM processing (will be downscaled respecting aspect ratio)
        languages: list[str] | None = None,  # languages for tesseract ocr engine
    ) -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode: {mode}. Valid options: {self.VALID_MODES}")

        self.mode = mode

        if llm_client:
            # Ensure client is a class with a `base_url` and `api_key`
            if not hasattr(llm_client, "base_url") or not hasattr(
                llm_client, "api_key"
            ):
                raise ValueError(
                    "llm_client must have 'base_url' and 'api_key' attributes."
                )
            if self.mode == "fast":
                print(
                    "Providing LLM client in fast mode is not supported. Use advanced mode."
                )
            base_url_str = str(llm_client.base_url)

            if "chat/completions" not in base_url_str:
                llm_client.base_url = AnyUrl(
                    url=urljoin(base_url_str, "chat/completions")
                )

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
        self.docling_ocr_engine = docling_ocr_engine
        self.force_ocr = force_ocr
        self.vlm_prompt = vlm_prompt
        self.max_image_dim = max_image_dim
        self.languages = languages

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process(
        self, source: Path | str | bytes, catch_exceptions: bool = False
    ) -> str:
        """Extract Markdown/plain text from *source*.

        Always returns a **string**; returns "" (empty) on unrecoverable errors.
        """

        if catch_exceptions:
            try:
                doc = self._prepare_document(source)
                handler = _BACKENDS.get(doc.mime)
                if handler:
                    return handler(doc, self)
                return self._default_process(doc)
            except Exception as exc:
                print(
                    f"[WARN] preprocessing failed for {source}: {exc}"
                )  # TODO: replace with logging
                return ""
        else:
            doc = self._prepare_document(source)
            handler = _BACKENDS.get(doc.mime)
            if handler:
                return handler(doc, self)
            return self._default_process(doc)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _prepare_document(self, source: Path | bytes) -> Document:
        """Convert arbitrary *source* into a `Document` and ensure temp cleanup."""
        # ---------------- Path input ----------------
        if isinstance(source, str):
            source = Path(source)
        if isinstance(source, Path):
            mime = detect_mime(source) or "application/octet-stream"
            return Document.from_path(source, mime)

        # ---------------- bytes input ---------------
        mime = detect_mime(source) or "application/octet-stream"
        suffix_map = {
            "application/pdf": ".pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
            "text/plain": ".txt",
        }
        tmp_suffix = suffix_map.get(mime, ".bin")
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=tmp_suffix)
        tmp_file.write(source)
        tmp_file.flush()
        tmp_path = Path(tmp_file.name)
        tmp_file.close()
        atexit.register(lambda p: os.remove(p) if p.exists() else None, tmp_path)
        return Document.from_path(tmp_path, mime)

    # ------------------------------------------------------------------

    def _default_process(self, doc: Document) -> str:
        """
        Core file routing: extracts plain text, PDF, or image input,
        and processes with selected OCR engine as appropriate.
        """
        path = doc.raw_path
        mime = doc.mime  # Detected via detect_mime
        suffix = path.suffix.lower()
        use_vlm = bool(self.client and self.llm_model) or self.use_local_vlm

        # ---------------- Plain text ----------------
        if mime and mime.startswith("text/") or suffix == ".txt":
            doc.text = path.read_text(errors="ignore")
            return doc.text

        # ---------------- PDF -----------------------
        if mime == "application/pdf" or suffix == ".pdf":
            if self.mode == "fast":
                text = "" if self.force_ocr else extract_pymupdf(path)
                if self.force_ocr or string_is_empty_or_garbage(text):
                    text = self._ocr_and_extract(path)
                doc.text = text
                return text

            # advanced mode
            text = extract_docling(
                path,
                self.enrich,
                use_vlm,
                self.client,
                self.llm_model,
                use_local_vlm=self.use_local_vlm,
                local_vlm_repo_id=self.local_vlm_repo_id,
                ocr_model_paths=self.ocr_model_paths,
                ocr_engine=self.docling_ocr_engine,
            )
            if (self.force_ocr or string_is_empty_or_garbage(text)) and not use_vlm:
                text = self._ocr_and_extract(path)
            doc.text = text
            return text

        # ---------------- Images (PNG, JPEG, TIFF, BMP, GIF, etc.) ----------------
        if mime and mime.startswith("image/"):
            # Pass image file directly to OCR engine (all runners now support this)
            text = self._ocr_and_extract(path)
            doc.text = text
            return text

        # ---------------- Unsupported / others ------
        if self.mode == "fast":
            raise ValueError(
                f"Unsupported MIME {mime} in fast mode. Use advanced mode or add a plugin."
            )

        # advanced fallback via Docling for unknown formats
        text = extract_docling(
            path,
            self.enrich,
            use_vlm,
            self.client,
            self.llm_model,
            use_local_vlm=self.use_local_vlm,
            local_vlm_repo_id=self.local_vlm_repo_id,
            ocr_model_paths=self.ocr_model_paths,
            vlm_prompt=self.vlm_prompt,
            ocr_engine=self.docling_ocr_engine,
        )
        if self.force_ocr or string_is_empty_or_garbage(text):
            text = self._ocr_and_extract(path)
        doc.text = text
        return text

    # ------------------------------------------------------------------
    def _ocr_and_extract(self, path: Path) -> str:
        if self.ocr_engine == "ocrmypdf":
            return run_tesseract_ocr(
                path, force_ocr=self.force_ocr, languages=self.languages
            )
        if self.ocr_engine == "paddleocr":
            return run_paddleocr(path, max_image_dim=self.max_image_dim)
        if self.ocr_engine == "marker":
            return run_marker(path)
        raise ValueError(f"Unknown OCR engine: {self.ocr_engine}")
