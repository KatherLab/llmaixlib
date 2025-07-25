"""
Document‑extraction back‑ends for the preprocessing pipeline.

* PyMuPDF → Markdown for fast text‑only extraction.
* Docling pipeline with optional VLM enrichment.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import pymupdf4llm

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    VlmPipelineOptions,          # NEW
    PdfPipelineOptions,
    EasyOcrOptions,
    RapidOcrOptions,
    TesseractOcrOptions,
    TableStructureOptions,
    TableFormerMode,
)
from docling.datamodel.pipeline_options_vlm_model import (
    InlineVlmOptions,
    ApiVlmOptions,
    ResponseFormat,
    InferenceFramework,
    TransformersModelType,
)
from docling.pipeline.vlm_pipeline import VlmPipeline


# ---------------------------------------------------------------------------
# Fast PyMuPDF extraction
# ---------------------------------------------------------------------------

def extract_pymupdf(pdf: Path) -> str:
    """
    Quick text‑only extraction using `pymupdf4llm.to_markdown`.
    """
    return pymupdf4llm.to_markdown(pdf)


# ---------------------------------------------------------------------------
# Docling extraction with optional enrichment
# ---------------------------------------------------------------------------

def _build_vlm_options_local(repo_id: str, prompt : str) -> InlineVlmOptions:
    """
    Helper for local HuggingFace models (SmolVLM, Granite‑Vision…).
    """
    return InlineVlmOptions(
        repo_id=repo_id,
        prompt=prompt,
        response_format=ResponseFormat.MARKDOWN,
        inference_framework=InferenceFramework.TRANSFORMERS,
        transformers_model_type=TransformersModelType.AUTOMODEL_VISION2SEQ,
        scale=2.0,
        temperature=0.0,
    )

def _build_vlm_options_remote(llm_client, llm_model: str, prompt: str) -> ApiVlmOptions:
    """
    Helper for remote multi‑modal chat APIs (OpenAI compatible, watsonx, LM Studio…).
    """
    return ApiVlmOptions(
        url=llm_client.base_url,
        headers={"Authorization": f"Bearer {llm_client.api_key}",
                 "Content-Type": "application/json"},
        params={"model": llm_model},
        prompt=prompt,
        timeout=60,
        scale=1.0,
        response_format=ResponseFormat.MARKDOWN,
    )



def extract_docling(
    path: Path,
    enrich: Dict[str, bool],
    use_vlm: bool = False,
    llm_client: Any = None,
    llm_model: Optional[str] = None,
    use_local_vlm: bool = False,
    local_vlm_repo_id: Optional[str] = None,
    ocr_engine: str = "rapidocr",
    ocr_langs: Optional[list] = None,
    force_full_page_ocr: bool = False,
    ocr_model_paths: Optional[dict] = None,
    vlm_prompt: Optional[str] = "Please perform OCR! Please extract the full text from the document and describe images and figures!"  # noqa: E501,
) -> str:
    """
    Convert *path* to Markdown using Docling, optionally augmented with VLM.
    """
    ocr_langs = ocr_langs or ["en"]

    # --- OCR options ---------------------------------------------------
    if ocr_engine == "rapidocr":
        ocr_opts = RapidOcrOptions(lang=ocr_langs, force_full_page_ocr=force_full_page_ocr)
        if ocr_model_paths:
            for k, v in ocr_model_paths.items():
                setattr(ocr_opts, k, v)
    elif ocr_engine == "easyocr":
        ocr_opts = EasyOcrOptions(lang=ocr_langs, force_full_page_ocr=force_full_page_ocr)
    elif ocr_engine == "tesseract":
        ocr_opts = TesseractOcrOptions(lang=ocr_langs, force_full_page_ocr=force_full_page_ocr)
    else:
        raise ValueError(f"Unsupported ocr_engine: {ocr_engine}")

    table_opts = TableStructureOptions(do_cell_matching=True, mode=TableFormerMode.ACCURATE)

    # --- choose pipeline + options -------------------------------------
    if use_vlm:
        if use_local_vlm and not local_vlm_repo_id:
            raise ValueError(
                "local_vlm_repo_id must be provided when use_local_vlm=True"
            )

        vlm_opt = (
            _build_vlm_options_local(local_vlm_repo_id, prompt=vlm_prompt)
            if use_local_vlm
            else _build_vlm_options_remote(llm_client, llm_model, prompt=vlm_prompt)
        )

        pipeline_options = VlmPipelineOptions(
            enable_remote_services=not use_local_vlm,
            vlm_options=vlm_opt,
        )

        format_options = {
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=VlmPipeline,
                pipeline_options=pipeline_options,
            )
        }

    else:
        pdf_opts = PdfPipelineOptions(
            do_ocr=True,
            ocr_options=ocr_opts,
            do_picture_description=enrich.get("picture", False),
            do_formula_enrichment=enrich.get("formula", False),
            do_code_enrichment=enrich.get("code", False),
            do_table_structure=True,
            table_structure_options=table_opts,
        )
        format_options = {InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)}

    converter = DocumentConverter(format_options=format_options)
    result = converter.convert(path)
    return result.document.export_to_markdown()