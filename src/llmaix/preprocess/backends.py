from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import pymupdf4llm


from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    EasyOcrOptions,
    RapidOcrOptions,
    TesseractOcrOptions,
    TableStructureOptions,
    TableFormerMode,
    PictureDescriptionVlmOptions,
    PictureDescriptionApiOptions,
)
from docling.pipeline.vlm_pipeline import VlmPipeline


def extract_pymupdf(pdf: Path) -> str:
    return pymupdf4llm.to_markdown(pdf)


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
) -> Tuple[str, None]:
    """
    Extract document content using Docling, with optional VLM enrichment.

    Args:
        path: Path to document (PDF)
        enrich: Dict with bool keys: 'picture', 'formula', 'code'
        use_vlm: If True, use VLM pipeline (Docling's VlmPipeline)
        llm_client: For remote VLM API: must have .base_url and .api_key
        llm_model: For remote VLM API: model name or id
        use_local_vlm: If True, use open-source VLM (e.g. SmolVLM)
        local_vlm_repo_id: Repo id for HuggingFace open-source VLM
        ocr_engine: OCR backend ("easyocr", "tesseract", "rapidocr")
        ocr_langs: List of OCR languages (default ["en"])
        force_full_page_ocr: If True, always run OCR even if text detected
        ocr_model_paths: For RapidOCR: dict with keys det_model_path, rec_model_path, etc.

    Returns:
        (markdown_text, None)
    """
    ocr_langs = ocr_langs or ["en"]

    # Build OCR options
    if ocr_engine == "rapidocr":
        ocr_opts = RapidOcrOptions(
            lang=ocr_langs,
            force_full_page_ocr=force_full_page_ocr,
        )
        if ocr_model_paths:
            if "det_model_path" in ocr_model_paths:
                ocr_opts.det_model_path = ocr_model_paths.get("det_model_path")
            if "rec_model_path" in ocr_model_paths:
                ocr_opts.rec_model_path = ocr_model_paths.get("rec_model_path")
            if "cls_model_path" in ocr_model_paths:
                ocr_opts.cls_model_path = ocr_model_paths.get("cls_model_path")
            if "rec_keys_path" in ocr_model_paths:
                ocr_opts.rec_keys_path = ocr_model_paths.get("rec_keys_path")
    elif ocr_engine == "easyocr":
        ocr_opts = EasyOcrOptions(
            lang=ocr_langs,
            force_full_page_ocr=force_full_page_ocr,
        )
    elif ocr_engine == "tesseract":
        ocr_opts = TesseractOcrOptions(
            lang=ocr_langs,
            force_full_page_ocr=force_full_page_ocr,
        )
    else:
        raise ValueError(f"Unsupported ocr_engine: {ocr_engine}")

    # Table structure enrichment
    table_opts = TableStructureOptions(
        do_cell_matching=True,
        mode=TableFormerMode.ACCURATE,
    )

    # Build PDF pipeline options
    pdf_opts = PdfPipelineOptions(
        do_ocr=True,
        ocr_options=ocr_opts,
        do_code_enrichment=enrich.get("code", False),
        do_formula_enrichment=enrich.get("formula", False),
        do_picture_description=enrich.get("picture", False),
        do_table_structure=True,
        table_structure_options=table_opts,
    )

    # Picture description options (API or local VLM)
    if enrich.get("picture", False):
        if use_vlm:
            if use_local_vlm:
                # Local HuggingFace VLM
                if not local_vlm_repo_id:
                    raise ValueError(
                        "Must provide local_vlm_repo_id if use_local_vlm is True"
                    )
                pdf_opts.picture_description_options = PictureDescriptionVlmOptions(
                    repo_id=local_vlm_repo_id,
                    prompt="Describe this image in a few sentences.",
                    generation_config={"max_new_tokens": 120, "do_sample": False},
                )
            else:
                # Remote API (OpenAI, watsonx, etc)
                if not (llm_client and llm_model):
                    raise ValueError(
                        "llm_client and llm_model are required for remote VLM API"
                    )
                pdf_opts.picture_description_options = PictureDescriptionApiOptions(
                    url=llm_client.base_url,
                    headers={"Authorization": f"Bearer {llm_client.api_key}"},
                    params={"model": llm_model},
                    prompt="Describe this image in a few sentences.",
                    timeout=60,
                )

    # Wrap options in PdfFormatOption as required by Docling!
    if use_vlm:
        format_options = {
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pdf_opts,
                pipeline_cls=VlmPipeline,
            )
        }
    else:
        format_options = {
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pdf_opts,
            )
        }

    converter = DocumentConverter(format_options=format_options)

    result = converter.convert(path)
    return result.document.export_to_markdown(), None
