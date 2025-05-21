import os
import shutil
import uuid
from urllib.parse import urljoin

import openai
from pathlib import Path
import pymupdf4llm
from markitdown import MarkItDown
import tempfile

from pydantic import AnyUrl

from .utils import (
    string_is_empty_or_garbage,
    pdf_to_images,
    add_text_layer_to_pdf_surya,
    get_full_text_surya,
    markdown_to_pdf,
)


def process_pdf(
    filename: Path | str,
    output: Path | None = None,
    output_file: bool = False,
    pdf_backend: str = "markitdown",
    ocr_backend: str | None = None,
    use_ocr: bool = False,
    force_ocr: bool = False,
    ocr_model: str | None = None,
    ocr_languages: list[str] | None = None,
    client: openai.OpenAI | None = None,
    llm_model: str | None = None,
    verbose: bool = False,
) -> tuple[str, bytes | None]:
    """
    Process a PDF file and extract its text content using the specified PDF processing backend and optional OCR.

    Parameters
    ----------
    filename : Path or str
        Path to the PDF file to process
    output : Path, optional
        Path where processed PDF will be saved, defaults to a temporary file if None
    output_file : bool, default=False
        If True, a tuple (text, File) will be returned instead of just text
    pdf_backend : str, default="markitdown"
        Backend library to use for PDF text extraction, options: "markitdown", "pymupdf4llm"
    ocr_backend : str, optional
        OCR engine to use when use_ocr=True, options: "ocrmypdf", "surya-ocr", "doctr", "paddleOCR", "olmocr"
    use_ocr : bool, default=False
        Whether to apply OCR processing to the PDF
    force_ocr : bool, default=False
        Force OCR processing even if text is already present in the PDF
    ocr_model : str, optional
        Specific OCR model to use (implementation depends on ocr_backend)
    ocr_languages : list[str], optional
        List of language codes for OCR processing (e.g., ["eng", "deu"])
    client : openai.OpenAI, optional
        OpenAI client instance for LLM-based text extraction (required with llm_model)
    llm_model : str, optional
        LLM model identifier to use for text extraction (required with client)
    verbose : bool, default=False

    Returns
    -------
    str
        Extracted text content from the PDF in plain text or markdown format

    Raises
    ------
    FileNotFoundError
        If the input file doesn't exist
    ValueError
        If file format is unsupported, file is empty, output directory doesn't exist,
        or if there are incompatible parameter combinations
    ImportError
        If required dependencies aren't installed
    NotImplementedError
        If requested OCR backend isn't implemented yet
    RuntimeError
        If PDF processing fails
    """

    if isinstance(filename, str):
        filename: Path = Path(filename)
    elif isinstance(filename, Path):
        filename: Path = filename
    else:
        raise ValueError(
            f"Invalid filename type: {type(filename)}. Expected str or Path."
        )

    if not filename.exists():
        raise FileNotFoundError(f"File {filename} does not exist.")
    if filename.suffix != ".pdf":
        raise ValueError(
            f"Unsupported file format: {filename.suffix}. Supported format is .pdf."
        )
    if filename.stat().st_size == 0:
        raise ValueError(f"File {filename} is empty.")
    if output and not output.parent.exists():
        raise ValueError(f"Output directory {output.parent} does not exist.")
    if (llm_model and not client) or (client and not llm_model):
        raise ValueError("Both llm_model and client must be provided together.")
    if (ocr_backend and not use_ocr) or (use_ocr and not ocr_backend):
        raise ValueError("Both ocr_backend and use_ocr must be provided together.")

    tmp_output: Path
    if not output:
        # use a temp dir with a random id as filename
        tmp_output = Path(tempfile.gettempdir()) / f"{uuid.uuid4()}.pdf"
        if verbose:
            print("Output file not specified. Using temporary file:", tmp_output)
    elif output.suffix != ".pdf":
        raise ValueError(
            f"Unsupported output format: {output.suffix}. Supported format is .pdf."
        )
    else:
        tmp_output = output

    if pdf_backend not in ["markitdown", "pymupdf4llm", "docling", "ocr_backend"]:
        raise ValueError(f"Unsupported PDF backend: {pdf_backend}")

    if pdf_backend == "ocr_backend" and not use_ocr:
        raise ValueError(
            "Cannot use the pdf_backend 'ocr_backend' without using OCR. Set use_ocr=True."
        )

    if pdf_backend == "ocr_backend" and ocr_backend == "ocrmypdf":
        raise ValueError(
            "Cannot use the pdf_backend 'ocr_backend' with ocrmypdf (tesseract). Tesseract produces an intermediate PDF,"
            " which needs to be parsed by the pdf_backend. Use 'markitdown' or 'pymupdf4llm' instead."
        )

    extracted_text: str = ""

    if use_ocr:
        if ocr_backend not in ["ocrmypdf", "surya-ocr", "doclingvlm"]:
            raise ValueError(f"Unsupported OCR backend: {ocr_backend}")

        if ocr_backend == "ocrmypdf":
            try:
                print("OCRing PDF with ocrmypdf...")
                import ocrmypdf as ocr

                if shutil.which("ocrmypdf") is None:
                    raise RuntimeError(
                        "ocrmypdf executable not found. Please install it in your system."
                    )

                if force_ocr:
                    ocr.ocr(
                        input_file=filename,
                        output_file=tmp_output,
                        force_ocr=True,
                        language=ocr_languages,
                    )
                else:
                    ocr.ocr(
                        input_file=filename,
                        output_file=tmp_output,
                        language=ocr_languages,
                    )
            except ImportError:
                raise ImportError(
                    "ocrmypdf is not installed. Please install it using 'pip install ocrmypdf'."
                )
            except Exception as e:
                raise RuntimeError(
                    f"An error occurred while processing the PDF with ocrmypdf: {e}"
                )

        elif ocr_backend == "surya-ocr":
            try:
                print("OCRing PDF with surya-ocr...")
                # TODO make this more efficient for multiple documents. Currently, it loads the model for each document.
                from surya.recognition import RecognitionPredictor
                from surya.detection import DetectionPredictor

                recognition_predictor = RecognitionPredictor()
                detection_predictor = DetectionPredictor()

                images = pdf_to_images(filename)

                predictions = recognition_predictor(
                    images,
                    det_predictor=detection_predictor,
                )

                if pdf_backend == "ocr_backend" and not (output or output_file):
                    extracted_text = get_full_text_surya(predictions)
                else:
                    extracted_text = add_text_layer_to_pdf_surya(
                        pdf_path=filename,
                        ocr_results=predictions,
                        output_path=tmp_output,
                    )

            except ImportError:
                raise ImportError(
                    "surya-ocr is not installed. Please install it using 'pip install llmaix[surya]'."
                )
            except Exception as e:
                # Add traceback if verbose
                if verbose:
                    import traceback

                    raise RuntimeError(
                        f"An error occurred while processing the PDF with surya-ocr: {e}. Traceback: {traceback.format_exc()}"
                    )
                else:
                    raise RuntimeError(
                        f"An error occurred while processing the PDF with surya-ocr: {e}"
                    )
        elif ocr_backend == "doclingvlm":
            try:
                print("OCRing PDF with docling...")
                from docling.datamodel.base_models import InputFormat
                from docling.datamodel.pipeline_options import (
                    VlmPipelineOptions,
                    ResponseFormat,
                    ApiVlmOptions,
                )
                from docling.document_converter import (
                    DocumentConverter,
                    PdfFormatOption,
                )
                from docling.pipeline.vlm_pipeline import VlmPipeline

                api_key = os.environ["OPENAI_API_KEY"]
                api_model = os.environ["OPENAI_MODEL"]
                full_url: AnyUrl = AnyUrl(
                    urljoin(os.environ["OPENAI_API_BASE"], "chat/completions")
                )

                def remote_vlm_options(model: str, prompt: str):
                    options = ApiVlmOptions(
                        url=full_url,
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                        params=dict(
                            model=model,
                        ),
                        prompt=prompt,
                        timeout=90,
                        scale=1.0,
                        response_format=ResponseFormat.MARKDOWN,
                    )
                    return options

                pipeline_options = VlmPipelineOptions(
                    enable_remote_services=True  # <-- this is required!
                )

                pipeline_options.vlm_options = remote_vlm_options(
                    model=api_model,
                    prompt="OCR the full page to markdown.",
                )

                doc_converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(
                            pipeline_options=pipeline_options,
                            pipeline_cls=VlmPipeline,
                        )
                    }
                )

                result = doc_converter.convert(filename)

                extracted_text = result.document.export_to_markdown()

                if pdf_backend != "ocr_backend" or (output or output_file):
                    markdown_to_pdf(extracted_text, tmp_output)

            except ImportError:
                raise ImportError(
                    "docling is not installed. Please install it using 'pip install llmaix[docling]'."
                )
            except Exception as e:
                raise RuntimeError(
                    f"An error occurred while processing the PDF with docling: {e}"
                )

        else:
            raise NotImplementedError(
                f"OCR backend {ocr_backend} is not implemented yet."
            )
    else:
        tmp_output: Path = filename

    if pdf_backend == "markitdown":
        if client and llm_model:
            print("Using MarkItDown with LLM...")
            print("LLM Model:", llm_model)
            extracted_text = (
                MarkItDown(client=client, llm_model=llm_model, enable_plugins=True)
                .convert(tmp_output)
                .text_content
            )
        else:
            print("Using MarkItDown without LLM...")
            extracted_text = (
                MarkItDown(enable_plugins=True).convert(tmp_output).text_content
            )
    elif pdf_backend == "pymupdf4llm":
        extracted_text = pymupdf4llm.to_markdown(tmp_output)
    elif pdf_backend == "ocr_backend":
        if not extracted_text:
            raise ValueError(
                "Cannot use the pdf_backend 'ocr_backend' in this configuration. The ocr_backend needs to return text."
            )
        else:
            if verbose:
                print(
                    "Taking shortcut by using the extracted text from the OCR backend directly. No intermediate PDF."
                )
    elif pdf_backend == "docling" and ocr_backend == "doclingvlm":
        pass
    elif pdf_backend == "docling":
        try:
            from docling.document_converter import DocumentConverter

            converter = DocumentConverter()
            result = converter.convert(tmp_output)
            extracted_text = result.document.export_to_markdown()
        except ImportError:
            raise ImportError(
                "docling is not installed. Please install it using 'pip install llmaix[docling]'."
            )
        except Exception as e:
            raise RuntimeError(
                f"An error occurred while processing the PDF with docling: {e}"
            )
    else:
        raise ValueError(f"Unsupported PDF backend: {pdf_backend}")

    if output_file:
        with open(tmp_output, "rb") as f:
            pdf_file = f.read()

    if output:
        # copy tmp_output to output
        if tmp_output and tmp_output != output:
            tmp_output.rename(output)
            if verbose:
                print(f"Temporary file {tmp_output} renamed to {output}.")
    else:
        # remove tmp_output
        if tmp_output and tmp_output.exists():
            tmp_output.unlink()
            if verbose:
                print(f"Temporary file {tmp_output} removed.")

    if output_file:
        return extracted_text, pdf_file
    return extracted_text, None


def preprocess_file(
    filename: Path | str,
    output: Path | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    client: openai.OpenAI | None = None,
    llm_model: str | None = None,
    pdf_backend: str | None = "markitdown",
    ocr_backend: str | None = "ocrmypdf",
    use_ocr: bool = False,
    ocr_model: str | None = None,
    ocr_languages: list[str] | None = None,
    force_ocr: bool = False,
    verbose: bool = False,
) -> str:
    """
    Preprocess document files for LLM input by extracting text content with intelligent fallback to OCR when needed.

    This function handles various document formats and automatically applies OCR when text extraction fails,
    providing clean, structured text ready for LLM processing.

    Parameters
    ----------
    filename : Path
        Path to the input file (.pdf, .txt, or .docx)
    output : Path, optional
        Path where processed output will be saved
    verbose : bool, default=False
        Enable verbose logging during processing
    base_url : str, optional
        Base URL for OpenAI-compatible API (alternative to providing client)
    api_key : str, optional
        API key for OpenAI-compatible API (required with base_url or llm_model). Used for image descriptions by markitdown.
    client : openai.OpenAI, optional
        Preconfigured OpenAI client instance (alternative to base_url+api_key)
    llm_model : str, optional
        LLM model identifier for advanced text extraction
    pdf_backend : str, default="markitdown"
        Backend library for PDF processing ("markitdown", "pymupdf4llm" and "docling")
    ocr_backend : str, default="ocrmypdf"
        OCR engine to use when text extraction fails or use_ocr=True
        Options: "ocrmypdf", "surya-ocr", "doclingvlm
    use_ocr : bool, default=False
        Whether to apply OCR processing to the document
    ocr_model : str, optional
        Specific OCR model name to use with selected OCR backend
    ocr_languages : list[str], optional
        List of language codes for OCR processing (e.g., ["eng", "deu"])
    force_ocr : bool, default=False
        Force OCR processing even if text is already detected

    Returns
    -------
    str
        Extracted text content from the document

    Raises
    ------
    FileNotFoundError
        If the input file doesn't exist
    ValueError
        If file format is unsupported, file is empty, output directory doesn't exist,
        if PDF is empty and OCR wasn't requested, or if there are incompatible parameter combinations
    """

    if verbose:
        print(
            f"Preprocessing {filename} with output={output}, verbose={verbose}, base_url={base_url}, and api_key={api_key}"
        )

    if isinstance(filename, str):
        filename = Path(filename)

    # Check if the file exists
    if not filename.exists():
        raise FileNotFoundError(f"File {filename} does not exist.")
    # Check if the file is a valid format (e.g., .txt, .json)
    if filename.suffix not in [".pdf", ".txt", ".docx"]:
        raise ValueError(
            f"Unsupported file format: {filename.suffix}. Supported formats are .txt and .json."
        )

    # Check if the file is empty
    if filename.stat().st_size == 0:
        raise ValueError(f"File {filename} is empty.")

    # Check if the output path is valid
    if output and not output.parent.exists():
        raise ValueError(f"Output directory {output.parent} does not exist.")

    # Check if either api_key or llm_model is provided, then both are required
    if (llm_model and not api_key) or (api_key and not llm_model):
        raise ValueError("Both llm_model and api_key must be provided together.")

    if base_url and api_key:
        client = openai.OpenAI(base_url=base_url, api_key=api_key)
    elif api_key:
        client = openai.OpenAI(api_key=api_key)

    if force_ocr and not use_ocr:
        raise ValueError(
            "force_ocr is set to True, but use_ocr is set to False. Please also set use_ocr=True."
        )

    extracted_text: str = ""

    if filename.suffix == ".pdf":
        extracted_text = pymupdf4llm.to_markdown(filename)

        if string_is_empty_or_garbage(extracted_text) and use_ocr:
            print("PDF: No text found, trying OCR...")
            extracted_text, _ = process_pdf(
                filename,
                output,
                pdf_backend=pdf_backend or "markitdown",
                ocr_backend=ocr_backend,
                use_ocr=use_ocr,
                ocr_model=ocr_model,
                ocr_languages=ocr_languages,
                client=client,
                llm_model=llm_model,
                force_ocr=force_ocr,
                verbose=verbose,
            )
        elif string_is_empty_or_garbage(extracted_text) and not use_ocr:
            raise ValueError(f"PDF {filename} is empty and no OCR was requested.")
        elif not string_is_empty_or_garbage(extracted_text) and use_ocr:
            print("PDF: Text found, OCR will be re-done and forced.")
            extracted_text, _ = process_pdf(
                filename,
                output,
                pdf_backend=pdf_backend or "markitdown",
                ocr_backend=ocr_backend,
                use_ocr=use_ocr,
                ocr_model=ocr_model,
                ocr_languages=ocr_languages,
                force_ocr=True,
                client=client,
                llm_model=llm_model,
                verbose=verbose,
            )
        elif not string_is_empty_or_garbage(extracted_text):
            pass

    return extracted_text
