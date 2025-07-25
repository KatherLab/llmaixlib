![Tests](https://github.com/KatherLab/llmaixlib/actions/workflows/tests.yml/badge.svg?branch=main)

# LLMAIxv2 Library

**LLMAIx** is an end‑to‑end toolkit for turning raw documents into structured knowledge with large‑language models. It now features a modular, Pydantic‑validated preprocessing core, richer OCR choices, and a revamped CLI.

> **Status:** the public API is still stabilising; expect small breaking changes before a 2.0 stable release.

## ✨ Key capabilities

* **Robust preprocessing** –extract Markdown or plain text from PDFs, DOCX, TXT and images. The pipeline tries cheap text extraction first (PyMuPDF) and falls back to OCR (OCR‑my‑PDF/PaddleOCR/Surya) only when needed).
* **Layout‑aware enrichment** – advanced mode plugs into the Docling pipeline for tables, formulas and picture descriptions, optionally powered by a local or remote vision‑language model.
* **MIME‑aware loading** – files (or byte buffers) are classified with **python‑magic** so even extension‑less uploads are handled correctly.
* **Information extraction** – send arbitrary prompts + a Pydantic schema and get back valid JSON, using any OpenAI‑compatible LLM endpoint.
* **CLI utilities** – one‑command document conversion (`llmaix preprocess`) and structured‑info extraction (`llmaix extract`).
* **Extensible** – register new back‑ends (e.g. EPUB) with a single decorator; models and OCR engines can be swapped freely.

## 🛠 Installation

```bash
pip install llmaix          # base
pip install llmaix[docling] # + Docling/VLM extras
pip install llmaix[surya]   # + Surya‑OCR
pip install llmaix[all]     # everything
```

If you need GPU PaddleOCR:

```bash
uv pip install \
  --index-url https://www.paddlepaddle.org.cn/packages/stable/cu129/ \
  paddlepaddle-gpu==3.1.0
```

PaddleOCR supports 80+ languages out‑of‑the‑box).
For MIME detection install **libmagic** (Linux/macOS) or `python-magic-win64` on Windows).

## 🚀 Quick start

### CLI

```bash
llmaix preprocess myscan.pdf                 # fast mode, auto‑OCR
llmaix preprocess doc.pdf --mode advanced \
    --enable-picture-description             # Docling + VLM captions
llmaix preprocess scan.pdf --force-ocr \
    --ocr-engine paddleocr -o out.md
llmaix extract -i "Acme Inc. raised $10 M..." # JSON extraction
```

### Python API

```python
from llmaix.preprocess import DocumentPreprocessor

# 1) simple PDF (born digital)
text = DocumentPreprocessor(mode="fast").process("report.pdf")

# 2) scanned PDF with multilingual OCR
proc = DocumentPreprocessor(
    mode="advanced",
    ocr_engine="surya",
    force_ocr=True,
    enable_picture_description=True,
    use_local_vlm=True,
    local_vlm_repo_id="HuggingFaceTB/SmolVLM-256M-Instruct",
    vlm_prompt="Please describe this document in detail.",
)
markdown = proc.process("scan_no_text.pdf")
```

**Information extraction**

```python
from llmaix import extract_info
from pydantic import BaseModel

class LabInfo(BaseModel):
    name: str
    location: str
    lead: str

sentence = (
    "The KatherLab is a research group at TU Dresden led by Prof. Jakob N. Kather."
)
json_out = extract_info(
    prompt=f"Extract lab facts: {sentence}",
    pydantic_model=LabInfo,
    llm_model="gpt-4o-mini"
)
print(json_out.json(indent=2))
```

## ⚙️ Back‑end matrix

| Task                | Engine                 | Notes                                                          |
| ------------------- | ---------------------- |----------------------------------------------------------------|
| **Text extraction** | PyMuPDF‑for‑LLM        | Fast Markdown conversion from PDFs |
|                     | Docling                | Layout‑aware; optional VLM captions       |
| **OCR**             | OCR‑my‑PDF (Tesseract) | Strong PDF/A support            |
|                     | Surya‑OCR              | Local transformer OCR, 90 + langs                |
|                     | PaddleOCR PP‑Structure | Table & formula detection                        |
| **MIME sniffing**   | python‑magic           | libmagic signatures                                |
| (optional)          | filetype               | pure‑Python fallback                             |

## 🧪 Tests

Clone and run:

```bash
git clone https://github.com/KatherLab/LLMAIx-v2.git
cd LLMAIx-v2
uv sync
uv run pytest          # full suite
```

You can focus on a backend:

```bash
uv run pytest tests/test_preprocess.py -k paddleocr
```