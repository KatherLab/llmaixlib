![Tests](https://github.com/KatherLab/llmaixlib/actions/workflows/tests.yml/badge.svg?branch=main)

# LLMAIxLib

**LLMAIxLib** is a Python toolkit for automated document preprocessing (including OCR) and information extraction using large language models. It is designed for users who need to extract structured facts from arbitrary documents (PDF, DOCX, images, etc.) and output them as Markdown, plain text, or validated JSON.

>[!CAUTION]
> 
> Under active development. Best suited for research or prototyping. Always validate results.

---

## 🚀 What LLMAIxLib Does

* **Preprocessing:** Extracts human-readable Markdown or plain text from a wide range of document types, automatically falling back to OCR for scanned or image-based files.
* **Information Extraction:** Uses a large language model (LLM) to transform unstructured or semi-structured text into structured data—validated by Pydantic models or JSON Schema—via an OpenAI-compatible API.

---

## ❗ What You Need

* **Python ≥3.12**
* **OCR tools:** Tesseract (for OCRmyPDF), a GPU for faster OCR (Marker and PaddleOCR)
* **OpenAI-compatible API endpoint:**
  Required for information extraction! This can be:

  * The official OpenAI API (or Azure OpenAI or ...)
  * A self-hosted API that matches the OpenAI chat/completions format, e.g. `vllm`, `llama.cpp` server, or other compatible backends
  * Your endpoint **must support structured output** (based on JSON schema).

---

## 🛠 Installation

Install base:

```bash
pip install llmaix
```

Add extras for advanced features:

```bash
pip install llmaix[docling]      # advanced layout + VLM support
pip install llmaix[marker]        # Marker (surya-OCR)
pip install llmaix[paddleocr]    # PaddleOCR
pip install llmaix[docling,marker,paddleocr] # all extras
```

---

## 📚 Usage

### CLI Examples

Environment variables are the recommended way to provide your API settings (see below).

```bash
llmaix preprocess file.pdf                # extract as Markdown, fast mode
llmaix preprocess scan.pdf --force-ocr --ocr-engine paddleocr -o out.md
llmaix preprocess paper.pdf --mode advanced --enable-picture-description
llmaix extract --input "Patient was a 73-year-old male..." --json-schema patient_schema.json
```

### Python API Example

```python
from llmaix.preprocess import DocumentPreprocessor
from llmaix import extract_info
from pydantic import BaseModel

# Preprocessing: get Markdown or text
proc = DocumentPreprocessor(mode="advanced", ocr_engine="marker")
markdown = proc.process("scan.pdf")

# Information extraction: structured JSON from text via LLM
class PersonInfo(BaseModel):
    name: str
    affiliation: str
    position: str

result = extract_info(
    prompt="Alice Smith is a Professor of AI at TU Dresden.",
    pydantic_model=PersonInfo,
    llm_model="o4-mini"
)
print(result.json(indent=2))
```

---

## 🔑 API Configuration

You must provide your LLM API settings by **environment variable** (recommended) or CLI flag:

```bash
export OPENAI_API_KEY=sk-xxx
export OPENAI_API_BASE=https://api.example.com/v1  # optional, default: OpenAI endpoint
export OPENAI_MODEL=gpt-4                         # optional, default: set in CLI or code
```

Or pass directly:

```bash
llmaix extract --input "..." --llm-model llama-3-8b-instruct --base-url http://localhost:8000/v1 --api-key sk-xxx --json-schema schema.json
```

---

## 🗂 Architecture Overview

### **Preprocessing**

* **DocumentPreprocessor**:

  * Detects MIME type and routes to the appropriate handler.
  * For PDFs: tries fast text extraction first, falls back to OCR (OCRmyPDF, PaddleOCR, Marker) if needed.
  * DOCX, TXT, and image formats supported.
  * Advanced mode: integrates Docling for tables, formulas, and (optionally) vision-language model for image captioning.
* **OCR Engines**: Pluggable; use Tesseract, Marker, PaddleOCR as needed.

### **Information Extraction**

* **extract\_info**:

  * Sends text and a schema (Pydantic or JSON Schema) to an OpenAI-compatible API endpoint.
  * Validates output as structured JSON.
  * CLI can load schema from file or as literal string.
  * *Your API endpoint must support structured outputs!*
  * Can be used with hosted (OpenAI, Azure) or self-hosted (e.g. llama.cpp, vllm) models that follow the OpenAI API.

---

## 🧩 JSON Schema Example

```json
{
  "type": "object",
  "properties": {
    "experiment_id": { "type": "string" },
    "date": { "type": "string", "format": "date" },
    "findings": { "type": "array", "items": { "type": "string" } }
  },
  "required": ["experiment_id", "findings"]
}
```

---

## ✅ Quick Checklist

1. **Set up API credentials** (see above).
2. **Install OCR backends** as required for your documents.
3. **Use `llmaix preprocess`** for robust text/Markdown extraction from documents.
4. **Use `llmaix extract`** (with prompt + schema or model) for LLM-powered structured extraction.

---

## 🧪 Testing

```bash
uv run pytest
uv run pytest tests/test_preprocess.py -k paddleocr
```

---

## ⚠️ Caveats & Notes

* Preprocessing only: No LLM API needed if you just want Markdown/text from documents.
* Information extraction: Requires an OpenAI-compatible API endpoint that supports structured outputs.
* If your LLM or endpoint does **not** support structured output via `reponse_format`, information extraction will not work as expected.
  * You can still use the `extract_info` function and provide a `prompt` or `system_prompt` argument which teaches the model to respond with valid JSON only in the desired format!

---

## 📄 License

MIT.

Contributions welcome.

Repo: [github.com/KatherLab/llmaixlib](https://github.com/KatherLab/llmaixlib)

---
