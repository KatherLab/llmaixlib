![Tests](https://github.com/KatherLab/llmaixlib/actions/workflows/tests.yml/badge.svg?branch=main)

# LLMAIx (v2) Library

The llmaix library contains the core functionality of the LLMAIx framework.

>[!CAUTION]
> The interface of the library is still in development and may change in the future. The library is not yet ready for production use.

## Features

- **Preprocessing**: The library provides tools for extracting text from various file formats, including PDF, DOCX, and TXT. It can apply OCR to images and PDFs, using tesseract, surya-ocr and VLMs via docling.

- **Information Extraction**: The library provides a wrapper helping you to get a JSON response from an LLM. All OpenAI-API compatible models are supported!

## Installation

```bash
pip install llmaix
```

To install dependencies for docling: 

```bash
pip install llmaix[docling]
```

Available Dependency groups: `surya`,`docling`

To install all dependencies:

```bash
pip install llmaix[all]
```

## Usage

### CLI

```bash
llmaix --help
```

### Python

**Preprocessing a PDF file without OCR:**
```python
from llmaix import preprocess_file

filename = "tests/testfiles/987462_text.pdf"

extracted_text = preprocess_file(filename)
```

**Preprocessing a PDF file with OCR:**
```python
from llmaix import preprocess_file

filename = "tests/testfiles/987462_notext.pdf"

extracted_text = preprocess_file(filename, use_ocr=True, ocr_backend="ocrmypdf")
```

| OCR Backends                                             | Comment                                                                                       |
|----------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| [ocrmypdf](https://github.com/ocrmypdf/OCRmyPDF)         | Uses tesseract. Needs to be installed on the system first!                                    |
| [surya-ocr](https://github.com/VikParuchuri/surya)       | Uses surya-ocr. Runs models via transformers library locally.                                 |
| [doclingvlm](https://github.com/docling-project/docling) | Uses docling to perform OCR using a VLM. Configure the model like for information extraction! |

| PDF Backends                                                         | Comment                                                                                                                        |
|----------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| [pymupdf4llm](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/) | Uses pymupdf to extract text as markdown from PDF files.                                                                       |
| [markitdown](https://github.com/microsoft/markitdown)                | Uses markitdown to extract text as markdown from PDF files.                                                                    |
| [docling](https://github.com/docling-project/docling)                | Uses docling to extract text as markdown from PDF files. Caution: docling itself might apply OCR even if you don't specify it. |
| ocr_backend                                                          | Directly use the text output from the OCR backend. Incompatible with ocrmypdf.                                                 |

**Extracting information from a text:**

1. Provide a .env file with your OpenAI API key:
```bash
echo "OPENAI_API_KEY=your_openai_api_key" > .env
```
2. (Optional) To use a custom base url, set the `OPENAI_API_BASE` environment variable:
```bash
echo "OPENAI_API_BASE=https://your_custom_base_url/v1" >> .env
```

2. (Optional) Configure model in the `.env` file:
```bash
echo "OPENAI_MODEL=gpt-4o-2024-08-06" >> .env
```

3. Use the `extract_info` function to extract information from a text. In this example, a pydantic model is used to define the expected output format. The output will be a JSON object.
```python
from llmaix import extract_info
from pydantic import BaseModel

extracted_text = "The KatherLab is a research group at the University of Technology Dresden, lead by Prof. Jakob N. Kather."

class LabInformation(BaseModel):
    name: str
    location: str
    lead: str

extracted_info = extract_info(
    prompt=f"Extract the name, location and lead of the lab from the following text: {extracted_text}",
    llm_model="Llama-4-Maverick-17B-128E-Instruct-FP8",
    pydantic_model=LabInformation,
)
```

Clone the repository and install the dependencies:
```bash
git clone https://github.com/KatherLab/LLMAIx-v2.git
cd LLMAIx-v2
uv sync
```

## Tests

Run the tests using the following command:

```bash
uv run pytest
```

Example to just run test for preprocessing with the ocrmypdf backend:
```bash
uv run pytest tests/test_preprocess.py --ocr-backend ocrmypdf
```
