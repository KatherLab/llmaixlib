def pytest_addoption(parser):
    parser.addoption(
        "--ocr-backend",
        action="append",
        dest="ocr_backend",
        default=None,  # Set default to None, not a list
        help="Specify OCR backends(s) to test",
        choices=["ocrmypdf", "marker", "doclingvlm"],
    )
    parser.addoption(
        "--pdf-backend",
        action="append",
        dest="pdf_backend",
        default=None,  # Set default to None, not a list
        help="Specify PDF backends(s) to test",
        choices=["markitdown", "pymupdf4llm", "docling", "ocr_backend"],
    )


def pytest_generate_tests(metafunc):
    """Dynamically parameterize tests that use ocr_backend"""
    if "ocr_backend" in metafunc.fixturenames:
        backends = metafunc.config.getoption("--ocr-backend")
        if backends is None or len(backends) == 0:
            backends = ["ocrmypdf", "marker"]  # Default to both backends
        metafunc.parametrize("ocr_backend", backends, ids=lambda x: x)
    if "pdf_backend" in metafunc.fixturenames:
        backends = metafunc.config.getoption("--pdf-backend")
        if backends is None or len(backends) == 0:
            backends = ["markitdown", "pymupdf4llm", "docling", "ocr_backend"]
        metafunc.parametrize("pdf_backend", backends, ids=lambda x: x)
