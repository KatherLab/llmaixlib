def pytest_addoption(parser):
    parser.addoption(
        "--ocr-backend",
        action="append",
        dest="ocr_backend",
        default=None,  # Set default to None, not a list
        help="Specify OCR backends(s) to test",
        choices=["ocrmypdf", "surya-ocr"]
    )

def pytest_generate_tests(metafunc):
    """Dynamically parameterize tests that use ocr_backend"""
    if "ocr_backend" in metafunc.fixturenames:
        backends = metafunc.config.getoption("--ocr-backend")
        if backends is None or len(backends) == 0:
            backends = ["ocrmypdf", "surya-ocr"]  # Default to both backends
        metafunc.parametrize("ocr_backend", backends, ids=lambda x: x)
