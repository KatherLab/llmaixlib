"""
Utility helpers for the document‑preprocessing pipeline.

Sections
--------
* Markdown → PDF conversion helpers
* Bounding‑box scaling / font‑size estimation
* String‑quality heuristics (entropy‑based garbage detection)
* PDF → PIL image conversion

All functions are pure and side‑effect free except where noted.
"""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from markdown_it import MarkdownIt
from PIL import Image

# ---------------------------------------------------------------------------
# Markdown → PDF helpers
# ---------------------------------------------------------------------------


class MarkdownSection:
    """A logical section of Markdown to be rendered into PDF."""

    def __init__(
        self,
        text: str,
        toc: bool = True,
        root: str = ".",
        paper_size: str = "A4",
        margins: tuple[int, int, int, int] = (36, 36, -36, -36),
    ) -> None:
        self.text = text
        self.toc = toc
        self.root = root
        self.paper_size = paper_size
        self.margins = margins


class PdfConverter:
    """
    Convert one or more :class:`MarkdownSection` objects into a single PDF.

    Parameters
    ----------
    toc_depth:
        Maximum heading level to include in the table of contents (0 = none).
    parser_mode:
        Mode passed to `markdown_it.MarkdownIt` (default: "commonmark").
    optimize:
        If *True*, runs `fitz.Document.ez_save` for size optimization.
    """

    meta = {
        "creationDate": fitz.get_pdf_now(),
        "modDate": fitz.get_pdf_now(),
        "creator": "llmaix library using PyMuPDF and markdown-it",
        "producer": None,
        "title": None,
        "author": None,
        "subject": None,
        "keywords": None,
    }

    def __init__(
        self,
        toc_depth: int = 6,
        parser_mode: str = "commonmark",
        optimize: bool = False,
    ) -> None:
        self.toc_depth = toc_depth
        self.toc: list[list] = []
        self.parser = MarkdownIt(parser_mode).enable("table")
        self.optimize = optimize
        self.buffer = io.BytesIO()
        self.writer = fitz.DocumentWriter(self.buffer)
        self.page_count = 0
        self.links: list[Any] = []

    # -- internal ----------------------------------------------------------------

    @staticmethod
    def _position_recorder(
        position: Any,
    ) -> None:  # callback for Story.element_positions
        position.page_num = position.pdf.page_count
        position.pdf.links.append(position)
        if not position.open_close & 1:
            return
        if not position.toc:
            return
        if 0 < position.heading <= position.pdf.toc_depth:
            position.pdf.toc.append(
                (
                    position.heading,
                    position.text,
                    position.pdf.page_count,
                    position.rect[1],
                )
            )

    # -- public ------------------------------------------------------------------

    def add_markdown(self, section: MarkdownSection, css: str | None = None) -> str:
        """Render *section* onto one or more new pages."""
        rect = fitz.paper_rect(section.paper_size)
        area = rect + section.margins
        html = self.parser.render(section.text)
        story = fitz.Story(html=html, archive=section.root, user_css=css)
        more = 1
        while more:
            self.page_count += 1
            device = self.writer.begin_page(rect)
            more, _ = story.place(area)
            story.element_positions(
                self._position_recorder, {"toc": section.toc, "pdf": self}
            )
            story.draw(device)
            self.writer.end_page()
        return html

    def save_to_file(self, file_path: str | Path) -> None:
        """Write accumulated pages to *file_path*."""
        self.writer.close()
        self.buffer.seek(0)
        doc = fitz.Story.add_pdf_links(self.buffer, self.links)
        doc.set_metadata(self.meta)
        if self.toc_depth > 0:
            doc.set_toc(self.toc)
        if self.optimize:
            doc.ez_save(str(file_path))
        else:
            doc.save(str(file_path))
        doc.close()


def markdown_to_pdf(markdown_text: str, output_path: str | Path) -> Path:
    """
    Convenience wrapper: convert *markdown_text* straight to a PDF file.
    """
    output_path = Path(output_path)
    converter = PdfConverter()
    converter.add_markdown(MarkdownSection(markdown_text))
    converter.save_to_file(output_path)
    return output_path


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def scale_bbox(bbox: list[float], src_dpi: int = 96, dst_dpi: int = 72) -> list[float]:
    """
    Scale bounding‑box coordinates when DPI changes.

    Parameters
    ----------
    bbox:
        ``[x1, y1, x2, y2]`` source rectangle.
    src_dpi / dst_dpi:
        Source and destination DPI values.

    Returns
    -------
    Scaled bbox as a list of four floats.
    """
    scale_factor = dst_dpi / src_dpi
    return [coord * scale_factor for coord in bbox]


def estimate_font_size(
    bbox_width: float,
    text_length: int,
    char_width_to_height_ratio: float = 0.5,
) -> float:
    """
    Roughly guess a font‑size that will fit *text_length* characters in *bbox_width*.

    Used when inserting an invisible OCR text layer.
    """
    if text_length == 0:
        return 12.0
    avg_char_width = bbox_width / text_length
    return avg_char_width / char_width_to_height_ratio


# ---------------------------------------------------------------------------
# String quality / garbage detection
# ---------------------------------------------------------------------------


def _shannon_entropy(s: str) -> float:
    """Compute Shannon entropy of *s* using log2."""
    if not s:
        return 0.0
    counts: dict[str, int] = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    length = len(s)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())


_ENTROPY_THRESHOLD = 1.2  # tweak empirically


def string_is_empty_or_garbage(s: str) -> bool:
    """
    Heuristic filter for useless text extracted from PDFs (e.g. scanner artefacts).

    Returns *True* when *s* looks empty or “garbage”; otherwise *False*.
    """
    if not s or s.isspace():
        return True

    compact = "".join(s.split())
    if len(compact) < 3:
        return True

    entropy = _shannon_entropy(compact[:1024])  # bound runtime on huge strings
    return entropy < _ENTROPY_THRESHOLD


# ---------------------------------------------------------------------------
# PDF → image conversion
# ---------------------------------------------------------------------------


def pdf_to_images(filename: str | Path) -> list[Image.Image]:
    """
    Convert every page of a PDF to a PIL Image.

    Raises
    ------
    FileNotFoundError
        If *filename* does not exist.
    ValueError
        If *filename* is not a PDF or cannot be opened.
    """
    file_path = Path(filename)
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    if file_path.suffix.lower() != ".pdf":
        raise ValueError(f"File must be a PDF: {file_path}")

    try:
        pdf_document = fitz.open(file_path)
    except Exception as e:  # pragma: no cover
        raise ValueError(f"Failed to open PDF file: {e}") from e

    images: list[Image.Image] = []
    for page in pdf_document:
        pix = page.get_pixmap(alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        images.append(img)
    pdf_document.close()
    return images
