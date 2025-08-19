import io
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import fitz
from markdown_it import MarkdownIt
from PIL import Image


class MarkdownSection:
    def __init__(
        self,
        text: str,
        toc: bool = True,
        root: str = ".",
        paper_size: str = "A4",
        margins: Tuple[int, int, int, int] = (36, 36, -36, -36),
    ):
        self.text = text
        self.toc = toc
        self.root = root
        self.paper_size = paper_size
        self.margins = margins


class PdfConverter:
    meta = {
        "creationDate": fitz.get_pdf_now(),
        "modDate": fitz.get_pdf_now(),
        "creator": "llmaix library using pymupdf and markdown-it",
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
    ):
        self.toc_depth = toc_depth
        self.toc = []
        self.parser = MarkdownIt(parser_mode).enable("table")
        self.optimize = optimize
        self.buffer = io.BytesIO()
        self.writer = fitz.DocumentWriter(self.buffer)
        self.page_count = 0
        self.links = []

    @staticmethod
    def _position_recorder(position):
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

    def add_markdown(self, section: MarkdownSection, css: Optional[str] = None) -> str:
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

    def save_to_file(self, file_path: Union[str, Path]) -> None:
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


def scale_bbox(bbox: list[float], src_dpi: int = 96, dst_dpi: int = 72) -> list[float]:
    """
    Scales a bounding box from one DPI to another.

    Args:
        bbox (List[float]): The bounding box coordinates [x1, y1, x2, y2].
        src_dpi (int, optional): The source DPI. Defaults to 96.
        dst_dpi (int, optional): The destination DPI. Defaults to 72.

    Returns:
        List[float]: The scaled bounding box coordinates.
    """
    scale_factor = dst_dpi / src_dpi
    return [coord * scale_factor for coord in bbox]


def estimate_font_size(
    bbox_width: float, text_length: int, char_width_to_height_ratio: float = 0.5
) -> float:
    """
    Estimates the font size based on the bounding box width and text length.

    Args:
        bbox_width (float): The width of the bounding box.
        text_length (int): The length of the text.
        char_width_to_height_ratio (float, optional): The ratio of character width to height. Defaults to 0.5.

    Returns:
        float: The estimated font size.
    """
    if text_length == 0:  # Prevent division by zero
        return 12.0  # Default font size if no text is present
    avg_char_width = bbox_width / text_length
    font_size = avg_char_width / char_width_to_height_ratio
    return font_size


def pdf_to_images(
    filename: Path | str,
) -> list[Image.Image]:
    """
    Convert a PDF file to a list of PIL Image objects.

    Args:
        filename: Path to the PDF file, can be a string or Path object

    Returns:
        A list of PIL Image objects, one for each page in the PDF

    Raises:
        FileNotFoundError: If the specified PDF file doesn't exist
        ValueError: If the file is not a valid PDF
    """
    # Convert to Path object if string is provided
    file_path = Path(filename) if isinstance(filename, str) else filename

    # Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    # Check if file is a PDF
    if file_path.suffix.lower() != ".pdf":
        raise ValueError(f"File must be a PDF: {file_path}")

    # Open the PDF file
    try:
        pdf_document = fitz.open(file_path)
    except Exception as e:
        raise ValueError(f"Failed to open PDF file: {e}")

    images = []

    # Iterate through each page
    for page_num in range(len(pdf_document)):
        # Get the page
        page = pdf_document.load_page(page_num)

        # Convert page to a pixmap (image)
        pix = page.get_pixmap(alpha=False)

        # Convert pixmap to PIL Image
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

        # Add image to list
        images.append(img)

    # Close the PDF document
    pdf_document.close()

    return images


def string_is_empty_or_garbage(s: str) -> bool:
    """Check if a string is empty, contains only whitespace, or scanner artifacts.

    This function detects various forms of "garbage" text that might be extracted
    from scanned PDFs, including:
      - Empty strings
      - Whitespace-only strings (spaces, tabs, etc.)
      - Strings with only linebreaks
      - Strings with only control characters
      - Common scanner artifacts like isolated dots, dashes, or repeated symbols
      - Patterns of whitespace with random punctuation

    Args:
        s: Input string from a potentially scanned PDF document

    Returns:
        bool: True if the string is empty or contains only noise, False otherwise
    """
    if not s:
        return True

    # Check if string contains only whitespace or control characters
    if all(c.isspace() or ord(c) < 32 for c in s):
        return True

    # Remove all whitespace and check if remaining content is meaningful
    stripped = "".join(s.split())

    # Empty after stripping
    if not stripped:
        return True

    # Check for repetitive patterns (like '...', '---', etc.)
    if len(set(stripped)) <= 2 and len(stripped) > 3:
        # Allow for one or two unique characters, but requires multiple instances
        return True

    # Check for isolated punctuation and symbols commonly added by scanners
    if all(c in ".,-_=+*/\\|:;#@!?~^()[]{}'\"`<>" for c in stripped):
        return True

    # Check for single-character noise
    if len(stripped) <= 2:
        return True

    # Consider strings with too low text-to-whitespace ratio as noise
    # This catches patterns like "- - - - -" or ". . . . ." that might have meaning
    # but are more likely scanner artifacts
    if len(stripped) < len(s) / 5 and len(s) > 10:
        return True

    return False


def markdown_to_pdf(markdown_text, output_path: Path | str) -> Path:
    """
    Convert markdown text to a PDF file.

    Args:
        markdown_text (str): The markdown text to convert
        output_path (str): Path where the PDF file will be saved

    Returns:
        str: Path to the created PDF file
    """

    if not isinstance(output_path, Path):
        output_path = Path(output_path)

    converter = PdfConverter()
    section = MarkdownSection(markdown_text)
    converter.add_markdown(section)
    converter.save_to_file(output_path)

    return output_path
