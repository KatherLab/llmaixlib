"""High‑level datamodel used across the preprocessing pipeline.

Switched from ``dataclasses`` to **Pydantic v2** for stronger validation,
JSON serialisability and future extensibility (e.g. settings import/export).
"""

from __future__ import annotations

import hashlib
import mimetypes
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _sha256(fp: Path, chunk: int = 8192) -> str:
    h = hashlib.sha256()
    with fp.open("rb") as fh:
        for chunk_bytes in iter(lambda: fh.read(chunk), b""):
            h.update(chunk_bytes)
    return h.hexdigest()


# ---------------------------------------------------------------------
# Pydantic model
# ---------------------------------------------------------------------
class Document(BaseModel):
    """A **single logical document** flowing through the pipeline."""

    raw_path: Path = Field(..., description="Filesystem location of the source file")
    mime: str = Field(..., description="Best‑effort MIME type, e.g. application/pdf")
    text: str = Field("", description="Extracted text (Markdown or plain)")
    ocr_pdf: Path | None = Field(
        None,
        description="Path to OCR‑layered PDF if such derivative is produced",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    # ---------------- Pydantic config ---------------

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True,
    )

    # ---------------- Validators --------------------

    @field_validator("mime")
    @classmethod
    def _strip_mime(cls, v: str) -> str:
        # Normalise “text/plain; charset=utf‑8” → “text/plain”
        return v.split(";")[0].strip().lower() if v else v

    # ---------------- Constructors ------------------
    @classmethod
    def from_path(cls, path: Path, mime: str | None = None) -> "Document":
        if not path.exists():
            raise FileNotFoundError(path)
        mime = mime or (
            mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        )
        return cls(
            raw_path=path,
            mime=mime,
            metadata={"sha256": _sha256(path)},
        )
