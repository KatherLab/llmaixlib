"""
MIME sniffing that prefers **filetype**.

Order of precedence
-------------------
1. ``filetype`` (signature‑based, lightweight & actively maintained)
2. ``python‑magic`` (libmagic bindings) – optional
3. ``mimetypes.guess_type`` (extension‑based fallback)

This keeps the best characteristics of each approach while avoiding the
heavy libmagic dependency when it isn’t available.
"""

from __future__ import annotations

import mimetypes
from pathlib import Path

import filetype

# ---------------------------------------------------------------------------
# Optional: python‑magic as a secondary content‑based detector
# ---------------------------------------------------------------------------
try:
    import magic  # type: ignore[import-untyped]

    _MAGIC_MIME = magic.Magic(mime=True)
except ImportError:  # pragma: no cover – optional dependency
    _MAGIC_MIME = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def detect_mime(src: Path | str | bytes | bytearray) -> str | None:
    """Best‑effort MIME type for *src* (bytes/bytearray, ``str`` path, or ``Path``).

    Returns ``None`` when the type cannot be determined.
    """
    # ------------------------------------------------------------------ bytes
    if isinstance(src, (bytes, bytearray)):
        # --- filetype --------------------------------------------------
        kind = filetype.guess(src)
        if kind:  # e.g. "application/pdf"
            return kind.mime

        # --- python‑magic ---------------------------------------------
        if _MAGIC_MIME:
            try:
                return _MAGIC_MIME.from_buffer(src)
            except Exception:  # pragma: no cover
                pass

        return None

    # ------------------------------------------------------------- filesystem
    path = Path(src)  # handles str or Path transparently

    # --- filetype on file ----------------------------------------------
    try:
        kind = filetype.guess(path)
        if kind:
            return kind.mime
    except Exception:  # pragma: no cover - corrupt / unreadable file
        pass

    # --- python‑magic on file ------------------------------------------
    if _MAGIC_MIME:
        try:
            return _MAGIC_MIME.from_file(str(path))
        except Exception:  # pragma: no cover
            pass

    # --- extension‑based heuristic -------------------------------------
    return mimetypes.guess_type(path.name)[0]
