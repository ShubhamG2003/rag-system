from pathlib import Path
from pypdf import PdfReader

import contextlib
import os


def load_pdf(file_path: str | Path) -> list[dict]:
    """Extract text from a PDF, preserving page numbers and source."""

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    reader = PdfReader(path)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stderr(devnull):
                text = page.extract_text() or ""

        if text.strip():
            pages.append({
                "source": path.name,
                "page": page_number,
                "text": text.strip()
            })

    if not pages:
        raise ValueError(
            f"No extractable text found in {path}. "
            "The PDF may be scanned or image-based."
        )

    return pages
