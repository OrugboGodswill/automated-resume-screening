from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

import os
import shutil


def _ocr_available() -> bool:
    """Return True only when both pytesseract and the Tesseract binary are present."""
    try:
        import pytesseract  # noqa: F401

        # Check system PATH first, then the common Windows install location
        if shutil.which("tesseract"):
            return True
        win_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(win_path):
            import pytesseract as pt
            pt.pytesseract.tesseract_cmd = win_path
            return True
    except ImportError:
        pass
    return False


def _ocr_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract text from scanned PDF pages using Tesseract OCR (Windows / Docker only)."""
    if not _ocr_available():
        return ""

    text_chunks = []
    try:
        import pypdfium2 as pdfium
        import pytesseract
        from PIL import Image  # noqa: F401

        pdf = pdfium.PdfDocument(pdf_bytes)
        for page in pdf:
            pil_img = page.render(scale=2).to_pil()
            ocr_text = pytesseract.image_to_string(pil_img)
            if ocr_text and ocr_text.strip():
                text_chunks.append(ocr_text.strip())
        if text_chunks:
            return "\n".join(text_chunks).strip()
    except Exception:
        pass

    return ""


def extract_pdf_text(source: bytes | str | Path) -> str:
    """Extract text from a PDF path or uploaded bytes with automatic OCR fallback."""
    if isinstance(source, bytes):
        raw_bytes = source
    else:
        raw_bytes = Path(source).read_bytes()

    text = ""
    try:
        reader = PdfReader(BytesIO(raw_bytes))
        text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    except Exception:
        text = ""

    # If the PDF is scanned or image-only, attempt OCR (only works where Tesseract is installed)
    if len(text.strip()) < 30:
        ocr_text = _ocr_pdf_bytes(raw_bytes)
        if len(ocr_text) > len(text):
            return ocr_text

    return text


def extract_text_from_upload(uploaded_file) -> str:
    return extract_pdf_text(uploaded_file.getvalue())
