from io import BytesIO
from pathlib import Path

from pypdf import PdfReader


import os
import shutil

def _ocr_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract text from scanned PDF pages using Tesseract OCR."""
    text_chunks = []
    try:
        import pypdfium2 as pdfium
        import pytesseract
        from PIL import Image

        if not shutil.which("tesseract"):
            win_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            if os.path.exists(win_tesseract):
                pytesseract.pytesseract.tesseract_cmd = win_tesseract

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

    # If the PDF is scanned or image-only, extract_text returns empty or negligible text
    if len(text.strip()) < 30:
        ocr_text = _ocr_pdf_bytes(raw_bytes)
        if len(ocr_text) > len(text):
            return ocr_text

    return text


def extract_text_from_upload(uploaded_file) -> str:
    return extract_pdf_text(uploaded_file.getvalue())

