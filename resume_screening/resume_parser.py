import asyncio
import concurrent.futures
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader


def _run_async(coro_fn, *args):
    """Run an async function safely, even if an event loop is already active."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is None:
        return asyncio.run(coro_fn(*args))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(coro_fn(*args))).result()


def _ocr_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract text from scanned PDF pages using OCR."""
    text_chunks = []
    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(pdf_bytes)
        
        # Try winocr first (native Windows OCR)
        try:
            import winocr
            for page in pdf:
                pil_img = page.render(scale=2).to_pil()
                res = _run_async(winocr.recognize_pil, pil_img, "en")
                if res and res.text:
                    text_chunks.append(res.text)
            if text_chunks:
                return "\n".join(text_chunks).strip()
        except Exception:
            pass

        # Try pytesseract as secondary fallback
        try:
            import pytesseract
            for page in pdf:
                pil_img = page.render(scale=2).to_pil()
                ocr_text = pytesseract.image_to_string(pil_img)
                if ocr_text:
                    text_chunks.append(ocr_text)
            if text_chunks:
                return "\n".join(text_chunks).strip()
        except Exception:
            pass
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

