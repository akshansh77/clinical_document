"""
document_loader.py — Document ingestion module

Handles PDF, image, and plain text file ingestion.
Uses pdfplumber for PDF text extraction, pytesseract + pdf2image for OCR fallback,
and direct decoding for plain text files.

All operations are timed and logged for performance profiling.
"""

import io
import logging
import time
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

# Minimum character count to consider PDF text extraction successful
# If below this threshold, we assume it's a scanned document and try OCR
MIN_TEXT_LENGTH = 50


def _try_import_pdfplumber():
    """Lazily import pdfplumber to provide a clear error if not installed."""
    try:
        import pdfplumber
        return pdfplumber
    except ImportError:
        raise ImportError(
            "pdfplumber is required for PDF processing. "
            "Install it with: pip install pdfplumber"
        )


def _try_import_ocr():
    """Lazily import OCR dependencies and check for system binaries."""
    try:
        import pytesseract
        from pdf2image import convert_from_bytes
        # Quick check that Tesseract binary is accessible
        pytesseract.get_tesseract_version()
        return pytesseract, convert_from_bytes
    except ImportError as e:
        raise ImportError(
            f"OCR dependencies not available: {e}. "
            "Install with: pip install pytesseract pdf2image"
        )
    except EnvironmentError:
        raise EnvironmentError(
            "Tesseract OCR binary not found. Please install Tesseract:\n"
            "  Windows: https://github.com/UB-Mannheim/tesseract/wiki\n"
            "  macOS:   brew install tesseract\n"
            "  Linux:   sudo apt-get install tesseract-ocr"
        )


def load_text(file_bytes: bytes, filename: str) -> str:
    """Load a plain text file from bytes."""
    t0 = time.perf_counter()
    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = file_bytes.decode("latin-1")
            logger.warning(f"File '{filename}' decoded with latin-1 fallback.")
        except Exception:
            return f"[ERROR] Unable to decode text file '{filename}'. Unsupported encoding."

    text = text.strip()
    elapsed = time.perf_counter() - t0
    if not text:
        logger.info(f"[TIMING] load_text('{filename}'): {elapsed:.3f}s — file is empty")
        return f"[ERROR] The text file '{filename}' is empty."

    logger.info(
        f"[TIMING] load_text('{filename}'): {elapsed:.3f}s — "
        f"{len(text)} chars, {len(file_bytes)} bytes"
    )
    return text


def load_pdf(file_bytes: bytes, filename: str) -> str:
    """
    Load a PDF file. Attempts direct text extraction first;
    if the result is too short (likely a scanned document), falls back to OCR.
    """
    t0_total = time.perf_counter()
    pdfplumber = _try_import_pdfplumber()

    # Step 1: Try direct text extraction
    t0_extract = time.perf_counter()
    text_pages = []
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            num_pages = len(pdf.pages)
            logger.info(f"[TIMING] PDF '{filename}': opened with {num_pages} page(s)")
            for i, page in enumerate(pdf.pages):
                t_page = time.perf_counter()
                page_text = page.extract_text()
                page_elapsed = time.perf_counter() - t_page
                if page_text:
                    text_pages.append(page_text)
                    logger.info(
                        f"[TIMING] PDF '{filename}' page {i+1}/{num_pages}: "
                        f"{page_elapsed:.3f}s — {len(page_text)} chars extracted"
                    )
                else:
                    logger.info(
                        f"[TIMING] PDF '{filename}' page {i+1}/{num_pages}: "
                        f"{page_elapsed:.3f}s — no text extracted"
                    )
    except Exception as e:
        logger.error(f"pdfplumber failed on '{filename}': {e}")
        text_pages = []

    extract_elapsed = time.perf_counter() - t0_extract
    combined_text = "\n\n".join(text_pages).strip()
    logger.info(
        f"[TIMING] PDF '{filename}' direct extraction total: {extract_elapsed:.3f}s — "
        f"{len(combined_text)} chars from {len(text_pages)} page(s)"
    )

    if len(combined_text) >= MIN_TEXT_LENGTH:
        total_elapsed = time.perf_counter() - t0_total
        logger.info(
            f"[TIMING] load_pdf('{filename}') TOTAL: {total_elapsed:.3f}s — "
            f"direct extraction succeeded"
        )
        return combined_text

    # Step 2: Fall back to OCR
    logger.info(
        f"PDF '{filename}' yielded only {len(combined_text)} chars via direct extraction. "
        "Attempting OCR fallback..."
    )
    result = _ocr_pdf(file_bytes, filename)
    total_elapsed = time.perf_counter() - t0_total
    logger.info(f"[TIMING] load_pdf('{filename}') TOTAL (with OCR): {total_elapsed:.3f}s")
    return result


def _ocr_pdf(file_bytes: bytes, filename: str) -> str:
    """Run OCR on a PDF by converting pages to images."""
    t0 = time.perf_counter()
    try:
        pytesseract, convert_from_bytes = _try_import_ocr()
    except (ImportError, EnvironmentError) as e:
        return (
            f"[ERROR] PDF '{filename}' appears to be a scanned document, "
            f"but OCR is unavailable: {e}"
        )

    t0_convert = time.perf_counter()
    try:
        images = convert_from_bytes(file_bytes, dpi=300)
    except Exception as e:
        return (
            f"[ERROR] Failed to convert PDF '{filename}' to images for OCR. "
            f"Poppler may not be installed: {e}"
        )
    convert_elapsed = time.perf_counter() - t0_convert
    logger.info(
        f"[TIMING] _ocr_pdf('{filename}') pdf-to-images: {convert_elapsed:.3f}s — "
        f"{len(images)} page image(s)"
    )

    ocr_pages = []
    for i, img in enumerate(images):
        t_page = time.perf_counter()
        try:
            page_text = pytesseract.image_to_string(img, lang="eng")
            page_elapsed = time.perf_counter() - t_page
            if page_text.strip():
                ocr_pages.append(page_text.strip())
                logger.info(
                    f"[TIMING] _ocr_pdf('{filename}') OCR page {i+1}: "
                    f"{page_elapsed:.3f}s — {len(page_text.strip())} chars"
                )
            else:
                logger.info(
                    f"[TIMING] _ocr_pdf('{filename}') OCR page {i+1}: "
                    f"{page_elapsed:.3f}s — no text"
                )
        except Exception as e:
            page_elapsed = time.perf_counter() - t_page
            logger.error(
                f"OCR failed on page {i + 1} of '{filename}' after {page_elapsed:.3f}s: {e}"
            )
            ocr_pages.append(f"[OCR failed on page {i + 1}]")

    combined = "\n\n".join(ocr_pages).strip()
    total_elapsed = time.perf_counter() - t0
    logger.info(
        f"[TIMING] _ocr_pdf('{filename}') TOTAL: {total_elapsed:.3f}s — "
        f"{len(combined)} chars from {len(ocr_pages)} page(s)"
    )

    if not combined or len(combined) < MIN_TEXT_LENGTH:
        return (
            f"[ERROR] OCR could not extract meaningful text from '{filename}'. "
            "The document may be unreadable or contain only images."
        )
    return combined


def load_image(file_bytes: bytes, filename: str) -> str:
    """Run OCR on an image file (PNG, JPG, JPEG)."""
    t0 = time.perf_counter()
    try:
        pytesseract, _ = _try_import_ocr()
    except (ImportError, EnvironmentError) as e:
        return f"[ERROR] OCR is unavailable for image '{filename}': {e}"

    t0_open = time.perf_counter()
    try:
        img = Image.open(io.BytesIO(file_bytes))
    except Exception as e:
        return f"[ERROR] Cannot open image '{filename}': {e}"
    open_elapsed = time.perf_counter() - t0_open
    logger.info(
        f"[TIMING] load_image('{filename}') image open: {open_elapsed:.3f}s — "
        f"size={img.size}, mode={img.mode}"
    )

    t0_ocr = time.perf_counter()
    try:
        text = pytesseract.image_to_string(img, lang="eng").strip()
    except Exception as e:
        return f"[ERROR] OCR failed on image '{filename}': {e}"
    ocr_elapsed = time.perf_counter() - t0_ocr
    total_elapsed = time.perf_counter() - t0

    logger.info(
        f"[TIMING] load_image('{filename}') OCR: {ocr_elapsed:.3f}s — "
        f"{len(text)} chars extracted"
    )
    logger.info(f"[TIMING] load_image('{filename}') TOTAL: {total_elapsed:.3f}s")

    if not text or len(text) < 10:
        return (
            f"[ERROR] OCR could not extract meaningful text from image '{filename}'. "
            "The image may be blank or unreadable."
        )
    return text


def load_document(file_bytes: bytes, filename: str) -> str:
    """
    Main entry point: load a document and return its text content.

    Supports .txt, .pdf, .png, .jpg, .jpeg files.
    Returns the extracted text, or an error message string starting with '[ERROR]'
    if extraction fails (never raises an exception).

    Parameters
    ----------
    file_bytes : bytes
        The raw file content.
    filename : str
        The original filename, used to determine the file type and for error messages.

    Returns
    -------
    str
        The extracted text, or an error message.
    """
    t0 = time.perf_counter()
    logger.info(f"[TIMING] load_document('{filename}'): START — {len(file_bytes)} bytes")

    if not file_bytes:
        return f"[ERROR] The uploaded file '{filename}' is empty (0 bytes)."

    suffix = Path(filename).suffix.lower()

    try:
        if suffix == ".txt":
            result = load_text(file_bytes, filename)
        elif suffix == ".pdf":
            result = load_pdf(file_bytes, filename)
        elif suffix in (".png", ".jpg", ".jpeg"):
            result = load_image(file_bytes, filename)
        else:
            result = (
                f"[ERROR] Unsupported file type '{suffix}' for '{filename}'. "
                "Please upload a .txt, .pdf, .png, or .jpg file."
            )
    except Exception as e:
        logger.exception(f"Unexpected error loading '{filename}'")
        result = f"[ERROR] Unexpected error processing '{filename}': {e}"

    elapsed = time.perf_counter() - t0
    is_error = result.startswith("[ERROR]")
    logger.info(
        f"[TIMING] load_document('{filename}'): DONE in {elapsed:.3f}s — "
        f"{'ERROR' if is_error else f'{len(result)} chars extracted'}"
    )
    return result
