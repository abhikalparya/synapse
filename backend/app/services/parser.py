"""Extract plain text from uploaded documents (.txt, .md, .pdf, .docx)."""

import logging
import re
from io import BytesIO

logger = logging.getLogger(__name__)


def _clean_extracted_text(text: str) -> str:
    """Normalize line breaks and collapse excessive whitespace."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_txt(file: bytes) -> str:
    """Decode UTF-8 text; replace undecodable bytes."""
    try:
        raw = file.decode("utf-8")
    except UnicodeDecodeError:
        raw = file.decode("utf-8", errors="replace")
        logger.warning("parse_txt: used UTF-8 replacement for invalid sequences")
    return _clean_extracted_text(raw)


def parse_md(file: bytes) -> str:
    """Markdown is read as plain text (same as .txt)."""
    return parse_txt(file)


def parse_pdf(file: bytes) -> str:
    """Extract text page-by-page with PyMuPDF."""
    try:
        import fitz
    except ImportError as exc:
        logger.error("parse_pdf: PyMuPDF (fitz) not installed")
        raise RuntimeError("PDF parsing requires pymupdf") from exc

    try:
        doc = fitz.open(stream=file, filetype="pdf")
    except Exception as exc:
        logger.warning("parse_pdf: failed to open PDF: %s", exc)
        raise ValueError(f"Invalid or unreadable PDF: {exc}") from exc

    try:
        parts: list[str] = []
        for page in doc:
            parts.append(page.get_text() or "")
    finally:
        doc.close()

    joined = "\n".join(parts)
    result = _clean_extracted_text(joined)
    logger.info("parse_pdf: pages=%s chars=%s", len(parts), len(result))
    return result


def parse_docx(file: bytes) -> str:
    """Extract paragraph text from a Word document."""
    try:
        from docx import Document
    except ImportError as exc:
        logger.error("parse_docx: python-docx not installed")
        raise RuntimeError("DOCX parsing requires python-docx") from exc

    try:
        doc = Document(BytesIO(file))
    except Exception as exc:
        logger.warning("parse_docx: failed to open document: %s", exc)
        raise ValueError(f"Invalid or unreadable DOCX: {exc}") from exc

    lines = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    result = _clean_extracted_text("\n".join(lines))
    logger.info("parse_docx: paragraphs=%s chars=%s", len(lines), len(result))
    return result
