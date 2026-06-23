import os
from typing import List

import fitz
from docx import Document
import re


# Simple heuristics to detect boilerplate/contact chunks
_EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(r"\b\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b")
_LOREM_RE = re.compile(r"lorem ipsum", re.IGNORECASE)


def read_pdf(file_path: str) -> str:
    """Read text from a PDF file using PyMuPDF."""
    document = fitz.open(file_path)
    pages = [page.get_text() for page in document]
    return "\n".join(pages).strip()


def read_docx(file_path: str) -> str:
    """Read text from a DOCX file using python-docx."""
    document = Document(file_path)
    paragraphs = [paragraph.text for paragraph in document.paragraphs]
    return "\n".join(paragraphs).strip()


def split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks."""
    if not text:
        return []

    normalized_text = " ".join(text.split())
    chunks: List[str] = []
    start = 0

    while start < len(normalized_text):
        end = min(start + chunk_size, len(normalized_text))
        chunks.append(normalized_text[start:end])
        if end == len(normalized_text):
            break
        start += chunk_size - overlap

    return chunks


def parse_document(file_path: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Read a document and return a list of text chunks."""
    if hasattr(file_path, "read"):
        file_name = getattr(file_path, "name", "uploaded.pdf")
        _, extension = os.path.splitext(file_name)
        extension = extension.lower()

        if extension == ".pdf":
            file_bytes = file_path.read()
            document = fitz.open(stream=file_bytes, filetype="pdf")
            text = "\n".join([page.get_text() for page in document]).strip()
        elif extension == ".docx":
            text = read_docx(file_path)
        else:
            raise ValueError(f"Unsupported file type: {extension}")
    else:
        _, extension = os.path.splitext(file_path)
        extension = extension.lower()

        if extension == ".pdf":
            text = read_pdf(file_path)
        elif extension == ".docx":
            text = read_docx(file_path)
        else:
            raise ValueError(f"Unsupported file type: {extension}")

    return split_text(text, chunk_size=chunk_size, overlap=overlap)


def clean_chunks(chunks: List[str], min_length: int = 60) -> List[str]:
    """Filter out short or boilerplate chunks.

    Heuristics:
    - Remove very short chunks (default length < 60 characters)
    - Remove chunks containing emails or phone numbers
    - Remove chunks matching lorem ipsum
    """
    cleaned: List[str] = []
    for c in chunks:
        if not c or not isinstance(c, str):
            continue
        s = c.strip()
        if len(s) < min_length:
            continue
        if _EMAIL_RE.search(s) or _PHONE_RE.search(s):
            continue
        if _LOREM_RE.search(s):
            # treat lorem ipsum as boilerplate
            continue
        # naive company/contact header/footer patterns
        if s.lower().startswith("your company") or s.lower().startswith("your name"):
            continue
        cleaned.append(s)
    return cleaned
