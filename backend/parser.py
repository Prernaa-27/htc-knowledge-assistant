import os
import re
from typing import List

try:
    import fitz
except ModuleNotFoundError:
    fitz = None
from docx import Document


_EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(r"\b\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b")
_LOREM_RE = re.compile(r"lorem ipsum", re.IGNORECASE)
_NOISE_LINE_PATTERNS = (
    r"^new message$",
    r"^hey this is important!$",
    r"^important!$",
    r"^copyright.*maven analytics.*$",
    r"^\*?copyright.*\*?$",
    r"^slide\s*\d+.*$",
    r"^page\s*\d+.*$",
    r"^figure\s*\d+.*$",
)


def _normalize_whitespace(text: str) -> str:
    """Collapse broken PDF layout and normalize whitespace without losing content."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _is_noise_line(line: str) -> bool:
    """Identify obvious formatting boilerplate and repeated slide artifacts."""
    if not line:
        return True
    text = line.strip()
    if not text:
        return True
    lowered = text.lower()
    if len(text) < 3:
        return True
    if _EMAIL_RE.search(text) or _PHONE_RE.search(text):
        return True
    if _LOREM_RE.search(text):
        return True
    if lowered.startswith("your company") or lowered.startswith("your name"):
        return True
    if lowered.startswith("copyright") or "copyright" in lowered:
        return True
    if "new message" in lowered or "hey this is important" in lowered:
        return True
    if re.fullmatch(r"[\W_]+", text):
        return True
    for pattern in _NOISE_LINE_PATTERNS:
        if re.fullmatch(pattern, lowered):
            return True
    return False


def _clean_pdf_like_text(text: str) -> str:
    """Remove obvious PDF noise while preserving meaningful document content."""
    text = _normalize_whitespace(text)
    lines = []
    seen = set()

    for raw_line in text.split("\n"):
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        clean_line = line.replace("*", "").strip()
        if _is_noise_line(clean_line):
            continue

        key = re.sub(r"\s+", " ", clean_line).lower()
        if key and key not in seen:
            seen.add(key)
            lines.append(clean_line)

    return "\n\n".join(lines).strip()


def read_pdf(file_path: str) -> str:
    """Read text from a PDF file using PyMuPDF."""
    if fitz is None:
        raise ModuleNotFoundError(
            "PyMuPDF is not installed (module 'fitz' missing).\n"
            "Install it with: python -m pip install PyMuPDF"
        )
    document = fitz.open(file_path)
    pages = [page.get_text() for page in document]
    return _clean_pdf_like_text("\n".join(pages))


def read_docx(file_path: str) -> str:
    """Read text from a DOCX file using python-docx."""
    document = Document(file_path)
    paragraphs = [paragraph.text for paragraph in document.paragraphs]
    return _clean_pdf_like_text("\n".join(paragraphs))


def _merge_paragraphs(text: str) -> List[str]:
    """Break text into paragraph-like blocks while keeping content together."""
    blocks = [block.strip() for block in re.split(r"\n\s*\n+", text) if block and block.strip()]
    merged: List[str] = []
    current = ""

    for block in blocks:
        if not block:
            continue
        if len(current) + len(block) + 1 <= 260:
            current = f"{current} {block}".strip() if current else block
        else:
            if current:
                merged.append(current)
            current = block
    if current:
        merged.append(current)
    return merged


def split_text(text: str, chunk_size: int = 700, overlap: int = 80) -> List[str]:
    """Split text into paragraph-aware overlapping chunks without breaking sentence flow."""
    if not text:
        return []

    text = _clean_pdf_like_text(text)
    blocks = _merge_paragraphs(text)
    if not blocks:
        return []

    chunks: List[str] = []
    buffer = ""
    for block in blocks:
        block = re.sub(r"\s+", " ", block).strip()
        if len(block) <= chunk_size:
            if not buffer:
                buffer = block
            elif len(buffer) + len(block) + 1 <= chunk_size:
                buffer = f"{buffer} {block}"
            else:
                chunks.append(buffer)
                buffer = block
        else:
            if buffer:
                chunks.append(buffer)
                buffer = ""
            sentences = re.split(r"(?<=[.!?])\s+", block)
            sentence_buffer = ""
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                if not sentence_buffer:
                    sentence_buffer = sentence
                elif len(sentence_buffer) + len(sentence) + 1 <= chunk_size:
                    sentence_buffer = f"{sentence_buffer} {sentence}"
                else:
                    chunks.append(sentence_buffer)
                    sentence_buffer = sentence
            if sentence_buffer:
                if buffer and len(buffer) + len(sentence_buffer) + 1 <= chunk_size:
                    buffer = f"{buffer} {sentence_buffer}"
                else:
                    if buffer:
                        chunks.append(buffer)
                    buffer = sentence_buffer

    if buffer:
        chunks.append(buffer)

    if not chunks:
        return []

    overlapped: List[str] = []
    for i, chunk in enumerate(chunks):
        prefix = ""
        if i > 0:
            previous_tail = chunks[i - 1][-overlap:].strip()
            prefix = previous_tail.split(" ", 1)[-1] if " " in previous_tail else ""
        overlapped.append(f"{prefix} {chunk}".strip() if prefix else chunk)

    result = []
    for idx, chunk in enumerate(overlapped):
        normalized = re.sub(r"\s+", " ", chunk).strip()
        if normalized:
            result.append(normalized)
    return result


def parse_document(file_path: str, chunk_size: int = 700, overlap: int = 80) -> List[str]:
    """Read a document and return a list of text chunks with minimal, useful metadata."""
    if hasattr(file_path, "read"):
        file_name = getattr(file_path, "name", "uploaded.pdf")
        _, extension = os.path.splitext(file_name)
        extension = extension.lower()

        if extension == ".pdf":
            if fitz is None:
                raise ModuleNotFoundError(
                    "PyMuPDF is not installed (module 'fitz' missing).\n"
                    "Install it with: python -m pip install PyMuPDF"
                )
            file_bytes = file_path.read()
            document = fitz.open(stream=file_bytes, filetype="pdf")
            text = "\n".join(page.get_text() for page in document).strip()
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

    chunks = split_text(text, chunk_size=chunk_size, overlap=overlap)
    source_name = os.path.basename(str(getattr(file_path, "name", file_path))) if hasattr(file_path, "name") else os.path.basename(str(file_path))
    if source_name and chunks:
        chunks = [f"[Document: {source_name}] {chunk}" for chunk in chunks]
    return chunks


def clean_chunks(chunks: List[str], min_length: int = 60) -> List[str]:
    """Filter out short or boilerplate chunks while keeping meaningful document content."""
    cleaned: List[str] = []
    seen = set()

    for chunk in chunks:
        if not chunk or not isinstance(chunk, str):
            continue
        s = chunk.strip()
        if not s:
            continue
        if len(s) < min_length:
            continue
        if _EMAIL_RE.search(s) or _PHONE_RE.search(s):
            continue
        if _LOREM_RE.search(s):
            continue
        if s.lower().startswith("your company") or s.lower().startswith("your name"):
            continue
        normalized = re.sub(r"\s+", " ", s).lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(s)
    return cleaned
