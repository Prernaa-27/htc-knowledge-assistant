"""Embeddings utilities using sentence-transformers.

Provides:
- load_model(): load and return the SentenceTransformer model
- generate_embeddings(chunks): encode a list/iterable of text chunks and
  return a NumPy array of embeddings

Uses model: all-MiniLM-L6-v2
"""
from typing import Iterable, List

import numpy as np
import hashlib

try:
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError:
    SentenceTransformer = None


# Target embedding dimension (matches all-MiniLM-L6-v2)
EMBED_DIM = 384


_MODEL: "SentenceTransformer | None" = None


def load_model():
    """Load and return the SentenceTransformer model if available.

    If `sentence_transformers` is not installed, returns None.
    """
    global _MODEL
    if SentenceTransformer is None:
        return None
    if _MODEL is None:
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


def _fallback_embed_text(text: str) -> np.ndarray:
    """Deterministic fallback embedding using SHA256 hashing.

    Produces a vector of length `EMBED_DIM` with values in [-1, 1]. This is
    only used when `sentence_transformers` is not available so the app can
    still run without hard dependency installation.
    """
    h = hashlib.sha256(text.encode("utf-8")).digest()
    # Expand hash digest into required dimension by repeated hashing
    values = []
    counter = 0
    while len(values) < EMBED_DIM:
        data = hashlib.sha256(h + counter.to_bytes(4, "little")).digest()
        for b in data:
            if len(values) >= EMBED_DIM:
                break
            # map byte 0-255 to float in [-1,1]
            values.append((b / 255.0) * 2.0 - 1.0)
        counter += 1
    vec = np.asarray(values, dtype=np.float32)
    # normalize to unit length to behave similarly to real embeddings
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def generate_embeddings(chunks: Iterable[str]) -> np.ndarray:
    """Generate embeddings for the provided chunks.

    Uses SentenceTransformer when available; otherwise falls back to a
    deterministic SHA256-based embedding so the app can operate without the
    `sentence-transformers` package installed.
    """
    # Normalize input to list
    if isinstance(chunks, str):
        chunks_list: List[str] = [chunks]
    else:
        chunks_list = list(chunks)

    model = load_model()

    if model is not None:
        if len(chunks_list) == 0:
            dim = model.get_sentence_embedding_dimension()
            return np.zeros((0, dim), dtype=np.float32)
        embeddings = model.encode(chunks_list, convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(embeddings, dtype=np.float32)

    # Fallback path
    if len(chunks_list) == 0:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)

    mats = [
        _fallback_embed_text(c if isinstance(c, str) else str(c))
        for c in chunks_list
    ]
    return np.vstack(mats)


__all__ = ["load_model", "generate_embeddings"]
