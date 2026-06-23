"""Embeddings utilities using sentence-transformers.

Provides:
- load_model(): load and return the SentenceTransformer model
- generate_embeddings(chunks): encode a list/iterable of text chunks and
  return a NumPy array of embeddings

Uses model: all-MiniLM-L6-v2
"""
from typing import Iterable, List

import numpy as np
from sentence_transformers import SentenceTransformer


_MODEL: SentenceTransformer | None = None


def load_model() -> SentenceTransformer:
    """Load and return the SentenceTransformer model.

    This function caches the model after the first load.
    """
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


def generate_embeddings(chunks: Iterable[str]) -> np.ndarray:
    """Generate embeddings for the provided chunks.

    Args:
        chunks: An iterable of strings (each string is a text chunk).

    Returns:
        A NumPy array of shape (n_chunks, embedding_dim).

    Notes:
        - This function will call :func:`load_model` internally.
        - If ``chunks`` is empty, an array with shape (0, embedding_dim)
          is returned.
    """
    # Normalize input to list
    if isinstance(chunks, str):
        chunks_list: List[str] = [chunks]
    else:
        chunks_list = list(chunks)

    model = load_model()

    if len(chunks_list) == 0:
        dim = model.get_sentence_embedding_dimension()
        return np.zeros((0, dim), dtype=np.float32)

    embeddings = model.encode(chunks_list, convert_to_numpy=True, show_progress_bar=False)

    # Ensure NumPy ndarray
    return np.asarray(embeddings)


__all__ = ["load_model", "generate_embeddings"]
