"""Simple FAISS-backed vector store utilities.

Functions:
- create_index(vectors): create and store an index from a NumPy array of vectors
- save_index(): persist the current index to disk under vector_db/faiss_index
- load_index(): load the index from disk and return it
- search(query_embedding, k=3): query the index and return (indices, scores)

Notes:
- Uses inner-product on L2-normalized vectors (cosine similarity).
"""
from pathlib import Path
from typing import Optional, Tuple

import faiss
import numpy as np
import json


INDEX_DIR = Path(__file__).resolve().parent.parent / "vector_db" / "faiss_index"
INDEX_FILE = INDEX_DIR / "index.faiss"
CHUNKS_FILE = INDEX_DIR.parent / "chunks.json"


# Module-level index cache
_INDEX: Optional[faiss.Index] = None


def _ensure_index_dir() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)


def create_index(vectors: np.ndarray) -> faiss.Index:
    """Create a FAISS index from the provided vectors and cache it.

    Args:
        vectors: NumPy array of shape (n, dim), dtype float32 (or castable).

    Returns:
        The created FAISS index.
    """
    global _INDEX

    if not isinstance(vectors, np.ndarray):
        vectors = np.asarray(vectors)

    if vectors.ndim != 2:
        raise ValueError("vectors must be a 2D array of shape (n, dim)")

    vectors = vectors.astype("float32")

    # Normalize to unit length to use inner-product as cosine similarity
    faiss.normalize_L2(vectors)

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    _INDEX = index
    return index


def save_index() -> str:
    """Save the cached index to disk under `vector_db/faiss_index/index.faiss`.

    Returns the filepath written.
    """
    if _INDEX is None:
        raise RuntimeError("No index to save. Create or load an index first.")

    _ensure_index_dir()
    faiss.write_index(_INDEX, str(INDEX_FILE))
    return str(INDEX_FILE)


def save_chunks(chunks: list) -> str:
    """Persist the list of chunks to disk as JSON and return filepath."""
    _ensure_index_dir()
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    return str(CHUNKS_FILE)


def load_chunks() -> list:
    """Load persisted chunks from disk. Raises FileNotFoundError if missing."""
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"Chunks file not found at {CHUNKS_FILE}")
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_index() -> faiss.Index:
    """Load the FAISS index from disk and cache it.

    Raises FileNotFoundError if the index file does not exist.
    """
    global _INDEX
    if not INDEX_FILE.exists():
        raise FileNotFoundError(f"FAISS index not found at {INDEX_FILE}")

    index = faiss.read_index(str(INDEX_FILE))
    _INDEX = index
    return index


def clear_index_cache() -> None:
    """Clear the in-memory FAISS index cache."""
    global _INDEX
    _INDEX = None


def search(query_embedding: np.ndarray, k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Search the cached index for the nearest neighbors to query_embedding.

    Args:
        query_embedding: 1-D array of length dim, or 2-D array (1, dim).
        k: number of nearest neighbors to return.

    Returns:
        (indices, scores) where both are 1-D NumPy arrays of length <= k.
    """
    if _INDEX is None:
        # Try to load from disk automatically
        try:
            load_index()
        except FileNotFoundError:
            raise RuntimeError("No index loaded. Create or load an index before searching.")

    qe = np.asarray(query_embedding, dtype="float32")
    if qe.ndim == 1:
        qe = qe.reshape(1, -1)
    if qe.ndim != 2:
        raise ValueError("query_embedding must be 1D or 2D array")

    # Normalize query to match index normalization
    faiss.normalize_L2(qe)

    distances, indices = _INDEX.search(qe, k)

    # return first row (since we searched a single query)
    return indices[0], distances[0]


__all__ = [
    "create_index",
    "save_index",
    "load_index",
    "search",
    "save_chunks",
    "load_chunks",
]
