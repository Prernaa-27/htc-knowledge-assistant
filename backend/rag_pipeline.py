"""RAG pipeline orchestration.

Provides a simple function to run a retrieval-augmented generation flow:

- generate query embedding
- search FAISS index
- retrieve top-k chunks
- call LLM with the retrieved context
- return answer, citations (texts), and chunk ids

Dependencies: backend.embeddings, backend.vectorstore, backend.llm
"""
from typing import Dict, List, Tuple

import numpy as np

from .embeddings import generate_embeddings
from .vectorstore import (
    search,
    create_index,
    save_index,
    save_chunks,
    load_chunks,
)
from .llm import generate_answer
from .parser import clean_chunks


def run_rag(question: str, chunks: List[str], top_k: int = 3) -> Dict[str, object]:
    """Run a simple RAG workflow.

    Args:
        question: User question string.
        chunks: List of chunk texts indexed in the FAISS index (index positions correspond to these list indices).
        top_k: Number of top chunks to retrieve and pass to the LLM.

    Returns:
        A dict with keys:
          - "answer": str -- LLM answer string
          - "citations": List[str] -- the texts of the retrieved chunks (in rank order)
          - "chunk_ids": List[int] -- integer indices of the retrieved chunks
    """
    # If caller didn't provide chunks, attempt to load persisted chunks
    if not isinstance(chunks, list):
        chunks = list(chunks)

    if not chunks:
        try:
            chunks = load_chunks()
        except Exception:
            # No chunks available; will fail later when we search
            chunks = []

    # Filter out boilerplate or very short chunks before retrieval
    try:
        chunks = clean_chunks(chunks)
    except Exception:
        # If cleaning fails for any reason, proceed with original chunks
        pass

    # 1) Generate query embedding
    q_emb = generate_embeddings([question])  # shape (1, dim)
    if q_emb.ndim != 2 or q_emb.shape[0] != 1:
        raise RuntimeError("Unexpected embedding shape for query")

    # 2) Search FAISS
    try:
        indices, scores = search(q_emb[0], k=top_k)
    except RuntimeError as e:
        # Provide a clearer error upstream (e.g., no index available)
        raise RuntimeError("Vector index not found or not loaded. Process documents first.") from e

    # Convert indices to ints and filter out invalid values
    chunk_ids: List[int] = []
    citations: List[str] = []
    for idx in indices:
        try:
            i = int(idx)
        except Exception:
            continue
        if i < 0 or i >= len(chunks):
            # skip out-of-range results
            continue
        chunk_ids.append(i)
        citations.append(chunks[i])

    # 3) Build context from retrieved chunks
    # Join with separators and include simple citation markers
    context_parts = []
    for rank, (cid, text) in enumerate(zip(chunk_ids, citations), start=1):
        context_parts.append(f"[chunk_{cid}] Rank {rank}: {text}")
    context = "\n\n".join(context_parts)

    # 4) Call LLM with the retrieved context and original question
    answer = generate_answer(context, question)

    return {"answer": answer, "citations": citations, "chunk_ids": chunk_ids}


__all__ = ["run_rag"]
