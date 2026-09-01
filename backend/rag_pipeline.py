"""RAG pipeline orchestration.

Provides a simple function to run a retrieval-augmented generation flow:

- generate query embedding
- search FAISS index
- retrieve top-k chunks
- evaluate retrieval relevance before Gemini is called
- call LLM with the retrieved context
- return answer, citations (texts), and chunk ids

Dependencies: backend.embeddings, backend.vectorstore, backend.llm
"""
import os
import re
from typing import Dict, List

import numpy as np

from .embeddings import generate_embeddings
from .vectorstore import load_chunks, search
from .llm import generate_answer
from .parser import clean_chunks

# FAISS configuration: we normalize vectors and use IndexFlatIP, which is cosine similarity
# after L2 normalization. Higher scores mean more similar; lower scores mean less similar.
RELEVANCE_THRESHOLD = 0.45


def _deduplicate_texts(texts: List[str]) -> List[str]:
    """Remove near-duplicate chunks while preserving meaning and ordering."""
    unique: List[str] = []
    seen = set()
    for text in texts:
        if not text or not isinstance(text, str):
            continue
        norm = re.sub(r"\s+", " ", text).strip().lower()
        if not norm:
            continue
        if norm in seen:
            continue
        seen.add(norm)
        unique.append(text)
    return unique


def _is_relevant_result(scores: np.ndarray) -> bool:
    """Use the actual FAISS similarity metric to gate retrieval relevance.

    In this project, FAISS is configured with L2-normalized vectors and IndexFlatIP,
    which behaves like cosine similarity. Higher score = more relevant. If the best
    result falls below the threshold, we reject the query before calling Gemini.
    """
    if scores is None or len(scores) == 0:
        return False
    best_score = float(np.max(scores))
    if os.getenv("DEBUG_RAG_RELEVANCE", "").lower() in {"1", "true", "yes"}:
        print(f"[RAG relevance] best score = {best_score:.4f}, threshold = {RELEVANCE_THRESHOLD:.4f}")
    return best_score >= RELEVANCE_THRESHOLD


def run_rag(question: str, chunks: List[str], top_k: int = 5) -> Dict[str, object]:
    """Run the retrieval flow, validate relevance, and build a clean context."""
    if not isinstance(chunks, list):
        chunks = list(chunks)

    if not chunks:
        try:
            chunks = load_chunks()
        except Exception:
            chunks = []

    try:
        chunks = clean_chunks(chunks)
    except Exception:
        pass

    if not chunks:
        return {"answer": "I couldn't find relevant information about this question in the uploaded documents.", "citations": [], "chunk_ids": []}

    q_emb = np.asarray(generate_embeddings([question]), dtype=np.float32)
    if q_emb.ndim != 2 or q_emb.shape[0] != 1:
        raise RuntimeError("Unexpected embedding shape for query")

    try:
        indices, scores = search(q_emb[0], k=max(3, min(top_k, len(chunks))))
    except RuntimeError as e:
        raise RuntimeError("Vector index not found or not loaded. Process documents first.") from e

    if not _is_relevant_result(np.asarray(scores, dtype=np.float32)):
        return {"answer": "I couldn't find relevant information about this question in the uploaded documents.", "citations": [], "chunk_ids": []}

    raw_matches: List[str] = []
    chunk_ids: List[int] = []
    for idx in indices:
        try:
            i = int(idx)
        except Exception:
            continue
        if i < 0 or i >= len(chunks):
            continue
        text = chunks[i]
        if not text or not text.strip():
            continue
        chunk_ids.append(i)
        raw_matches.append(text)

    deduped_matches = _deduplicate_texts(raw_matches)
    if not deduped_matches:
        return {"answer": "I couldn't find relevant information about this question in the uploaded documents.", "citations": [], "chunk_ids": []}

    context_parts = []
    for rank, text in enumerate(deduped_matches[: min(len(deduped_matches), top_k)], start=1):
        clean_text = re.sub(r"\[Document: .*?\]\s*", "", text).strip()
        doc_name = ""
        doc_match = re.search(r"\[Document:\s*(.*?)\]", text)
        if doc_match:
            doc_name = doc_match.group(1).strip()
        section = ""
        section_match = re.search(r"\[Section:\s*(.*?)\]", text)
        if section_match:
            section = section_match.group(1).strip()
        prefix = ""
        if doc_name:
            prefix += f"[Document: {doc_name}]\n"
        if section:
            prefix += f"[Section: {section}]\n"
        prefix += f"[Relevant passage {rank}]\n"
        context_parts.append(f"{prefix}{clean_text}")

    context = "\n\n".join(context_parts)
    answer = generate_answer(context, question)

    return {"answer": answer, "citations": deduped_matches[: min(len(deduped_matches), top_k)], "chunk_ids": chunk_ids[: min(len(chunk_ids), top_k)]}


__all__ = ["run_rag", "RELEVANCE_THRESHOLD"]
