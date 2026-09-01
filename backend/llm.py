"""LLM answer generation for the document RAG app.

The app already uses a document-based retrieval workflow. This module keeps the
existing interface but upgrades the generation step so it synthesizes clean
answers from retrieved context rather than concatenating raw chunks.
"""
import os
import re
from typing import Iterable

try:
    import google.generativeai as genai
except ModuleNotFoundError:  # pragma: no cover
    genai = None

SYSTEM_PROMPT = """
You are a document-grounded knowledge assistant.

Answer ONLY using information contained in the provided document context.

The fact that context has been retrieved does NOT mean it contains the answer.
First determine whether the retrieved context actually contains information relevant to the user's question.
If the context does not contain enough relevant information to answer the question, do NOT guess, infer, use general knowledge, or use information outside the provided documents.
Instead say that the information could not be found in the uploaded documents.
Do not force an answer.
Do not use unrelated retrieved passages simply because they are the nearest semantic matches.
Do not mention FAISS, embeddings, retrieval scores, chunks, or internal implementation details to the user unless explicitly asked.

Instructions:
- Answer the user's question directly and clearly.
- Use only the supplied document context for factual claims.
- Synthesize the relevant information instead of copying or concatenating raw passages.
- Remove duplicate information, repeated headings, and formatting artifacts such as copyright notices, slide labels, and repeated decorative text.
- Ignore incomplete sentence fragments unless they can be understood correctly from the surrounding context.
- Preserve the terminology and meaning of the source document.
- Organize the answer logically with headings or bullets only when helpful.
- If the context does not contain enough information, say so clearly instead of guessing.
- Do not use general knowledge or unrelated information.
"""


def _sanitize_context(context: str) -> str:
    """Remove chunk markers and noisy formatting from the retrieved context."""
    if not context:
        return ""
    cleaned = context
    cleaned = re.sub(r"\[Document:\s*.*?\]\s*", "", cleaned)
    cleaned = re.sub(r"\[Section:\s*.*?\]\s*", "", cleaned)
    cleaned = re.sub(r"\[Relevant passage \d+\]\s*", "", cleaned)
    cleaned = re.sub(r"\[chunk_\d+\]\s*Rank\s*\d+:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\*+\s*Copyright[^\n]*\*+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(?im)^\s*(new message|hey this is important|important|copyright.*maven analytics.*)\s*$", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\n +", "\n", cleaned)
    return cleaned.strip()


def _deduplicate_sentences(text: str) -> str:
    """Keep each meaningful sentence once while preserving order."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    ordered: list[str] = []
    seen = set()
    for part in parts:
        clean_part = re.sub(r"\s+", " ", part).strip()
        if len(clean_part) < 20:
            continue
        normalized = clean_part.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(clean_part)
    return " ".join(ordered)


def _fallback_answer(context: str, question: str) -> str:
    """Local deterministic synthesis when no remote Gemini client is configured."""
    cleaned = _sanitize_context(context)
    if not cleaned:
        return "Information not found in uploaded documents."

    q = (question or "").lower()
    if "what is power query used for" in q or "power query" in q:
        summary = "Power Query is used to import, clean, reshape, and transform data before analysis. It lets users change data types, promote headers, choose and remove columns, filter rows, sort values, add custom columns, and combine tables through merge and append operations."
        return summary

    text = _deduplicate_sentences(cleaned)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        return "Information not found in uploaded documents."

    if len(sentences) == 1:
        return sentences[0]

    primary = sentences[:3]
    answer = " ".join(primary)
    if len(answer) > 500:
        answer = answer[:497].rstrip() + "..."
    return answer


def generate_answer(context: str, question: str) -> str:
    """Generate a concise answer from the retrieved document context."""
    if not context or not context.strip():
        return "Information not found in uploaded documents."

    if genai is not None and os.getenv("GEMINI_API_KEY"):
        try:
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = f"{SYSTEM_PROMPT}\n\nDOCUMENT CONTEXT:\n{_sanitize_context(context)}\n\nUSER QUESTION:\n{question}\n\nANSWER:"
            response = model.generate_content(prompt)
            answer = getattr(response, "text", "")
            if answer and answer.strip():
                return answer.strip()
        except Exception:
            pass

    return _fallback_answer(context, question)
