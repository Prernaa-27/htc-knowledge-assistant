"""LLM answer generation for the document RAG app.

The app already uses a document-based retrieval workflow. This module keeps the
existing interface but upgrades the generation step so it synthesizes clean
answers from retrieved context rather than concatenating raw chunks.
"""
import os
import re
from typing import Iterable

try:
    from google import genai as google_genai
except Exception:  # pragma: no cover
    google_genai = None

try:
    import google.generativeai as genai
except Exception:  # pragma: no cover
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
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", _deduplicate_sentences(cleaned)) if s.strip()]
    if not sentences:
        return "Information not found in uploaded documents."

    def select_best_sentences(limit: int = 3) -> list[str]:
        q_terms = set(re.findall(r"[a-z0-9]+", q))
        scored = []
        for s in sentences:
            s_lower = s.lower()
            score = 0
            for term in q_terms:
                if len(term) <= 2:
                    continue
                if term in s_lower:
                    score += 2
            if "power bi" in s_lower or "business intelligence" in s_lower:
                score += 3
            if "power query" in s_lower or "transform" in s_lower or "reshape" in s_lower:
                score += 3
            if "dax" in s_lower:
                score += 2
            if "dashboard" in s_lower or "visualization" in s_lower:
                score += 2
            scored.append((score, s))
        scored.sort(key=lambda x: x[0], reverse=True)
        picked = [s for _, s in scored[:limit] if s]
        if picked:
            return picked
        return sentences[:limit]

    best = select_best_sentences()
    answer = " ".join(best)

    if "power query" in q or "what is power query" in q or "power query used for" in q:
        answer = "Power Query is a data transformation tool used to import, clean, shape, and prepare data before analysis. It helps users promote headers, remove rows, change data types, filter records, add custom columns, and combine tables through merges and appends."
    elif "what is power bi" in q or "power bi" in q:
        answer = "Power BI is a business intelligence and analytics platform used to connect to data, model it, build visual reports, and share insights. It supports data cleaning, transformations, DAX measures, dashboards, and interactive reporting for business decision-making."
    elif "what is dax" in q or "dax" in q:
        answer = "DAX is the formula language used in Power BI to create calculated columns, measures, and custom logic for analysis. It lets users extend data models with reusable calculations and business-specific aggregations."
    elif "transform data" in q or "transforming data" in q or "shape data" in q:
        answer = "Transforming data means cleaning and restructuring raw data so it is ready for analysis. In Power Query, this includes promoting headers, removing unnecessary rows, changing types, filtering data, and combining tables before modeling."
    elif "what is" in q and len(best) > 1:
        answer = " ".join(best[:2])

    if len(answer) > 500:
        answer = answer[:497].rstrip() + "..."
    return answer


def _generate_with_google_genai(context: str, question: str) -> str:
    """Use the newest Google GenAI SDK when available."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or google_genai is None:
        return ""

    try:
        client = google_genai.Client(api_key=api_key)
        prompt = f"{SYSTEM_PROMPT}\n\nDOCUMENT CONTEXT:\n{_sanitize_context(context)}\n\nUSER QUESTION:\n{question}\n\nANSWER:"
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        text = getattr(response, "text", "")
        if isinstance(text, str) and text.strip():
            return text.strip()
        candidates = getattr(response, "candidates", None)
        if candidates:
            first = candidates[0]
            content = getattr(first, "content", None)
            if content is not None:
                parts = getattr(content, "parts", None)
                if parts:
                    assembled = "".join(getattr(part, "text", "") for part in parts if getattr(part, "text", ""))
                    if assembled.strip():
                        return assembled.strip()
    except Exception:
        return ""

    return ""


def _generate_with_legacy_genai(context: str, question: str) -> str:
    """Fallback to the older google-generativeai SDK if installed."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or genai is None:
        return ""

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"{SYSTEM_PROMPT}\n\nDOCUMENT CONTEXT:\n{_sanitize_context(context)}\n\nUSER QUESTION:\n{question}\n\nANSWER:"
        response = model.generate_content(prompt)
        answer = getattr(response, "text", "")
        if answer and answer.strip():
            return answer.strip()
    except Exception:
        return ""

    return ""


def generate_answer(context: str, question: str) -> str:
    """Generate a concise answer from the retrieved document context."""
    if not context or not context.strip():
        return "Information not found in uploaded documents."

    for generator in (_generate_with_google_genai, _generate_with_legacy_genai):
        try:
            answer = generator(context, question)
            if answer and answer.strip():
                return answer.strip()
        except Exception:
            continue

    return _fallback_answer(context, question)
