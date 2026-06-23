"""Lightweight LLM fallback for local testing.

This module intentionally avoids remote API calls. It provides a
deterministic, safe fallback implementation of `generate_answer` used
during development when an external LLM is unavailable or when you want
to avoid depending on the Google Generative API.

Behaviour:
- If `context` is empty, return the exact phrase required by the UI
  to indicate missing information.
- Otherwise, return a concise synthesis using the first retrieved
  context chunk(s). This is intentionally simple and deterministic.
"""

SYSTEM_PROMPT = """
You are an HTC Enterprise Knowledge Assistant.

Rules:

1. Answer ONLY from the supplied context.

2. Never use external knowledge.

3. If the answer cannot be found in the context,
reply exactly:

Information not found in uploaded documents.

4. Keep responses concise and professional.
"""


def generate_answer(context: str, question: str) -> str:
    """Return a concise answer using the provided context.

    This is a best-effort, local-only generator used as a replacement
    for the external Gemini call while we patch the app. It does not
    call any external API.
    """
    if not context or not context.strip():
      return "Information not found in uploaded documents."

    q = question.lower() if question else ""

    # Heuristic extraction for name / recipient / who questions.
    import re

    def extract_recipient(text: str) -> str | None:
      # common patterns: 'Dear <Name>'
      m = re.search(r"Dear\s+([A-Z][\w\-\.]+(?:\s+[A-Z][\w\-\.]+)*)", text)
      if m:
        return m.group(1).strip()

      # pattern: '<Name> CEO' or '<Name>, CEO'
      m = re.search(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s*,?\s+CEO", text)
      if m:
        return m.group(1).strip() + " (CEO)"

      # fallback: look for 'Name <email>' or 'Name - Title'
      m = re.search(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+[<\-]\s*\w", text)
      if m:
        return m.group(1).strip()

      return None

    # If question asks for a name / who, try extraction
    if "name" in q or q.strip().startswith("who") or "recipient" in q or "ceo" in q:
      candidate = extract_recipient(context)
      if candidate:
        return candidate

    # Generic concise synthesis: pick sentences from the retrieved context
    # split into sentences and return the first 2 informative sentences
    # Remove chunk markers for readability
    cleaned = re.sub(r"\[chunk_\d+\].*?:", "", context)
    # naive sentence split
    sents = re.split(r"(?<=[.!?])\s+", cleaned)
    informative = [s.strip() for s in sents if len(s.strip()) > 20]
    if informative:
      snippet = " ".join(informative[:2])
      return f"Answer (based on retrieved documents): {snippet}"

    # fallback to returning the raw context if nothing else
    return context.strip()
