import os
from pathlib import Path
import streamlit as st

from backend import parser
from backend.rag_pipeline import run_rag


st.set_page_config(page_title="Enterprise Knowledge Assistant", layout="wide")

# Load HTC styling
css_path = Path(__file__).parent / "assets" / "styles.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as style_file:
        st.markdown(f"<style>{style_file.read()}</style>", unsafe_allow_html=True)

# Initialize session state
if "chunks" not in st.session_state:
    st.session_state["chunks"] = []
if "documents_processed" not in st.session_state:
    st.session_state["documents_processed"] = 0
if "questions_asked" not in st.session_state:
    st.session_state["questions_asked"] = 0
if "last_doc_name" not in st.session_state:
    st.session_state["last_doc_name"] = ""

if "processed_doc_names" not in st.session_state:
    st.session_state["processed_doc_names"] = []

# Try to load persisted chunks/index at startup
try:
    from backend.vectorstore import load_chunks, load_index

    persisted_chunks = None
    try:
        persisted_chunks = load_chunks()
    except FileNotFoundError:
        persisted_chunks = None

    if persisted_chunks and not st.session_state.get("chunks"):
        st.session_state["chunks"] = persisted_chunks
    # Attempt to load FAISS index into memory so searches work immediately
    try:
        load_index()
    except Exception:
        # index may not exist yet; that's okay
        pass
except Exception:
    # ignore errors during startup loading; app will work without persistence
    pass


# Top navbar (compact)
st.markdown(
    """
    <div class="navbar-container">
        <div class="navbar-left">
            <div class="navbar-logo">HTC</div>
            <div class="navbar-title">Enterprise Knowledge Assistant</div>
        </div>
        <div class="navbar-right">
            <div class="navbar-avatar">AD</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# Note: sidebar removed per user request


# Main content
st.markdown('<div class="main-content"><div class="content-wrapper">', unsafe_allow_html=True)

# Minimal hero (title + subtitle)
st.markdown(
    """
    <div class="hero-section">
        <h1 class="hero-title">Enterprise Knowledge Assistant</h1>
        <p class="hero-subtitle">AI-powered enterprise document intelligence</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# Upload area (main) - allow multiple files
uploaded_files = st.file_uploader("Upload documents (PDF or DOCX)", type=["pdf", "docx"], accept_multiple_files=True, label_visibility="visible")

# Append mode toggle - when false, new uploads replace existing chunks
append_mode = st.checkbox("Append to existing documents", value=True, key="append_docs")

process_clicked = st.button("Process Documents")
clear_clicked = st.button("Clear All Documents")

if clear_clicked:
    # Clear session and remove persisted index/chunks
    st.session_state["chunks"] = []
    st.session_state["documents_processed"] = 0
    st.session_state["processed_doc_names"] = []
    st.session_state["last_doc_name"] = ""
    st.session_state["chat_history"] = []
    st.session_state["questions_asked"] = 0
    if "chat_input" in st.session_state:
        st.session_state["chat_input"] = ""
    # remove persisted files if present
    vdb = Path(__file__).resolve().parent.parent / "vector_db"
    idx = vdb / "faiss_index" / "index.faiss"
    chunks_file = vdb / "chunks.json"
    try:
        if idx.exists():
            idx.unlink()
        if chunks_file.exists():
            chunks_file.unlink()
    except Exception as e:
        st.warning(f"Could not remove persisted files: {e}")
    try:
        from backend.vectorstore import clear_index_cache

        clear_index_cache()
    except Exception:
        pass
    st.success("Cleared all processed documents, chat memory, and persisted index.")

if process_clicked:
    if not uploaded_files:
        st.warning("Please upload one or more documents first")
    else:
        all_new_chunks = []
        new_names = []
        # Processing with spinner
        with st.spinner("Processing documents..."):
            for uploaded_file in uploaded_files:
                try:
                    chunks = parser.parse_document(uploaded_file)
                except Exception:
                    st.error(f"Unable to process {getattr(uploaded_file, 'name', 'uploaded')}. Skipping.")
                    chunks = []
                if chunks:
                    # filter out boilerplate and short chunks before adding
                    try:
                        cleaned = parser.clean_chunks(chunks)
                    except Exception:
                        cleaned = chunks
                    if cleaned:
                        all_new_chunks.extend(cleaned)
                    new_names.append(getattr(uploaded_file, "name", "uploaded_document"))

        if all_new_chunks:
            # Either append or replace existing chunks
            if append_mode:
                st.session_state["chunks"].extend(all_new_chunks)
                st.session_state["documents_processed"] += len(new_names)
                st.session_state["processed_doc_names"].extend(new_names)
            else:
                st.session_state["chunks"] = list(all_new_chunks)
                st.session_state["documents_processed"] = len(new_names)
                st.session_state["processed_doc_names"] = list(new_names)

            st.session_state["last_doc_name"] = new_names[-1]

            # Build embeddings and create FAISS index, then persist both
            try:
                from backend.embeddings import generate_embeddings
                from backend.vectorstore import create_index, save_index, save_chunks

                vectors = generate_embeddings(st.session_state["chunks"])
                create_index(vectors)
                save_index()
                save_chunks(st.session_state["chunks"])
            except Exception as e:
                st.error(f"Error creating or saving vector index: {e}")

            st.success(f"Processed {len(new_names)} document(s) successfully")
        else:
            st.info("No valid chunks extracted from uploaded files.")


# Metrics
col1, col2, col3 = st.columns(3, gap="medium")
with col1:
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">Documents Processed</div>
          <div class="metric-value">{st.session_state['documents_processed']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">Chunks Created</div>
          <div class="metric-value">{len(st.session_state['chunks'])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">Questions Asked</div>
          <div class="metric-value">{st.session_state['questions_asked']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Preview processed chunks inside an expander
if st.session_state["chunks"]:
    with st.expander("Preview Processed Chunks", expanded=True):
        # Document metadata
        doc_name = st.session_state.get("last_doc_name", "-")
        num_chunks = len(st.session_state["chunks"])
        chars_extracted = sum(len(c) for c in st.session_state["chunks"]) if num_chunks else 0

        md = "**Document Name:** {}<br>**Number of Chunks:** {}<br>**Characters Extracted:** {}".format(
            doc_name, num_chunks, chars_extracted
        )
        st.markdown(md, unsafe_allow_html=True)

        # Show first three chunks
        preview = st.session_state["chunks"][:3]
        for i, chunk in enumerate(preview, start=1):
            st.markdown(f"**Chunk {i}**")
            st.write(chunk)
else:
    st.info("No processed chunks yet. Use the upload area above and click Process Documents.")

# Chat panel and RAG-enabled question handling
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []  # list of dicts: {role: 'user'|'assistant', text: str, citations: list, chunk_ids: list}

st.markdown('<div style="margin-top:24px;">', unsafe_allow_html=True)
st.markdown('<div class="chat-section-title">Enterprise Chat</div>', unsafe_allow_html=True)

col_left, col_right = st.columns([2, 3])

with col_left:
    user_question = st.text_input("Ask the assistant about your uploaded documents", key="chat_input")
    if st.button("Send"):
        if not user_question or not user_question.strip():
            st.warning("Please enter a question.")
        else:
            if not st.session_state.get("chunks"):
                st.warning("No documents processed. Upload and process documents first.")
            else:
                # record user turn
                st.session_state["chat_history"].append({"role": "user", "text": user_question.strip(), "citations": [], "chunk_ids": []})
                # call RAG pipeline
                try:
                    # ask for more candidates during retrieval to improve extraction
                    result = run_rag(user_question.strip(), st.session_state.get("chunks", []), top_k=5)
                except Exception as e:
                    st.error(f"Error during RAG pipeline: {e}")
                    result = {"answer": "", "citations": [], "chunk_ids": []}

                answer = result.get("answer", "")
                citations = result.get("citations", [])
                chunk_ids = result.get("chunk_ids", [])

                # record assistant turn
                st.session_state["chat_history"].append({"role": "assistant", "text": answer, "citations": citations, "chunk_ids": chunk_ids})

                # update questions counter
                st.session_state["questions_asked"] += 1

with col_right:
    # Render chat history as scrollable bubbles
    chat_html = ['<div class="chat-history" style="max-height:420px; overflow:auto; padding:12px">']
    if not st.session_state["chat_history"]:
        chat_html.append('<div class="chat-history-empty">No messages yet. Ask a question to begin.</div>')
    else:
        for turn in st.session_state["chat_history"]:
            role = turn.get("role")
            text = turn.get("text", "")
            if role == "user":
                chat_html.append(f'<div class="chat-bubble user" style="background:#E6F0FF;padding:12px;border-radius:12px;margin:8px 0;">{text}</div>')
            else:
                # assistant bubble with citations
                bubble = f'<div class="chat-bubble assistant" style="background:#F3F4F6;padding:12px;border-radius:12px;margin:8px 0;">{text}'
                if turn.get("chunk_ids"):
                    cites = []
                    for cid, ctext in zip(turn.get("chunk_ids", []), turn.get("citations", [])):
                        cites.append(f'<div class="citation" style="font-size:12px;color:#6B7280;margin-top:8px;">Chunk {cid}: {ctext[:140]}...</div>')
                    bubble += "<div class=\"assistant-citations\">" + "".join(cites) + "</div>"
                bubble += '</div>'
                chat_html.append(bubble)

    chat_html.append('</div>')
    st.markdown("".join(chat_html), unsafe_allow_html=True)

st.markdown('</div></div>', unsafe_allow_html=True)

# Dark mode toggle removed with the sidebar
