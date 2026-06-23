Setup and run

1. (Recommended) create and activate a virtual environment.

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```bash
pip install -r requirements.txt
# If `faiss-cpu` fails on Windows, use conda:
# conda install -c conda-forge faiss-cpu
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

Notes

- The app persists chunks to `vector_db/chunks.json` and the FAISS index to `vector_db/faiss_index/index.faiss`.
- `backend/llm.py` currently uses a safe local fallback. To re-enable a remote LLM, update `backend/llm.py` with an appropriate client and credentials.
