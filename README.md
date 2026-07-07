# My RAG Project

Minimal README to set up and run the project locally.

## Setup (Windows PowerShell)

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If `faiss-cpu` installation fails on your platform, consider installing a prebuilt wheel or use `pip install faiss-gpu` if you have CUDA.

## Prepare data

Place PDFs into the `pdfs/` folder, then run:

```powershell
python prepare_data.py
```

This will create `data_chunks/` with chunked text files.

## Build vector store (FAISS)

```powershell
python build_vector_store.py
```

This will create `faiss_index/` with the saved index.

## Run the Streamlit app

```powershell
streamlit run query_rag_hf.py
```

## Notes

- Environment variables (e.g., `GEMINI_API_KEY`) can be placed in a `.env` file for `query_rag_hf.py`.
- If you prefer ChromaDB (used by the Streamlit app), ensure `chromadb` is configured and `chroma_db/` is writable.
