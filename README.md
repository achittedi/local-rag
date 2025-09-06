# Run a Model Locally with Your Documents (RAG Starter)

A minimal, **fully local** Retrieval‑Augmented Generation (RAG) starter. It runs a local LLM via **Ollama**, indexes your docs with **FAISS**, and answers questions grounded in those docs using **LangChain**.

> Works on **Windows 11**, **macOS**, and **Linux**. Supports CPU or NVIDIA GPU. Includes a CLI and a small **FastAPI** server.


---

## What this does

1. **Ingest**: Load files from `./docs/`, chunk them, and compute embeddings (locally).
2. **Index**: Store embeddings in **FAISS** at `./rag_index/`.
3. **Retrieve + Generate**: At query time, retrieve the top‑k chunks, stuff them into the prompt, and have a local LLM answer **using that context**.


---

## Prerequisites

- **Python 3.10+** and **pip**
- **Ollama** (download: https://ollama.com/download)
  - Start the service/app (`ollama serve` on macOS/Linux; launch the app on Windows).
- Optional: **Docker Desktop** (use the included `docker-compose.yml`)
- Optional: **NVIDIA GPU** (Ollama will use it automatically if available)


---

## Quick Start — Native (recommended)

> **Windows PowerShell** shown; macOS/Linux versions follow each block.

1) Create a virtual environment and install dependencies
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Pull a local LLM + embedding model (first run only)
```powershell
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

3) Put your **PDF / TXT / MD / DOCX** files into `./docs/`  
   (DOCX requires `docx2txt`; install it with `pip install docx2txt`.)

4) Build the vector index
```powershell
python ingest.py
```

5) Ask questions (CLI)
```powershell
python rag_query.py "What are the key takeaways across my onboarding docs?"
```

6) Or run the HTTP API
```powershell
uvicorn api:app --reload --port 8000
# POST {"question":"..."} to http://localhost:8000/ask
```


---

## Quick Start — Docker Compose

> Requires Docker Desktop. For GPU, enable NVIDIA GPU support in Docker Desktop settings.

```powershell
docker compose up -d --build
# Pull models INSIDE the Ollama container:
docker exec -it ollama ollama pull llama3.1:8b
docker exec -it ollama ollama pull nomic-embed-text
# Build the index after adding docs:
docker exec -it rag-api python ingest.py
# Ask via API:
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d "{\"question\":\"Summarize my PDFs\"}"
```

The API container talks to Ollama at `http://ollama:11434` as configured in `docker-compose.yml`.


---

## Project Structure

```
.
├─ docs/                 # Put your PDFs/TXT/MD/DOCX here
├─ rag_index/            # FAISS index (created by ingest.py)
├─ ingest.py             # Load → chunk → embed (Ollama) → build FAISS index
├─ rag_query.py          # CLI to ask questions (MMR retriever, k=4)
├─ api.py                # FastAPI server (POST /ask)
├─ requirements.txt      # Python dependencies
├─ docker-compose.yml    # Ollama + API
└─ Dockerfile            # API image
```


---

## Configuration & Tuning

- **Model (generation)** — in `rag_query.py`:
  ```python
  from langchain_ollama import ChatOllama
  llm = ChatOllama(model="llama3.1:8b", temperature=0.2)
  # Examples: "llama3.1:8b-instruct-q4_K_M", "mistral:7b", "qwen2.5:7b"
  ```

- **Embeddings** — default uses Ollama `nomic-embed-text`. To switch to HuggingFace CPU embeddings:
  ```bash
  pip install sentence-transformers
  ```
  ```python
  from langchain_community.embeddings import HuggingFaceEmbeddings
  embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
  ```

- **Retriever** — `rag_query.py` uses MMR with `k=4`. Tune `k` and try `search_type="similarity"` for tighter matches.

- **Chunking** — `ingest.py` defaults to `chunk_size=800, chunk_overlap=150`. Increase size for long passages; reduce if retrieval gets noisy.

- **Endpoint** — Set `OLLAMA_BASE_URL` if Ollama isn’t on localhost (Docker uses `http://ollama:11434`).


---

## Supported Document Types

- **PDF**, **TXT**, **MD** out of the box.
- **DOCX** with `pip install docx2txt` (the `ingest.py` in this starter already supports it if installed).
- **Scanned PDFs** (image‑only) need OCR → convert to text/markdown or swap in a PDF loader with OCR.


---

## Common Tasks

**Re‑index after adding/updating docs**
```powershell
python ingest.py
```

**Change models** (faster/smaller)
```powershell
ollama pull llama3.1:8b-instruct-q4_K_M
# then edit rag_query.py to point to the new model
```

**Add citations in answers**
- Include each chunk’s metadata (e.g., filename/page) in the final prompt or output. (Easy to add if desired.)

**Hybrid retrieval / reranking**
- Add BM25 for keyword + vector fusion or use a cross‑encoder (e.g., bge‑reranker) to re‑score top hits.


---

## Troubleshooting

**FAISS IndexError (embeddings list empty)**  
- Usually means your files produced **0 text** (unsupported type or scanned PDFs).  
- Ensure you have supported types, install `docx2txt` for DOCX, or OCR scanned PDFs.

**Ollama connection refused**  
- Ensure Ollama is running (`ollama serve` / app open) and models are pulled.

**Slow or OOM**  
- Use quantized models (e.g., `llama3.1:8b-instruct-q4_K_M`), reduce context, or switch to a smaller model family.

**Weird PDF extraction**  
- Some PDFs contain complex layouts. Converting to `.txt`/`.md` can improve quality.


---

## Security Notes

- Everything runs **locally** by default. If you expose the API, restrict it to localhost or put it behind auth/proxy.  
- Keep indexes and docs on encrypted disk if they’re sensitive. Avoid logging full content.


---

## Roadmap

- Source **citations** in responses
- **Hybrid** retrieval (BM25 + vectors)
- **Reranking** with cross‑encoder
- Incremental re‑ingest (watch `docs/`)
- Connectors: **Confluence/Jira/SharePoint** sync


---

## License

MIT (or your org’s standard license).
