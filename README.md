# PDF-RAG-Agent

A Retrieval-Augmented Generation (RAG) system for querying PDF documents. Built with LangChain, LangGraph, FastAPI, and Streamlit.

The agent retrieves relevant content from your PDFs, generates answers using a local LLM, and falls back to web search (via Tavily) when the documents don't contain sufficient information.

---



## How it works

```
User question
     │
     ▼
Classifier — is this a "fresh/recent" query?
     │
     ├── Yes → Web Search → Generate answer
     │
     └── No  → Retrieve from vectorstore
                    │
                    ├── Sufficient? → Generate → Grade answer
                    │                                  │
                    │                     ┌────────────┤
                    │                     │            │
                    │               Good enough   Not good enough
                    │                     │            │
                    │                  Answer      Web Search
                    │                              → Generate answer
                    │
                    └── Not sufficient → Web Search → Generate answer
```

---

## Project Structure

```
pdf-rag-api/
├── app/                  # FastAPI application
│   ├── main.py           # App entry point + startup pipeline
│   ├── config.py         # Settings loaded from .env
│   ├── schemas.py        # Request/response models
│   └── routers/
│       ├── health.py     # GET  /health
│       ├── query.py      # POST /api/query
│       └── ingest.py     # POST /api/ingest
├── pipeline/             # Core RAG logic
│   ├── extractor.py      # PDF text extraction (PyMuPDF + OCR fallback)
│   ├── chunker.py        # Token / semantic / agentic chunking
│   ├── vectorstore.py    # FAISS / Chroma / PGVector
│   ├── generator.py      # LLM wrapper (HuggingFace)
│   └── rag_pipeline.py   # Orchestrates extract → chunk → index
├── agent/                # LangGraph agentic layer
│   ├── state.py          # AgentState definition
│   ├── nodes.py          # All graph nodes + routers
│   ├── graph.py          # Graph wiring + compilation
│   └── websearch.py      # Tavily web search wrapper
├── config/
│   └── config.json       # LLM prompt templates
├── k8s/                  # Kubernetes manifests
├── ui.py                 # Streamlit chat interface
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Quickstart

### 1. Clone and set up environment

```bash
git clone https://github.com/your-username/pdf-rag-agent.git
cd pdf-rag-agent

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

### 2. Set up environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your API keys:

```
HUGGING_FACE_API=hf_your_token_here
TAVILY_API_KEY=tvly_your_key_here
```

### 3. Run the API

```bash
uvicorn app.main:app --reload
```

The API starts at `http://localhost:8000`. Visit `http://localhost:8000/docs` to see the interactive API explorer.

### 4. Run the UI (separate terminal)

```bash
streamlit run ui.py
```

UI opens at `http://localhost:8501`.

---

## Docker

```bash
# Build and run
docker compose up --build

# Run in background
docker compose up -d
```

---

## Kubernetes

```bash
# Create secrets
kubectl create secret generic pdf-rag-secrets \
  --from-literal=HUGGING_FACE_API=hf_xxx \
  --from-literal=TAVILY_API_KEY=tvly_xxx

# Deploy
kubectl apply -f k8s/
```

---

## Tech Stack

| Component | Technology |
|---|---|
| API framework | FastAPI |
| UI | Streamlit |
| LLM orchestration | LangChain + LangGraph |
| LLM | HuggingFace Transformers (Llama 3.1) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector store | FAISS (default), Chroma, PGVector |
| PDF extraction | PyMuPDF + Tesseract OCR |
| Web search | Tavily |
| Containerisation | Docker + Kubernetes |