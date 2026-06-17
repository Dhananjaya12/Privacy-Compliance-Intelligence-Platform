# Privacy Compliance Intelligence Platform

An AI-powered compliance auditing system that analyzes company privacy policies against regulatory frameworks like **GDPR, CCPA, HIPAA, and NIST** and surfaces actionable gaps, risk levels, and remediation steps. Built on a Neo4j Knowledge Graph, Azure AI Search, and a LangGraph multi-node agent pipeline with a React dashboard.

---

## What It Does

- **Natural-language compliance auditing**: Ask "What GDPR gaps exist in the privacy policy?" and get a structured breakdown of missing obligations, severity levels, and article references.
- **Multi-framework analysis**: A single query audits against multiple regulations simultaneously; the system auto-detects which frameworks are relevant from the query.
- **Cross-document coverage discovery**: Ask "Which of these policies address HIPAA breach notification?" and get a yes/no matrix across all indexed company policies.
- **Knowledge Graph reasoning**: Regulations are stored as a Neo4j graph (2,887 nodes, 5,922 edges, 71 communities). The pipeline uses 1-hop and multi-hop traversal to extract obligations and detect cross-regulation conflicts.
- **Remediation guidance**: Every detected gap includes a concrete recommendation on what clause or disclosure to add.
- **Live streaming pipeline**: The 8-node agent pipeline streams progress to the UI in real time via Server-Sent Events (SSE).
- **Downloadable PDF reports**: Full audit results export as formatted PDF documents.
- **Run history & trends**: All audit runs are tracked in MLflow; the dashboard shows gap trends over time.

---

## Architecture

### Agent Pipeline — 8 LangGraph Nodes

```
User Query
    │
    ▼
┌──────────────────┐
│   doc_resolver   │  ← Identifies target policy document(s)
└────────┬─────────┘   Classifies intent: single-doc audit / cross-doc
         │             coverage discovery / clarification request
         ▼
┌────────────────────────┐
│  jurisdiction_detector │  ← Auto-detects applicable regulations
└──────────┬─────────────┘   (GDPR / CCPA / HIPAA / NIST) from query text
           │
           ▼
┌──────────────────┐
│   kg_retriever   │  ← Fetches obligations from Neo4j KG
└────────┬─────────┘   + policy chunks from Azure AI Search
         │
         ▼
┌──────────────────┐
│   gap_analyzer   │  ← Compares policy text against regulatory obligations
└────────┬─────────┘   per regulation, per document
         │
         ▼
┌───────────────────────┐
│   conflict_detector   │  ← Finds CONFLICTS_WITH / STRICTER_THAN edges in KG
└──────────┬────────────┘   (e.g. GDPR 72h vs. HIPAA 60d breach notification)
           │
           ▼
┌──────────────────┐
│   risk_scorer    │  ← Scores gaps by severity: critical / high / medium / low
└────────┬─────────┘   Groups gaps into thematic clusters
         │
         ▼
┌──────────────────┐
│   remediation    │  ← Generates "add this clause" recommendations per gap
└────────┬─────────┘
         │
         ▼
┌──────────────────────┐
│  report_generator    │  ← Renders structured markdown report
└──────────────────────┘   (gap analysis or coverage matrix, based on intent)
```

### Services & Infrastructure

| Service | Purpose |
|---------|---------|
| **Azure AI Search** | Single vector index (`pdf-rag-index`) storing both regulation and policy chunks; hybrid keyword + semantic search with OData filters scoped by `paper_id` |
| **Azure OpenAI (gpt-4o-mini)** | LLM for gap analysis, entity extraction, report generation, and remediation recommendations |
| **Neo4j AuraDB** | Compliance knowledge graph — 2,887 obligation nodes, 5,922 edges, 38 cross-regulation conflict relationships (`CONFLICTS_WITH` / `STRICTER_THAN`) |
| **MLflow** | Local experiment tracking (`mlruns/`); logs per-audit parameters, gap metrics, and full report artifacts. Can be pointed at an Azure ML workspace by setting `MLFLOW_TRACKING_URI` to the workspace MLflow URI |
| **FastAPI + uvicorn** | REST + SSE streaming backend |
| **React + Vite (TypeScript)** | Frontend dashboard |
| **D3.js** | Interactive knowledge graph explorer |
| **Apache Spark (local mode)** | Batch PDF ingestion — distributes extraction and chunking across CPU cores before uploading to Azure AI Search |
| **pymupdf4llm** | PDF text extraction |

---

## Project Structure

```
PDF-RAG-AGENT/
│
├── agent/                          # LangGraph agent
│   ├── compliance_nodes.py         # All 8 pipeline node implementations
│   ├── graph.py                    # LangGraph StateGraph wiring
│   └── state.py                    # AgentState TypedDict definition
│
├── app/                            # FastAPI backend
│   ├── main.py
│   ├── core/
│   │   ├── config.py               # Environment config (Azure, Neo4j, MLflow)
│   │   └── lifespan.py             # Startup: loads pipeline + agent
│   ├── api/v1/endpoints/
│   │   ├── query.py                # POST /query  +  POST /query/stream (SSE)
│   │   ├── ingest.py               # POST /ingest — upload new policy PDFs
│   │   ├── health.py               # GET /health/ready
│   │   ├── history.py              # GET /history  GET /trends
│   │   ├── conflicts.py            # GET /conflicts
│   │   ├── graph.py                # GET /graph — Neo4j nodes/edges for D3
│   │   ├── regulations.py          # GET /regulations
│   │   └── report.py               # POST /report/pdf
│   ├── models/schemas.py           # Pydantic request/response models
│   └── services/rag_service.py     # Wraps agent; handles SSE streaming
│
├── pipeline/                       # Data ingestion & retrieval
│   ├── kg_builder.py               # Builds Neo4j KG from regulation PDFs
│   ├── compliance_retriever.py     # Azure AI Search + Neo4j queries
│   ├── spark_ingestion.py          # Batch ingestion via Apache Spark (local[*])
│   ├── extractor.py                # PDF text extraction (pymupdf4llm)
│   ├── chunker.py                  # Token-based chunking strategy
│   └── vectorstore.py              # Azure AI Search vectorstore wrapper
│
├── frontend/                       # React + Vite UI (TypeScript)
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Audit.tsx           # Main query + results UI (SSE streaming)
│   │   │   ├── Dashboard.tsx       # Trends + cross-regulation conflicts
│   │   │   ├── History.tsx         # Past audit runs (MLflow-backed)
│   │   │   ├── GraphExplorer.tsx   # D3.js interactive KG visualizer
│   │   │   └── Ingest.tsx          # Upload new policy PDFs
│   │   └── lib/api.ts              # Typed API client (axios + fetch for SSE)
│   ├── .env                        # VITE_API_URL=<backend-url>
│   └── vite.config.ts              # Proxy config — routes /api/* to backend
│
├── mlops/
│   └── compliance_tracker.py       # MLflow run logging wrapper
│
├── data/
│   ├── regulations/                # GDPR, CCPA, HIPAA, NIST PDFs + KG cache
│   └── policy_docs/pdfs/           # Indexed company privacy policy PDFs
│
├── tests/                          # pytest suite (unit + API + integration)
├── config/default.json
├── requirements.txt
└── colab_eval.py                   # Pre-demo eval — 23 test queries, 7 categories
```

---

## Setup & Running

### Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.10+ | Backend runtime |
| Node.js 18+ | Frontend |
| Azure AI Search | One index — `pdf-rag-index` |
| Azure OpenAI | `gpt-4o-mini` deployment |
| Neo4j AuraDB | Free tier (1 GB) is sufficient |

### Environment Variables

Create a `.env` file in the project root:

```env
# Azure AI Search
AZURE_SEARCH_ENDPOINT=https://<your-resource>.search.windows.net
AZURE_SEARCH_KEY=<your-key>
AZURE_SEARCH_INDEX_NAME=pdf-rag-index

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_KEY=<your-key>
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
OPENAI_API_VERSION=2024-02-15-preview

# Neo4j AuraDB
NEO4J_URI=neo4j+s://<your-instance>.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<your-password>
NEO4J_DATABASE=neo4j

# MLflow — local by default; set to Azure ML workspace URI for cloud tracking
MLFLOW_TRACKING_URI=mlruns
```

**Frontend** — create `frontend/.env`:
```env
VITE_API_URL=https://<your-backend-url>
```

---

### Step 1 — Build the Knowledge Graph *(one-time)*

Parses regulation PDFs (GDPR, CCPA, HIPAA, NIST), extracts obligation nodes using `LLMGraphTransformer`, and stores conflict edges in Neo4j:

```python
pip install -r requirements.txt

from pipeline.kg_builder import ComplianceKGBuilder
result = ComplianceKGBuilder().build_from_pdfs("data/regulations/")
# → {'graph_nodes': 2887, 'graph_edges': 5922, 'conflicts_in_graph': 38}
```

### Step 2 — Index Policy Documents *(one-time)*

Uses Apache Spark in local mode to distribute PDF extraction and chunking, then batch-uploads to Azure AI Search:

```python
from pipeline.spark_ingestion import run_ingestion
run_ingestion(input_dir="data/policy_docs/pdfs/", strategy="token", partitions=2)
```

To add a new policy later, use the **Ingest** page in the UI or `POST /api/v1/ingest`.

### Step 3 — Start the Backend

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Wait for the pipeline readiness message before sending queries. The agent loads Neo4j connections and Azure Search clients at startup.

### Step 4 — Start the Frontend

```bash
cd frontend
npm install       # first time only
npm run dev       # → http://localhost:5173
```

> Update `frontend/.env` with the correct backend URL and fully restart `npm run dev` whenever the backend URL changes — the Vite proxy reads it only at server start.

---

## Key Findings

The system successfully processed privacy policy documents against GDPR, CCPA, HIPAA, and NIST regulatory frameworks, producing the following observations:

- **Breach notification obligations** differed significantly between frameworks where GDPR requires supervisory authority notification within 72 hours while HIPAA allows up to 60 days. The knowledge graph captured this as a `STRICTER_THAN` conflict edge, which the `conflict_detector` node surfaces automatically.
- **Right to erasure** (GDPR Art. 17) and medical record retention mandates (HIPAA) represent a structural tension that the system flags for healthcare data controllers operating across jurisdictions.
- **NIST technical controls** (access control, encryption standards) were consistently underrepresented in policy documents, privacy policies rarely make binding commitments at the control level.
- **Query intent classification** correctly distinguished audit queries (gap analysis mode) from coverage discovery queries (cross-document matrix mode), with clarification requests returned for ambiguous inputs.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent orchestration | LangGraph (StateGraph) |
| LLM | Azure OpenAI gpt-4o-mini |
| Vector search | Azure AI Search |
| Knowledge graph | Neo4j AuraDB + LangChain LLMGraphTransformer |
| Backend framework | FastAPI + uvicorn |
| Frontend | React 18, TypeScript, Vite, Tailwind CSS |
| Graph visualization | D3.js (force-directed) |
| ML experiment tracking | MLflow (local or Azure ML) |
| Batch ingestion | Apache Spark (local mode) |
| PDF extraction | pymupdf4llm |
