# Privacy Compliance Intelligence Platform

A compliance-focused RAG application for auditing privacy policy PDFs against GDPR, CCPA, HIPAA, and NIST requirements.

The project combines Spark-based PDF ingestion, Azure AI Search retrieval, a Neo4j regulation knowledge graph, an Azure OpenAI-compatible LLM, a LangGraph compliance workflow, and a React frontend.

This is an assistive compliance review tool. It can surface possible gaps and remediation steps, but it does not replace legal review.

---

## Architecture

```text
Policy / Regulation PDFs
        |
        v
Spark ingestion pipeline
pipeline/spark_ingestion.py
- extract text with PyMuPDF / pymupdf4llm
- use Azure Document Intelligence OCR only for scanned PDFs
- chunk text
- create embeddings
- upload chunks to Azure AI Search
        |
        v
Azure AI Search
- compliance-policies
- compliance-regulations

Regulation PDFs
        |
        v
Neo4j KG builder
pipeline/kg_builder.py
- extract regulation obligations and concepts
- store regulation graph in Neo4j
- store conflict/stricter-than relationships

User
        |
        v
React frontend -> FastAPI backend -> LangGraph compliance agent
        |
        +--> Azure AI Search: policy and regulation chunks
        +--> Neo4j AuraDB: regulation obligations/conflicts
        +--> Azure OpenAI-compatible LLM: gap analysis and remediation
```

Important: privacy policies are not stored as policy nodes in Neo4j. Neo4j is used for the regulation knowledge graph.

---

## Main Runtime Flow

```text
User question + optional selected policy
  -> FastAPI query endpoint
  -> LangGraph workflow
  -> resolve target policy
  -> detect relevant frameworks
  -> retrieve policy/regulation chunks from Azure AI Search
  -> retrieve regulation obligations/conflicts from Neo4j
  -> compare policy text against obligations
  -> score gaps by severity
  -> generate remediation checklist
  -> return structured result to frontend
```

Active LangGraph nodes:

```text
doc_resolver -> jurisdiction_detector -> kg_retriever -> gap_analyzer -> conflict_detector -> risk_scorer -> remediation -> report_generator
```

---

## Preparing Data Stores

Run these before using the app for audits.

### 1. Build the regulation knowledge graph

```python
from pipeline.kg_builder import ComplianceKGBuilder

builder = ComplianceKGBuilder()
summary = builder.build_from_pdfs("data/regulations", clear_first=True)
print(summary)
```

This reads regulation PDFs, extracts regulation concepts/obligations with the LLM graph transformer, and writes the regulation graph to Neo4j.

### 2. Index regulation PDFs in Azure AI Search

```bash
python -m pipeline.spark_ingestion --input_dir data/regulations --strategy token --doc_type regulation --index compliance-regulations
```

### 3. Index policy PDFs in Azure AI Search

```bash
python -m pipeline.spark_ingestion --input_dir data/policies --strategy token --doc_type policy --index compliance-policies
```

The same `pipeline/spark_ingestion.py` file handles extraction, scanned-PDF OCR fallback, chunking, embeddings, and Azure AI Search upload.

Policies can also be uploaded from the frontend Audit page through the embedded ingest component.

---

## Setup

### Backend

```bash
pip install -r requirements.txt
```

Create a root `.env` file:

```env
AZURE_SEARCH_ENDPOINT=https://<resource>.search.windows.net
AZURE_SEARCH_KEY=<key>
AZURE_SEARCH_REGULATIONS_INDEX=compliance-regulations
AZURE_SEARCH_POLICIES_INDEX=compliance-policies

AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/openai/v1
AZURE_OPENAI_KEY=<key>
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini

AZURE_DOC_INTEL_ENDPOINT=https://<resource>.cognitiveservices.azure.com/
AZURE_DOC_INTEL_KEY=<key>

NEO4J_URI=neo4j+s://<instance>.databases.neo4j.io
NEO4J_USERNAME=<username>
NEO4J_PASSWORD=<password>
NEO4J_DATABASE=neo4j

APPLICATIONINSIGHTS_CONNECTION_STRING=<optional>
MLFLOW_TRACKING_URI=<optional>
```

Run the backend:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Create `frontend/.env`:

```env
VITE_API_URL=http://localhost:8000
```

If using ngrok/Colab, set `VITE_API_URL` to the public backend URL and restart `npm run dev`.

---

## Frontend Pages

| Page | Purpose |
|---|---|
| Dashboard | Backend status and sample audit questions. |
| Audit | Main audit UI, policy dropdown, streaming progress, gaps, remediation. |
| History | Recent audit runs and browser-downloaded history report. |

---

## API Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /api/v1/health` | Liveness check. |
| `GET /api/v1/health/ready` | Readiness check. |
| `POST /api/v1/query` | Run an audit query. |
| `POST /api/v1/query/stream` | Run an audit with streaming progress. |
| `POST /api/v1/ingest` | Upload policy PDFs. |
| `GET /api/v1/policies` | List indexed policy documents. |
| `GET /api/v1/history` | Read MLflow-backed audit history when configured. |
| `GET /api/v1/conflicts` | Read regulation conflicts from Neo4j. |
| `GET /api/v1/regulations` | List indexed regulations. |
| `GET /api/v1/regulations/chunks` | Browse regulation chunks. |

---

## Tracking and Monitoring

### MLflow / Azure MLflow

The backend logs successful audit runs through `mlops/compliance_tracker.py`.

Logged items include:

| Type | Examples |
|---|---|
| Parameters | query, selected policy, jurisdictions |
| Metrics | compliance score, risk score, gap counts, latency |
| Artifacts | report markdown, gaps JSON, risk scores JSON |

For Azure MLflow, set:

```env
MLFLOW_TRACKING_URI=azureml://<region>.api.azureml.ms/mlflow/v2.0/subscriptions/<subscription-id>/resourceGroups/<resource-group>/providers/Microsoft.MachineLearningServices/workspaces/<workspace-name>
```

Then restart the backend and run an audit. The experiment name is:

```text
privacy-compliance-audits
```

### Application Insights

Application Insights is used for runtime monitoring, not experiment artifacts.

Set:

```env
APPLICATIONINSIGHTS_CONNECTION_STRING=<your-application-insights-connection-string>
```

Then restart the backend and run an audit. In Application Insights, check Logs or Transaction Search for events such as:

```text
query_received
node_jurisdiction_detector
node_gap_analyzer
node_risk_scorer
query_completed
query_error
```

---

## Repository Structure

```text
agent/                     LangGraph compliance workflow
app/                       FastAPI backend
frontend/                  React + Vite frontend
mlops/                     MLflow tracking helper
pipeline/kg_builder.py     Neo4j regulation KG builder
pipeline/spark_ingestion.py Spark PDF ingestion, extraction, chunking, embeddings, Azure Search upload
data/regulations/          Regulation PDFs
data/policies/             Policy PDFs
```

---

## Limitations

- This system provides compliance assistance, not legal advice.
- Policy filename changes do not anonymize company names inside PDF text.
- If multiple policies exist and no policy is selected, the system may ask for clarification.
- MLflow and Application Insights are optional observability layers.