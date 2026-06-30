# Privacy Compliance Intelligence Platform

A compliance-focused RAG application for auditing privacy policy PDFs against regulatory frameworks such as GDPR, CCPA, HIPAA, and NIST.

The project combines Spark-based PDF ingestion, Azure AI Search retrieval, a Neo4j regulatory-framework knowledge graph, an Azure OpenAI-compatible LLM, a LangGraph compliance workflow, and a React frontend.

This is an assistive compliance review tool. It can surface possible gaps and remediation steps, but it does not replace legal review.

**Demo video:** [Privacy-Compliance-Intelligence-Platform Demo Walkthrough.mp4](https://drive.google.com/file/d/1fJ116xLkkeYu2BC7wDhsTNAe9oC7oX2M/view?usp=sharing)

---

## Architecture

```text
Policy / Regulatory-framework PDFs
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

Regulatory-framework PDFs
        |
        v
Neo4j KG builder
pipeline/kg_builder.py
- extract framework obligations and concepts
- store framework graph in Neo4j
- store connected obligation/conflict/stricter-than relationships

User
        |
        v
React frontend -> FastAPI backend -> LangGraph compliance agent
        |
        +--> Azure AI Search: policy and framework chunks
        +--> Neo4j AuraDB: connected framework obligations and conflicts
        +--> Azure OpenAI-compatible LLM: gap analysis and remediation
```

Important: privacy policies are not stored as policy nodes in Neo4j. Neo4j is used for the regulatory-framework knowledge graph.

---

## Main Runtime Flow

```text
User question + optional selected policy
  -> FastAPI query endpoint
  -> LangGraph workflow
  -> resolve target policy
  -> detect relevant regulatory frameworks
  -> retrieve policy/framework chunks from Azure AI Search
  -> retrieve connected framework obligations/conflicts from Neo4j
  -> compare policy text against obligations
  -> score gaps by severity
  -> generate remediation checklist
  -> return structured result to frontend
```

Active LangGraph nodes:

```text
doc_resolver -> jurisdiction_detector -> kg_retriever -> gap_analyzer -> conflict_detector -> risk_scorer -> remediation -> report_generator
```

### How the audit is implemented

This section maps the audit behavior to the actual files and functions in the project.

| Step | Implementation | What happens in code |
|---|---|---|
| PDF ingestion | `pipeline/spark_ingestion.py` -> `run_ingestion()` | Scans the input folder, distributes PDFs across Spark partitions, extracts text, chunks it, embeds it, and uploads documents to the selected Azure AI Search index. |
| Text extraction | `pipeline/spark_ingestion.py` -> `extract_text()` | Uses PyMuPDF / `pymupdf4llm` for text PDFs. If `is_scanned()` finds very little embedded text, it falls back to Azure Document Intelligence OCR. |
| Chunking | `pipeline/spark_ingestion.py` -> `chunk_text()` | Uses token chunking by default; semantic chunking is available through `SemanticChunker`. Each chunk keeps metadata such as `paper_id`, `doc_type`, `regulation`, `page`, and `chunk_id`. |
| Framework KG build | `pipeline/kg_builder.py` -> `ComplianceKGBuilder.build_from_pdfs()` | Reads framework PDFs from `data/regulations`, splits them into article/section chunks, runs `LLMGraphTransformer`, and writes graph documents to Neo4j. |
| KG node/edge schema | `pipeline/kg_builder.py` -> `ALLOWED_NODES`, `ALLOWED_RELATIONSHIPS` | Nodes include `Regulation`, `Article`, `Obligation`, `Right`, `Entity`, `Concept`, `Penalty`, and `Timeframe`. Edges include `REQUIRES`, `GRANTS`, `REFERENCES`, `MAPS_TO`, `PART_OF`, `DEFINES`, `APPLIES_TO`, `IMPOSES`, `CONFLICTS_WITH`, and `STRICTER_THAN`. |
| Value conflict extraction | `pipeline/kg_builder.py` -> `_extract_value_conflicts()` | Runs a dedicated LLM pass over framework text and writes concrete `CONFLICTS_WITH` / `STRICTER_THAN` edges using Cypher. Stored properties include `concept`, `value_a`, `value_b`, `unit`, `description`, and `source_quote`. |
| Policy selection | `agent/compliance_nodes.py` -> `doc_resolver_node()` | Uses the frontend dropdown value first. If absent, it tries filename/query matching; if multiple policies are possible, it returns a clarification response. |
| Framework detection | `agent/compliance_nodes.py` -> `jurisdiction_detector_node()` | Detects the applicable regulatory frameworks from the question and available context. These names are then used to scope obligations, conflicts, and scoring. |
| Evidence retrieval | `agent/compliance_nodes.py` -> `kg_retriever_node()` and `pipeline/compliance_retriever.py` -> `_azure_search()` | Retrieves policy chunks from Azure AI Search with a filter like `doc_type eq 'policy' and paper_id eq '<selected.pdf>'`. It also retrieves framework chunks from the framework index. |
| KG context retrieval | `pipeline/compliance_retriever.py` -> `multi_hop()` | Extracts important terms from the retrieved text, then asks Neo4j for nearby connected framework facts. In plain English: if the query mentions breach notification, the graph helps pull related duties, timeframes, rights, penalties, and framework sections connected to that concept. |
| Obligation structuring | `agent/compliance_nodes.py` -> `_triples_to_obligations()` | Converts Neo4j relationship results into simple obligation records with framework, obligation type, source, and text fields. Keyword rules classify obligations such as breach notification, data subject rights, access control, encryption, retention, consent, and lawful basis. |
| Gap analysis | `agent/compliance_nodes.py` -> `gap_analyzer_node()` and `_analyze_gaps_with_llm()` | Sends the selected policy text plus structured obligations to the LLM in batches. The LLM returns JSON gaps with severity, framework, obligation ID, theme, evidence, and missing/weak policy language. |
| Conflict filtering | `agent/compliance_nodes.py` -> `conflict_detector_node()` | Reads `CONFLICTS_WITH` and `STRICTER_THAN` edges from Neo4j and keeps only conflicts relevant to the detected frameworks for that audit. |
| Priority scoring | `agent/compliance_nodes.py` -> `risk_scorer_node()` | Converts gaps into a simple prioritization score for the UI/report. Severity weights are `critical=10`, `high=7.5`, `medium=5`, `low=2.5`, `info=1`. Framework weights are `GDPR=0.35`, `HIPAA=0.30`, `CCPA=0.20`, `NIST=0.15`. The displayed compliance score is calculated by `risk_to_compliance() = 100 - risk * 10`, clamped to 0-100. This is not a legal certification score; it is a way to rank and summarize findings. |
| Remediation | `agent/compliance_nodes.py` -> `remediation_node()` and `_generate_remediations()` | Groups gaps by closed-vocabulary themes and asks the LLM for one concrete checklist action per theme. The final report renders these as remediation checklist items. |
| Runtime telemetry | `app/core/telemetry.py` and `agent/compliance_nodes.py` | Emits visible console logs and Application Insights traces for `query_received`, `node_jurisdiction_detector`, `node_gap_analyzer`, `node_risk_scorer`, and `query_completed`. |

The important implementation detail is that the app does not simply ask an LLM to judge a policy from scratch. It first retrieves policy evidence from Azure AI Search, uses Neo4j to pull connected framework facts around the audit topic, converts those graph results into structured obligations, and then asks the LLM to compare the policy language against those obligations.

KG conflicts are value-level framework differences, for example different deadlines, thresholds, or limits for the same obligation. The conflict extraction focuses on breach notification deadlines, deletion/erasure response deadlines, data subject access deadlines, minor consent age thresholds, retention limits, and opt-out response deadlines.

---

## Preparing Data Stores

Run these before using the app for audits.

### 1. Build the regulatory-framework knowledge graph

```python
from pipeline.kg_builder import ComplianceKGBuilder

builder = ComplianceKGBuilder()
summary = builder.build_from_pdfs("data/regulations", clear_first=True)
print(summary)
```

This reads framework PDFs, extracts framework concepts/obligations with the LLM graph transformer, and writes the framework graph to Neo4j.

### 2. Index regulatory-framework PDFs in Azure AI Search

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
| `GET /api/v1/conflicts` | Read framework conflicts from Neo4j. |
| `GET /api/v1/regulations` | List indexed regulatory frameworks. |
| `GET /api/v1/regulations/chunks` | Browse regulatory-framework chunks. |

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
pipeline/kg_builder.py     Neo4j regulatory-framework KG builder
pipeline/spark_ingestion.py Spark PDF ingestion, extraction, chunking, embeddings, Azure Search upload
data/regulations/          Regulatory-framework PDFs
data/policies/             Policy PDFs
```

---

## Limitations

- This system provides compliance assistance, not legal advice.
- Policy filename changes do not anonymize company names inside PDF text.
- If multiple policies exist and no policy is selected, the system may ask for clarification.
- MLflow and Application Insights are optional observability layers.