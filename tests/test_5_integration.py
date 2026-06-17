"""
tests/test_5_integration.py

Layer 5 — Integration smoke tests.
These tests make REAL network calls to Azure AI Search, Neo4j AuraDB,
and Azure OpenAI. They are skipped automatically if credentials are absent.

Run only when you have a live .env:
    python -m pytest tests/test_5_integration.py -v -s

Or a single stage:
    python -m pytest tests/test_5_integration.py::TestAzureSearch -v -s
"""

import os
import pytest
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


# ── Skip guard ─────────────────────────────────────────────────────────────────

def _require_env(*keys):
    missing = [k for k in keys if not os.getenv(k)]
    if missing:
        pytest.skip(f"Skipped: missing env vars {missing}")


# ── Azure AI Search ────────────────────────────────────────────────────────────

class TestAzureSearch:

    def test_can_connect_and_count_documents(self):
        _require_env("AZURE_SEARCH_ENDPOINT", "AZURE_SEARCH_KEY")

        from azure.search.documents import SearchClient
        from azure.core.credentials import AzureKeyCredential

        index_name = os.getenv("AZURE_SEARCH_INDEX_NAME", "pdf-rag-index")
        client = SearchClient(
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            index_name=index_name,
            credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY")),
        )
        count = client.get_document_count()
        print(f"\n  Azure Search index '{index_name}': {count} documents")
        assert count >= 0   # just confirms the connection works

    def test_search_returns_results_for_compliance_query(self):
        _require_env("AZURE_SEARCH_ENDPOINT", "AZURE_SEARCH_KEY")

        from pipeline.vectorstore import VectorStoreFactory

        with pytest.MonkeyPatch.context() as mp:
            # Use a lightweight local embedding for the test call
            pass

        factory = VectorStoreFactory(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            index_name=os.getenv("AZURE_SEARCH_INDEX_NAME", "pdf-rag-index"),
        )
        # Build without uploading — just connect
        factory.vectorstore = __import__(
            "langchain_community.vectorstores.azuresearch", fromlist=["AzureSearch"]
        ).AzureSearch(
            azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            azure_search_key=os.getenv("AZURE_SEARCH_KEY"),
            index_name=os.getenv("AZURE_SEARCH_INDEX_NAME", "pdf-rag-index"),
            embedding_function=factory.embeddings.embed_query,
        )

        results = factory.vectorstore.similarity_search("breach notification", k=3)
        print(f"\n  Search results for 'breach notification': {len(results)}")
        for r in results:
            print(f"    - {r.page_content[:80]}...")
        # Index may be empty in a fresh environment — just check no exception
        assert isinstance(results, list)


# ── Neo4j AuraDB ──────────────────────────────────────────────────────────────

class TestNeo4j:

    def test_can_connect_and_count_nodes(self):
        _require_env("NEO4J_URI", "NEO4J_PASSWORD")

        from neo4j import GraphDatabase

        uri      = os.getenv("NEO4J_URI")
        user     = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD")
        database = os.getenv("NEO4J_DATABASE", "neo4j")

        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session(database=database) as session:
            result = session.run("MATCH (n) RETURN count(n) AS count")
            count  = result.single()["count"]

        driver.close()
        print(f"\n  Neo4j node count in '{database}': {count}")
        # If KG is built, expect 2174 nodes per briefing
        assert count >= 0

    def test_triples_exist_for_gdpr(self):
        _require_env("NEO4J_URI", "NEO4J_PASSWORD", "NEO4J_DATABASE")

        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD")),
        )
        with driver.session(database=os.getenv("NEO4J_DATABASE")) as session:
            result = session.run(
                "MATCH (a)-[r]->(b) WHERE r.regulation = 'GDPR' "
                "RETURN count(r) AS count LIMIT 1"
            )
            count = result.single()["count"]

        driver.close()
        print(f"\n  GDPR triples in Neo4j: {count}")
        # Non-zero confirms KG is populated
        assert count > 0, "No GDPR triples found — has the KG been built?"


# ── Azure OpenAI ──────────────────────────────────────────────────────────────

class TestAzureOpenAI:

    def test_gpt4o_mini_responds(self):
        _require_env("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_KEY", "AZURE_OPENAI_DEPLOYMENT")

        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
            temperature=0,
        )
        response = llm.invoke("Reply with the single word: OK")
        print(f"\n  Azure OpenAI response: '{response.content}'")
        assert "ok" in response.content.lower()


# ── ComplianceRetriever end-to-end ────────────────────────────────────────────

class TestComplianceRetrieverE2E:
    """
    Full pipeline: Azure Search + Neo4j + gpt-4o-mini.
    Mirrors what test_compliance_phase2.py Stage 1 does,
    but as a proper pytest test with assertions.
    """

    def test_query_returns_required_keys(self):
        _require_env(
            "AZURE_SEARCH_ENDPOINT", "AZURE_SEARCH_KEY",
            "NEO4J_URI", "NEO4J_PASSWORD", "NEO4J_DATABASE",
            "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_KEY",
        )

        from pipeline.compliance_retriever import ComplianceRetriever

        retriever = ComplianceRetriever()
        result    = retriever.query("Does the policy address GDPR breach notification?")

        required_keys = {"answer", "gaps", "jurisdictions", "raw_chunks_used",
                         "raw_triples_used", "latency_ms"}
        assert required_keys.issubset(result.keys()), \
            f"ComplianceRetriever.query() missing keys: {required_keys - result.keys()}"

        print(f"\n  Jurisdictions: {result['jurisdictions']}")
        print(f"  Chunks used:   {result['raw_chunks_used']}")
        print(f"  Triples used:  {result['raw_triples_used']}")
        print(f"  Gaps:          {len(result['gaps'])}")
        print(f"  Latency:       {result['latency_ms']}ms")
        print(f"  Answer: {result['answer'][:200]}...")

        assert isinstance(result["jurisdictions"], list)
        assert len(result["answer"]) > 0
        retriever.close()

    def test_jurisdiction_detection_gdpr(self):
        _require_env("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_KEY")

        from pipeline.compliance_retriever import ComplianceRetriever

        # detect_jurisdictions is a static method — no Azure Search needed
        hits = ComplianceRetriever.detect_jurisdictions(
            "Does the policy comply with GDPR Article 17 right to erasure?", []
        )
        assert "GDPR" in hits, f"Expected GDPR in detected jurisdictions, got: {hits}"

    def test_jurisdiction_detection_hipaa(self):
        _require_env("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_KEY")

        from pipeline.compliance_retriever import ComplianceRetriever

        hits = ComplianceRetriever.detect_jurisdictions(
            "How does the policy handle protected health information and PHI?", []
        )
        assert "HIPAA" in hits, f"Expected HIPAA in detected jurisdictions, got: {hits}"


# ── Full Phase 2 graph nodes (requires all credentials) ──────────────────────

class TestPhase2NodesE2E:
    """
    Runs the full 8-node compliance chain directly (no graph compile needed).
    Mirrors agent/graph.py's linear ordering:
      doc_resolver -> jurisdiction_detector -> kg_retriever -> gap_analyzer
      -> conflict_detector -> risk_scorer -> remediation -> report_generator
    """

    def _blank_state(self, query: str):
        return {
            "query":                query,
            "final_answer":         "",
            "policy_document":      None,
            "target_documents":     [],
            "clarification_needed": None,
            "jurisdictions":        [],
            "kg_chunks":            [],
            "kg_triples":           [],
            "obligations":          [],
            "per_doc_results":      {},
            "gaps":                 [],
            "conflicts":            [],
            "risk_scores":          {},
            "per_reg_compliance":   {},
            "overall_score":        0.0,
            "compliance_score":     0.0,
            "financial_exposure":   "",
            "remediations":         [],
            "compliance_report":    "",
        }

    def test_phase2_chain_produces_report(self):
        _require_env(
            "AZURE_SEARCH_ENDPOINT", "AZURE_SEARCH_KEY",
            "NEO4J_URI", "NEO4J_PASSWORD", "NEO4J_DATABASE",
            "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_KEY",
        )

        from pipeline.compliance_retriever import ComplianceRetriever
        from agent.compliance_nodes import ComplianceNodes

        retriever = ComplianceRetriever()
        nodes     = ComplianceNodes(retriever=retriever)
        state     = self._blank_state(
            "Does the policy comply with GDPR Article 17 and HIPAA breach notification?"
        )

        # Node 0 — resolve which policy document(s) to audit
        state = nodes.doc_resolver_node(state)
        print(f"\n  Targets: {state['target_documents']} | clarification: {state['clarification_needed']}")
        if state["clarification_needed"]:
            pytest.skip(f"No unambiguous policy document to audit: {state['clarification_needed']}")

        # Node 1 — jurisdiction detection
        state = nodes.jurisdiction_detector_node(state)
        print(f"  Jurisdictions: {state['jurisdictions']}")
        assert len(state["jurisdictions"]) >= 1

        # Node 2 — KG retrieval (per document)
        state = nodes.kg_retriever_node(state)
        print(f"  Chunks: {len(state['kg_chunks'])} | Obligations: {len(state['obligations'])}")
        assert isinstance(state["kg_chunks"], list)
        assert isinstance(state["obligations"], list)

        # Node 3 — gap analysis (per document, untruncated policy text)
        state = nodes.gap_analyzer_node(state)
        print(f"  Gaps: {len(state['gaps'])}")
        assert isinstance(state["gaps"], list)

        # Node 4 — cross-regulation conflict detection (KG, value-level)
        state = nodes.conflict_detector_node(state)
        print(f"  Conflicts: {len(state['conflicts'])}")
        assert isinstance(state["conflicts"], list)

        # Node 5 — risk scoring -> 0-100 compliance
        state = nodes.risk_scorer_node(state)
        print(f"  Compliance score: {state['compliance_score']:.1f}/100")
        assert 0.0 <= state["compliance_score"] <= 100.0

        # Node 6 — remediation recommendations
        state = nodes.remediation_node(state)
        print(f"  Remediations: {len(state['remediations'])}")
        assert isinstance(state["remediations"], list)

        # Node 7 — render the final markdown report
        state = nodes.report_generator_node(state)
        assert len(state["compliance_report"]) > 100
        assert state["final_answer"] == state["compliance_report"]
        assert "# Privacy Compliance Audit Report" in state["compliance_report"]

        # Save to disk for manual inspection
        out = Path("data/compliance_report_e2e.md")
        out.parent.mkdir(exist_ok=True)
        out.write_text(state["compliance_report"], encoding="utf-8")
        print(f"  Report saved to {out}")

        retriever.close()
