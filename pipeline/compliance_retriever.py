"""
compliance_retriever.py

Replaces the weak keyword-loop in test_compliance_query().

Flow for every question:
  1. Azure AI Search semantic+hybrid search  → top-k relevant chunks
  2. Extract named entities from those chunks via LLM
  3. Query Neo4j for 1-hop triples around each entity
  4. Send chunks + triples to gpt-4o-mini for a grounded answer

Design decisions:
  - AzureSearch is initialised lazily so the object is cheap to import
    in Colab cells that just import the class definition.
  - All Neo4j calls are read-only and use parameterised queries.
  - Exponential-backoff retry wraps every external call (Azure, Neo4j, LLM).
  - Telemetry mirrors the pattern in app/core/telemetry.py.
"""

from __future__ import annotations

import os
import re
import time
import json
import logging
from typing import List, Dict, Tuple, Optional

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

# ── Lazy imports (avoid heavy deps at module level in notebooks) ───────────────
# langchain_community, neo4j, langchain_openai are imported inside methods
# so you can `from pipeline.compliance_retriever import ComplianceRetriever`
# without triggering import errors when optional deps are absent.


logger = logging.getLogger("pdf_rag_agent.compliance_retriever")

# ── Retry decorator shared by all external calls ──────────────────────────────

_RETRY = retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)

# ── Regulation keyword → jurisdiction mapping ─────────────────────────────────

JURISDICTION_KEYWORDS: Dict[str, List[str]] = {
    "GDPR":  ["gdpr", "general data protection", "european union", "eu", "dpa",
               "data subject", "controller", "processor", "article 17", "right to erasure",
               "legitimate interest", "supervisory authority"],
    "CCPA":  ["ccpa", "california consumer privacy", "california", "ccpr", "opt-out",
               "sale of personal information", "do not sell", "business purpose"],
    "HIPAA": ["hipaa", "health insurance portability", "protected health information",
               "phi", "covered entity", "business associate", "hitech", "minimum necessary"],
    "NIST":  ["nist", "national institute", "cybersecurity framework", "nist sp",
               "csf", "identify", "protect", "detect", "respond", "recover",
               "id.im", "pr.ac", "de.cm", "rs.rp", "rc.rp"],
}

# Financial exposure caps per regulation (for risk_scorer)
PENALTY_CAPS: Dict[str, str] = {
    "GDPR":  "€20M or 4% of global annual turnover (whichever higher)",
    "CCPA":  "$7,500 per intentional violation, $2,500 per unintentional",
    "HIPAA": "$100–$50,000 per violation category; $1.9M annual cap per category",
    "NIST":  "No statutory fines — contractual / federal-procurement risk",
}


# ═══════════════════════════════════════════════════════════════════════════════
# ComplianceRetriever
# ═══════════════════════════════════════════════════════════════════════════════

class ComplianceRetriever:
    """
    Hybrid retriever: Azure AI Search (semantic) + Neo4j KG triples.

    Usage
    -----
    retriever = ComplianceRetriever()
    result    = retriever.query("Does the policy address breach notification?")
    print(result["answer"])
    print(result["gaps"])          # List[str]
    print(result["jurisdictions"]) # e.g. ["GDPR", "HIPAA"]
    """

    # Max chunks to pull from Azure Search
    TOP_K: int = 10
    # Max entities to expand in Neo4j per query
    MAX_ENTITIES: int = 15
    # Max triples to include in LLM context (keeps prompt bounded)
    MAX_TRIPLES: int = 40

    def __init__(
        self,
        azure_search_endpoint: Optional[str] = None,
        azure_search_key: Optional[str] = None,
        index_name: Optional[str] = None,
        regulations_index: Optional[str] = None,
        policies_index: Optional[str] = None,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        neo4j_database: Optional[str] = None,
    ) -> None:
        # Azure Search
        self._endpoint  = azure_search_endpoint  or os.getenv("AZURE_SEARCH_ENDPOINT")
        self._key       = azure_search_key        or os.getenv("AZURE_SEARCH_KEY")
        # Two separate indexes: regulation text vs uploaded company policies.
        self._regulations_index = regulations_index or os.getenv(
            "AZURE_SEARCH_REGULATIONS_INDEX", "compliance-regulations"
        )
        self._policies_index = policies_index or os.getenv(
            "AZURE_SEARCH_POLICIES_INDEX", "compliance-policies"
        )
        # Legacy single-index name kept only for back-compat callers.
        self._index = index_name or os.getenv("AZURE_SEARCH_INDEX_NAME", "pdf-rag-index")

        # Neo4j
        self._neo4j_uri  = neo4j_uri      or os.getenv("NEO4J_URI")
        self._neo4j_user = neo4j_user     or os.getenv("NEO4J_USERNAME", "neo4j")
        self._neo4j_pass = neo4j_password or os.getenv("NEO4J_PASSWORD")
        self._neo4j_db   = neo4j_database or os.getenv("NEO4J_DATABASE", "neo4j")

        self._regulations_vs = None   # lazy
        self._policies_vs     = None   # lazy
        self._driver          = None   # lazy

        logger.info("ComplianceRetriever initialised (lazy — no connections yet).")

    # ── Lazy connection helpers ───────────────────────────────────────────────

    def _build_vs(self, index_name: str):
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores.azuresearch import AzureSearch

        from pipeline.search_schema import build_compliance_fields

        if not self._endpoint or not self._key:
            raise ValueError("AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_KEY must be set.")

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vs = AzureSearch(
            azure_search_endpoint=self._endpoint,
            azure_search_key=self._key,
            index_name=index_name,
            embedding_function=embeddings.embed_query,
            fields=build_compliance_fields(),
        )
        logger.info("Azure AI Search vectorstore connected (index=%s).", index_name)
        return vs

    def _get_policies_vs(self):
        if self._policies_vs is None:
            self._policies_vs = self._build_vs(self._policies_index)
        return self._policies_vs

    def _get_regulations_vs(self):
        if self._regulations_vs is None:
            self._regulations_vs = self._build_vs(self._regulations_index)
        return self._regulations_vs

    def _get_driver(self):
        if self._driver is not None:
            return self._driver

        from neo4j import GraphDatabase

        if not self._neo4j_uri or not self._neo4j_pass:
            raise ValueError("NEO4J_URI and NEO4J_PASSWORD must be set.")

        self._driver = GraphDatabase.driver(
            self._neo4j_uri,
            auth=(self._neo4j_user, self._neo4j_pass),
        )

        # self._driver = GraphDatabase.driver(uri, auth=(username, password))


        logger.info("Neo4j driver connected (uri=%s).", self._neo4j_uri)
        return self._driver

    def _get_llm(self):
        # from langchain_openai import AzureChatOpenAI
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
        temperature=0,
    )

        # return AzureChatOpenAI(
        #     azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
        #     api_key=os.getenv("AZURE_OPENAI_KEY"),
        #     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        #     api_version=os.getenv("OPENAI_API_VERSION", "2024-02-15-preview"),
        #     temperature=0,
        # )

    # ── Step 1: Azure AI Search ───────────────────────────────────────────────

    @_RETRY
    def _azure_search(
        self,
        question: str,
        k: int,
        index: str = "policies",
        filters: Optional[str] = None,
    ) -> List[Dict]:
        """
        Hybrid (vector + keyword) search against one index, optionally scoped
        by an OData $filter (e.g. "paper_id eq 'google_privacy_policy.pdf'").

        Parameters
        ----------
        index   : "policies" (default) | "regulations"
        filters : OData filter string, or None for no filter.

        Returns list of dicts with keys: page_content, metadata, score.
        """
        vs = self._get_regulations_vs() if index == "regulations" else self._get_policies_vs()

        search_kwargs = {"k": k}
        if filters:
            # langchain_community AzureSearch forwards `filters` to the SDK.
            search_kwargs["filters"] = filters

        docs_with_scores = vs.similarity_search_with_score(question, **search_kwargs)
        results = []
        for doc, score in docs_with_scores:
            results.append({
                "page_content": doc.page_content,
                "metadata":     doc.metadata,
                "score":        float(score),
            })
        logger.info(
            "Azure Search returned %d chunks (index=%s, filter=%s).",
            len(results), index, filters or "none",
        )
        return results

    @staticmethod
    def policy_filter(paper_id: str) -> str:
        """OData filter scoping a search to a single policy document."""
        safe = paper_id.replace("'", "''")
        return f"doc_type eq 'policy' and paper_id eq '{safe}'"

    @_RETRY
    def list_policy_documents(self) -> List[str]:
        """
        Return the distinct policy filenames currently in the policies index —
        the document registry used for filename-substring targeting.
        """
        vs = self._get_policies_vs()
        client = vs.client  # azure.search.documents.SearchClient
        seen: List[str] = []
        try:
            results = client.search(
                search_text="*",
                filter="doc_type eq 'policy'",
                facets=["paper_id,count:1000"],
                top=0,
            )
            facets = results.get_facets() or {}
            for f in facets.get("paper_id", []):
                if f.get("value"):
                    seen.append(f["value"])
        except Exception as exc:
            logger.warning("Facet listing failed (%s) — falling back to scan.", exc)
            try:
                results = client.search(
                    search_text="*",
                    filter="doc_type eq 'policy'",
                    select=["paper_id"],
                    top=1000,
                )
                for r in results:
                    pid = r.get("paper_id")
                    if pid and pid not in seen:
                        seen.append(pid)
            except Exception as exc2:
                logger.error("Policy document listing failed: %s", exc2)
        logger.info("Policy registry: %d documents.", len(seen))
        return seen

    @_RETRY
    def multi_hop(self, entities: List[str], hops: int = 2) -> List[Dict]:
        """
        Pull multi-hop obligation chains around the given entities from Neo4j.
        Thin wrapper that mirrors kg_builder.multi_hop_query but works against
        the live driver here. Returns {source, path_rels, target, distance}.
        """
        if not entities:
            return []

        driver = self._get_driver()
        cypher = """
        MATCH path = (n)-[*1..%d]-(m)
        WHERE any(term IN $terms
                  WHERE toLower(coalesce(n.id, n.name, '')) CONTAINS toLower(term))
        RETURN DISTINCT
            coalesce(n.id, n.name)              AS source,
            coalesce(m.id, m.name)              AS target,
            [r IN relationships(path) | type(r)] AS path_rels,
            length(path)                         AS distance
        ORDER BY distance
        LIMIT $limit
        """ % int(hops)

        out: List[Dict] = []
        with driver.session(database=self._neo4j_db) as session:
            try:
                result = session.run(cypher, terms=entities[:5], limit=self.MAX_TRIPLES)
                out = [dict(r) for r in result]
            except Exception as exc:
                logger.warning("multi_hop query failed: %s", exc)
        logger.info("multi_hop returned %d paths.", len(out))
        return out

    # ── Step 2: Extract named entities from chunks via LLM ───────────────────

    @_RETRY
    def _extract_entities(self, chunks: List[Dict], question: str) -> List[str]:
        """
        Ask the LLM to pull compliance-relevant entities from retrieved text.
        Returns deduplicated list, capped at MAX_ENTITIES.
        """
        combined_text = "\n\n---\n\n".join(c["page_content"] for c in chunks[:6])

        prompt = f"""You are a compliance analyst. Extract ALL named compliance entities
from the text below that are relevant to this question: "{question}"

Include: regulation names, article/section IDs, obligations, concepts (e.g. "consent",
"data breach", "right to erasure"), entity types (controller, processor), and timeframes.

Return ONLY a JSON array of strings. No explanation. Example:
["GDPR Article 17", "right to erasure", "controller", "30 days"]

TEXT:
{combined_text}

JSON array:"""

        llm = self._get_llm()
        response = llm.invoke(prompt)
        raw = response.content.strip()

        # Strip markdown fences if present
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()

        try:
            entities = json.loads(raw)
            if not isinstance(entities, list):
                raise ValueError("Not a list")
        except Exception:
            # Fallback: regex-extract quoted strings
            entities = re.findall(r'"([^"]{3,80})"', raw)

        entities = list(dict.fromkeys(e.strip() for e in entities if e.strip()))
        logger.info("Extracted %d entities from chunks.", len(entities))
        return entities[: self.MAX_ENTITIES]

    # ── Step 3: Neo4j triple retrieval ───────────────────────────────────────

    @_RETRY
    def _neo4j_triples(self, entities: List[str]) -> List[Dict]:
        """
        For each entity, pull 1-hop triples from Neo4j.
        Returns list of {source, relation, target, regulation} dicts.

        Results are gathered per entity-batch and then interleaved
        round-robin before the MAX_TRIPLES cap is applied. Without this, a
        single high-cardinality entity (e.g. "CCPA", which substring-matches
        hundreds of CCPA-derived nodes) fills the cap on its own and every
        other entity's — and regulation's — triples get discarded.
        """
        if not entities:
            return []

        driver = self._get_driver()

        cypher = """
        MATCH (n)-[r]->(m)
        WHERE any(term IN $terms
                  WHERE toLower(n.id) CONTAINS toLower(term)
                     OR toLower(n.name) CONTAINS toLower(term))
        OPTIONAL MATCH (d:Document)-[:MENTIONS]->(n)
        WITH n, r, m, collect(DISTINCT d.regulation) AS regs
        RETURN
            coalesce(n.name, n.id)  AS source,
            type(r)                  AS relation,
            coalesce(m.name, m.id)  AS target,
            coalesce(n.regulation, r.regulation, head(regs), '') AS regulation
        LIMIT $limit
        """

        # Batch into groups of 5 to avoid huge OR predicates
        batch_size = 5
        seen: set = set()
        batches: List[List[Dict]] = []

        with driver.session(database=self._neo4j_db) as session:
            for i in range(0, len(entities), batch_size):
                batch = entities[i : i + batch_size]
                bucket: List[Dict] = []
                try:
                    result = session.run(
                        cypher,
                        terms=batch,
                        limit=self.MAX_TRIPLES,
                    )
                    for record in result:
                        key = (record["source"], record["relation"], record["target"])
                        if key not in seen:
                            seen.add(key)
                            bucket.append(dict(record))
                except Exception as exc:
                    logger.warning("Neo4j batch query failed for %s: %s", batch, exc)
                batches.append(bucket)

        # Round-robin interleave across batches so no single batch dominates.
        triples: List[Dict] = []
        max_len = max((len(b) for b in batches), default=0)
        for idx in range(max_len):
            for bucket in batches:
                if idx < len(bucket):
                    triples.append(bucket[idx])
                    if len(triples) >= self.MAX_TRIPLES:
                        break
            if len(triples) >= self.MAX_TRIPLES:
                break

        logger.info("Neo4j returned %d unique triples (interleaved).", len(triples))
        return triples

    # ── Step 4: Detect jurisdictions ─────────────────────────────────────────

    @staticmethod
    def detect_jurisdictions(text: str, triples: List[Dict]) -> List[str]:
        """
        Rule-based jurisdiction detection from question + KG triples.
        Fast — no LLM call needed here.
        """
        combined = text.lower() + " ".join(
            f"{t.get('source','')} {t.get('target','')} {t.get('regulation','')}"
            for t in triples
        ).lower()

        detected = []
        for jurisdiction, keywords in JURISDICTION_KEYWORDS.items():
            if any(kw in combined for kw in keywords):
                detected.append(jurisdiction)

        # Always include at least the regulation mentioned in triples
        reg_from_triples = {
            t["regulation"].upper()
            for t in triples
            if t.get("regulation") and t["regulation"].upper() in JURISDICTION_KEYWORDS
        }
        detected = list(dict.fromkeys(detected + list(reg_from_triples)))

        logger.info("Detected jurisdictions: %s", detected)
        return detected if detected else ["UNKNOWN"]

    # ── Step 5: LLM synthesis ─────────────────────────────────────────────────

    @_RETRY
    def _synthesize(
        self,
        question: str,
        chunks: List[Dict],
        triples: List[Dict],
        jurisdictions: List[str],
    ) -> Dict:
        """
        Sends chunks + triples to the LLM and asks for:
          - answer
          - gaps (obligations the policy does NOT address)
          - recommendations
        Returns dict with those three keys plus raw_chunks / raw_triples counts.
        """
        chunks_text = "\n\n---\n\n".join(
            f"[Source: {c['metadata'].get('source','unknown')}]\n{c['page_content']}"
            for c in chunks
        )

        triples_text = "\n".join(
            f"({t['source']}) --[{t['relation']}]--> ({t['target']})"
            + (f"  [{t['regulation']}]" if t.get("regulation") else "")
            for t in triples
        ) or "No KG triples found."

        jurisdiction_str = ", ".join(jurisdictions) if jurisdictions else "Unknown"
        penalty_info = "\n".join(
            f"  {j}: {PENALTY_CAPS[j]}" for j in jurisdictions if j in PENALTY_CAPS
        ) or "  See regulation-specific documentation."

        prompt = f"""You are a senior privacy compliance analyst auditing a company policy
document against regulatory obligations.

APPLICABLE FRAMEWORKS: {jurisdiction_str}

PENALTY EXPOSURE:
{penalty_info}

=== RETRIEVED POLICY / REGULATORY CHUNKS ===
{chunks_text}

=== KNOWLEDGE GRAPH TRIPLES (regulatory obligations) ===
{triples_text}

=== COMPLIANCE QUESTION ===
{question}

Respond ONLY with a JSON object with exactly these keys:
{{
  "answer": "<direct answer to the question, citing specific articles where possible>",
  "gaps": ["<gap 1: obligation not addressed>", "<gap 2>", ...],
  "recommendations": ["<rec 1>", "<rec 2>", ...]
}}

Be specific. Reference article numbers. If the policy fully satisfies an obligation, say so.
Do NOT include markdown fences. Output only valid JSON."""

        llm = self._get_llm()
        response = llm.invoke(prompt)
        raw = response.content.strip()
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()

        try:
            result = json.loads(raw)
            if not all(k in result for k in ("answer", "gaps", "recommendations")):
                raise ValueError("Missing keys")
        except Exception as exc:
            logger.warning("LLM synthesis JSON parse failed: %s — using raw text.", exc)
            result = {
                "answer":          raw,
                "gaps":            [],
                "recommendations": [],
            }

        result["raw_chunks_used"]  = len(chunks)
        result["raw_triples_used"] = len(triples)
        result["jurisdictions"]    = jurisdictions
        return result

    # ── Public entrypoint ─────────────────────────────────────────────────────

    def query(self, question: str) -> Dict:
        """
        Full hybrid retrieval + compliance analysis for one question.

        Returns
        -------
        {
          "answer":           str,
          "gaps":             List[str],
          "recommendations":  List[str],
          "jurisdictions":    List[str],
          "raw_chunks_used":  int,
          "raw_triples_used": int,
          "chunks":           List[Dict],   # for downstream agents
          "triples":          List[Dict],   # for downstream agents
        }
        """
        t0 = time.time()
        logger.info("compliance_retriever.query started: %s", question[:120])

        try:
            chunks    = self._azure_search(question, k=self.TOP_K)
            print(f"Retrieved {len(chunks)} chunks from Azure Search.")
            entities  = self._extract_entities(chunks, question)

            triples   = self._neo4j_triples(entities)
            jurisdictions = self.detect_jurisdictions(question, triples)
            result    = self._synthesize(question, chunks, triples, jurisdictions)
        except Exception as exc:
            logger.error("compliance_retriever.query failed: %s", exc, exc_info=True)
            raise

        result["chunks"]  = chunks
        result["triples"] = triples
        result["latency_ms"] = round((time.time() - t0) * 1000)
        logger.info(
            "compliance_retriever.query done in %dms | gaps=%d | jurisdictions=%s",
            result["latency_ms"],
            len(result.get("gaps", [])),
            result.get("jurisdictions"),
        )
        return result

    # ── Endpoints support: regulation listing + graph export ──────────────────

    @_RETRY
    def list_regulation_documents(self) -> List[str]:
        """Distinct regulation names in the regulations index."""
        vs = self._get_regulations_vs()
        client = vs.client
        out: List[str] = []
        try:
            results = client.search(
                search_text="*",
                facets=["regulation,count:100"],
                top=0,
            )
            for f in (results.get_facets() or {}).get("regulation", []):
                if f.get("value"):
                    out.append(f["value"])
        except Exception as exc:
            logger.warning("Regulation listing failed: %s", exc)
        return out

    @_RETRY
    def get_regulation_chunks(self, regulation: Optional[str] = None, top: int = 50) -> List[Dict]:
        """Browse regulation chunks (optionally filtered to one regulation)."""
        vs = self._get_regulations_vs()
        client = vs.client
        flt = f"regulation eq '{regulation}'" if regulation else None
        out: List[Dict] = []
        try:
            kwargs = {"search_text": "*", "top": top, "select": ["content", "regulation", "paper_id", "chunk_id"]}
            if flt:
                kwargs["filter"] = flt
            for r in client.search(**kwargs):
                out.append({
                    "regulation": r.get("regulation", ""),
                    "paper_id":   r.get("paper_id", ""),
                    "chunk_id":   r.get("chunk_id", ""),
                    "content":    (r.get("content", "") or "")[:1200],
                })
        except Exception as exc:
            logger.warning("Regulation chunk browse failed: %s", exc)
        return out

    @_RETRY
    def get_graph(self, limit: int = 200, regulation: Optional[str] = None) -> Dict:
        """
        Export KG nodes/edges for the D3 explorer.
        Returns {"nodes": [{id, label}], "links": [{source, target, type, ...}]}.
        """
        driver = self._get_driver()
        nodes: Dict[str, Dict] = {}
        links: List[Dict] = []
        cypher = """
        MATCH (a)-[r]->(b)
        RETURN coalesce(a.id, a.name) AS source,
               coalesce(b.id, b.name) AS target,
               type(r)                 AS rel_type,
               r.description           AS description,
               r.value_a               AS value_a,
               r.value_b               AS value_b
        LIMIT $limit
        """
        with driver.session(database=self._neo4j_db) as session:
            for rec in session.run(cypher, limit=limit):
                s, t = rec["source"], rec["target"]
                if not s or not t:
                    continue
                nodes.setdefault(s, {"id": s, "label": s})
                nodes.setdefault(t, {"id": t, "label": t})
                links.append({
                    "source": s, "target": t, "type": rec["rel_type"],
                    "description": rec.get("description"),
                    "value_a": rec.get("value_a"), "value_b": rec.get("value_b"),
                })
        return {"nodes": list(nodes.values()), "links": links}

    @_RETRY
    def get_conflicts(self, limit: int = 50) -> List[Dict]:
        """
        Return all cross-regulation conflicts (CONFLICTS_WITH / STRICTER_THAN
        edges with a description) for the Dashboard's global conflicts view —
        independent of any single audit's jurisdiction scope.
        """
        driver = self._get_driver()
        with driver.session(database=self._neo4j_db) as session:
            result = session.run(
                """
                MATCH (a)-[r:CONFLICTS_WITH|STRICTER_THAN]->(b)
                WHERE r.description IS NOT NULL
                RETURN a.id           AS source,
                       b.id           AS target,
                       type(r)        AS rel_type,
                       r.description  AS description,
                       r.source_quote AS source_quote,
                       r.concept      AS concept,
                       r.value_a      AS value_a,
                       r.value_b      AS value_b,
                       r.unit         AS unit
                LIMIT $limit
                """,
                limit=limit,
            )
            return [dict(r) for r in result]

    def close(self) -> None:
        """Release Neo4j driver. Call when done (e.g. after a Colab cell run)."""
        if self._driver:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j driver closed.")