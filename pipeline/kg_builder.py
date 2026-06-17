"""
kg_builder.py

Automated Knowledge Graph Construction Pipeline for Regulatory Documents.

Pipeline:
  1. Load regulation PDFs from data/regulations/
  2. Split into individual articles using header detection
  3. LLMGraphTransformer → extract nodes + relationships per article
  4. Neo4j → persist full graph
  5. NetworkX → graph analytics (centrality, PageRank, communities)
  6. PyVis → export interactive HTML visualization

Usage:
  from pipeline.kg_builder import ComplianceKGBuilder
  builder = ComplianceKGBuilder()
  builder.build_from_pdfs("data/regulations/")
  stats = builder.get_analytics()
"""

from __future__ import annotations

import os
import re
import json
import asyncio
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# LangChain
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph

# Neo4j direct driver (for custom Cypher)
from neo4j import GraphDatabase

# Graph analytics
import networkx as nx
from networkx.algorithms.community import louvain_communities

# Visualization
from pyvis.network import Network


def _run_async(coro):
    """
    Run a coroutine to completion, whether or not an event loop is already
    running (e.g. inside Jupyter/Colab, which runs its own asyncio loop).
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    else:
        with concurrent.futures.ThreadPoolExecutor(1) as executor:
            return executor.submit(asyncio.run, coro).result()


# ── Article splitter ──────────────────────────────────────────────────────────

class ArticleSplitter:

    MIN_CHARS = 50

    PATTERNS = {
        "GDPR":  r'\n((?:##\s+)?Article\s+\d+[a-z]?\s+[A-Z][^\n]{2,80})\n',
        "CCPA":  r'^##\s+\*\*§\s*(7\d{3}\.[^\*\n]*)\*\*',
        "HIPAA": r'^##\s+\*\*((?:§\s*\d+\.\d+|Subpart\s+[A-Z]|PART\s+\d+)[^\*\n]*)\*\*',
        "NIST":  r'\*\*([A-Z]{2,3}\.[A-Z]{2,3}-P\d+)\s*:\s*\*\*',
        "HIPPA": r'^##\s+\*\*((?:§\s*\d+\.\d+|Subpart\s+[A-Z]|PART\s+\d+)[^\*\n]*)\*\*',
    }

    TOC_SKIP = {
        "HIPAA": r'##\s+\*\*Subpart\s+A',
        "HIPPA": r'##\s+\*\*Subpart\s+A',
    }

    CLEAN_TABLE = {"NIST"}

    def split(self, text: str, regulation_name: str) -> List[Dict]:
        reg = regulation_name.upper()
        pattern = self.PATTERNS.get(reg)

        if not pattern:
            print(f"  ⚠️ No pattern for {regulation_name} — size chunking")
            return self._chunk_by_size(text, regulation_name)

        if reg in self.TOC_SKIP:
            toc_match = re.search(self.TOC_SKIP[reg], text, re.IGNORECASE)
            if toc_match:
                print(f"  → Skipping {toc_match.start():,} chars of TOC")
                text = text[toc_match.start():]

        flags = re.MULTILINE if pattern.startswith('^') else 0
        matches = list(re.finditer(pattern, text, flags))
        print(f"  → {reg}: found {len(matches)} headers")

        if len(matches) < 3:
            print(f"  ⚠️ Too few headers — size chunking")
            return self._chunk_by_size(text, regulation_name)

        chunks = []
        for i, match in enumerate(matches):
            article_id = match.group(1) if match.lastindex else match.group()
            article_id = re.sub(r'^##\s*', '', article_id)
            article_id = re.sub(r'\s+', ' ', article_id).strip()

            start   = match.end()
            end     = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start:end].strip()

            if reg in self.CLEAN_TABLE:
                content = re.sub(r'\|', ' ', content)
                content = re.sub(r'<br>', '\n', content)
                content = re.sub(r'\n{3,}', '\n\n', content)
                content = content.strip()

            full = f"{article_id}\n\n{content}"
            if len(full) < self.MIN_CHARS:
                continue

            chunks.append({
                "regulation":    regulation_name,
                "article_id":    article_id,
                "heading_level": 2,
                "text":          full,
            })

        print(f"  → Split '{regulation_name}' into {len(chunks)} chunks")
        return chunks

    def _chunk_by_size(self, text: str, regulation_name: str,
                       chunk_size: int = 2000) -> List[Dict]:
        words = text.split()
        chunks: List[Dict] = []
        current: List[str] = []
        current_len = 0

        for word in words:
            current.append(word)
            current_len += len(word) + 1
            if current_len >= chunk_size:
                chunks.append({
                    "regulation":    regulation_name,
                    "article_id":    f"section_{len(chunks) + 1}",
                    "heading_level": 0,
                    "text":          " ".join(current),
                })
                current = []
                current_len = 0

        if current:
            chunks.append({
                "regulation":    regulation_name,
                "article_id":    f"section_{len(chunks) + 1}",
                "heading_level": 0,
                "text":          " ".join(current),
            })

        print(f"  → Size-chunked '{regulation_name}' into {len(chunks)} sections")
        return chunks


# ── LLM setup ─────────────────────────────────────────────────────────────────

def _get_llm():
    from langchain_openai import ChatOpenAI
    import os

    return ChatOpenAI(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
        temperature=0,
    )


# ── Core builder ──────────────────────────────────────────────────────────────

class ComplianceKGBuilder:

    ALLOWED_NODES = [
        "Regulation", "Article", "Obligation", "Right", "Entity",
        "Concept", "Penalty", "Timeframe",
    ]

    ALLOWED_RELATIONSHIPS = [
        "REFERENCES", "REQUIRES", "GRANTS", "CONFLICTS_WITH",
        "MAPS_TO", "STRICTER_THAN", "PART_OF", "DEFINES",
        "APPLIES_TO", "IMPOSES",
    ]

    def __init__(self) -> None:
        uri      = os.getenv("NEO4J_URI")
        username = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD")

        if not uri or not password:
            raise ValueError("NEO4J_URI and NEO4J_PASSWORD must be set in .env")

        self.database = os.getenv("NEO4J_DATABASE", "neo4j")
        self.neo4j_graph = Neo4jGraph(
            url=uri,
            username=username,
            password=password,
            database=self.database,
        )
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        print("✅ Neo4j connected")

        llm = _get_llm()
        self.transformer = LLMGraphTransformer(
            llm=llm,
            allowed_nodes=self.ALLOWED_NODES,
            allowed_relationships=self.ALLOWED_RELATIONSHIPS,
            ignore_tool_usage=True,
        )
        print("✅ LLMGraphTransformer initialized")

        self.splitter = ArticleSplitter()
        self.nx_graph: Optional[nx.DiGraph] = None

    # ── Step 0: Clear graph (clean rebuild) ───────────────────────────────────

    def clear_graph(self) -> None:
        """Delete ALL nodes and relationships. Used before a clean rebuild."""
        print("🔹 Clearing existing Neo4j graph (DETACH DELETE all nodes)...")
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("✅ Graph cleared")

    # ── Step 1: Load and split PDFs ───────────────────────────────────────────

    def load_pdfs(self, regulations_dir: str) -> List[Dict]:
        """
        Loads all PDFs from data/regulations/ using the existing
        Azure Document Intelligence extractor.
        Returns list of article chunks across all documents.
        """
        from pipeline.extractor import PDFTextExtractor

        path = Path(regulations_dir)
        pdf_files = list(path.glob("*.pdf"))

        if not pdf_files:
            raise FileNotFoundError(f"No PDFs found in {regulations_dir}")

        print(f"🔹 Found {len(pdf_files)} PDFs: {[f.name for f in pdf_files]}")

        extractor = PDFTextExtractor()
        all_chunks = []

        for pdf_file in pdf_files:
            regulation_name = pdf_file.stem.upper()
            cache_path = path / f"{pdf_file.stem}_extracted.jsonl"

            if cache_path.exists():
                print(f"🔹 Loading {regulation_name} from cache...")
                docs = []
                with open(cache_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            entry = json.loads(line)
                            docs.append(Document(
                                page_content=entry["text"],
                                metadata=entry["metadata"],
                            ))
            else:
                print(f"🔹 Extracting {regulation_name} via Azure Doc Intelligence...")
                docs = extractor.extract(
                    pdf_path=str(pdf_file),
                    output_path=cache_path,
                    existing_documents=[],
                )

            full_text = " ".join(d.page_content for d in docs)
            chunks = self.splitter.split(full_text, regulation_name)
            all_chunks.extend(chunks)
            print(f"  ✅ {regulation_name}: {len(chunks)} chunks")

            chunks_path = path / f"{pdf_file.stem}_chunks.jsonl"
            with open(chunks_path, "w", encoding="utf-8") as f:
                for i, chunk in enumerate(chunks):
                    f.write(json.dumps({
                        "chunk_index":   i,
                        "regulation":    chunk["regulation"],
                        "article_id":    chunk["article_id"],
                        "heading_level": chunk.get("heading_level", 0),
                        "char_count":    len(chunk["text"]),
                        "text_preview":  chunk["text"][:200],
                        "text":          chunk["text"],
                    }) + "\n")
            print(f"  💾 Chunks saved to {chunks_path}")

        print(f"✅ Total: {len(all_chunks)} article chunks across all regulations")

        return all_chunks

    # ── Step 2: LLMGraphTransformer extraction ────────────────────────────────

    async def _extract_graph_async(self, chunks: List[Dict]) -> List:
        documents = [
            Document(
                page_content=chunk["text"],
                metadata={
                    "regulation": chunk["regulation"],
                    "article_id": chunk["article_id"],
                    "source": f"{chunk['regulation']}:{chunk['article_id']}",
                },
            )
            for chunk in chunks
        ]

        print(f"🔹 Running LLMGraphTransformer on {len(documents)} articles (async)...")
        graph_documents = await self.transformer.aconvert_to_graph_documents(documents)
        print(f"✅ Extracted graph documents from {len(graph_documents)} articles")
        return graph_documents

    # ── Step 4: Store in Neo4j ────────────────────────────────────────────────

    def _store_llm_transformer_graph(self, graph_documents: List) -> None:
        print("🔹 Storing LLMGraphTransformer graph in Neo4j...")
        self.neo4j_graph.add_graph_documents(
            graph_documents,
            baseEntityLabel=True,
            include_source=True,
        )
        print("✅ LLMGraphTransformer graph stored")

    # ── Step 5: NetworkX analytics ────────────────────────────────────────────

    def _build_networkx_graph(self) -> nx.DiGraph:
        """Build a NetworkX graph from the Neo4j graph for analytics/visualization."""
        G = nx.MultiDiGraph()

        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (a)-[r]->(b)
                RETURN coalesce(a.id, a.name) AS source,
                       coalesce(b.id, b.name) AS target,
                       type(r)                 AS label
                """
            )
            for rec in result:
                s, t = rec["source"], rec["target"]
                if not s or not t:
                    continue
                G.add_node(s)
                G.add_node(t)
                G.add_edge(s, t, label=rec["label"])

        return G

    def get_analytics(self) -> Dict:
        if self.nx_graph is None or self.nx_graph.number_of_nodes() == 0:
            return {"error": "Graph not built yet. Run build_from_pdfs() first."}

        G  = self.nx_graph
        H  = nx.Graph(G)
        DG = nx.DiGraph(G)

        print("🔹 Running graph analytics...")

        degree_cent  = nx.degree_centrality(H)
        between_cent = nx.betweenness_centrality(H)
        pagerank     = nx.pagerank(DG) if DG.number_of_edges() > 0 else {}

        try:
            communities = louvain_communities(H, seed=42)
            community_map = {}
            for i, community in enumerate(communities):
                for node in community:
                    community_map[node] = i
        except Exception:
            communities = []
            community_map = {}

        def top_n(d, n=10):
            return sorted(d.items(), key=lambda x: -x[1])[:n]

        analytics = {
            "total_nodes": G.number_of_nodes(),
            "total_edges": G.number_of_edges(),
            "communities_detected": len(communities),
            "top_by_degree_centrality": top_n(degree_cent),
            "top_by_betweenness":       top_n(between_cent),
            "top_by_pagerank":          top_n(pagerank),
            "community_assignments":    community_map,
        }

        print(f"✅ Analytics complete: {G.number_of_nodes()} nodes, "
              f"{G.number_of_edges()} edges, {len(communities)} communities")

        return analytics

    # ── Step 6: Visualization ─────────────────────────────────────────────────

    def export_visualization(self, output_path: str = "data/kg_visualization.html") -> str:
        if self.nx_graph is None:
            raise ValueError("Graph not built. Run build_from_pdfs() first.")

        G = self.nx_graph
        analytics = self.get_analytics()
        pagerank = dict(analytics.get("top_by_pagerank", []))
        community_map = analytics.get("community_assignments", {})

        PALETTE = [
            "#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
            "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
        ]

        net = Network(
            height="750px", width="100%",
            directed=True,
            bgcolor="#1a1a2e",
            font_color="#ffffff",
            notebook=False,
            cdn_resources="in_line",
        )
        net.barnes_hut(gravity=-15000, spring_length=200, spring_strength=0.05)

        for node in G.nodes():
            size  = 12 + 60 * pagerank.get(node, 0.01)
            color = PALETTE[community_map.get(node, 0) % len(PALETTE)]
            net.add_node(
                node,
                label=node,
                color=color,
                size=size,
                title=f"PageRank: {pagerank.get(node, 0):.4f} | Community: {community_map.get(node, '?')}",
            )

        for s, t, data in G.edges(data=True):
            net.add_edge(s, t, label=data.get("label", ""), arrows="to", color="#888888")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        net.write_html(output_path, notebook=False, open_browser=False)
        print(f"✅ Visualization exported to {output_path}")
        return output_path

    # ── Step 7: Query interface ───────────────────────────────────────────────

    def query(self, question: str) -> str:
        from langchain_neo4j import GraphCypherQAChain

        chain = GraphCypherQAChain.from_llm(
            llm=_get_llm(),
            graph=self.neo4j_graph,
            verbose=False,
            allow_dangerous_requests=True,
        )
        result = chain.invoke({"query": question})
        return result.get("result", "No answer found in knowledge graph.")

    def multi_hop_query(self, start_entity: str, hops: int = 2) -> List[Dict]:
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH path = (start {name: $entity})-[*1..$hops]-(connected)
                RETURN DISTINCT
                    connected.name      AS name,
                    connected.id        AS id,
                    [r IN relationships(path) | type(r)]  AS rel_types,
                    length(path)                          AS distance
                ORDER BY distance
                """,
                entity=start_entity,
                hops=hops,
            )
            return [dict(r) for r in result]

    def get_conflicts(self) -> List[Dict]:
        """
        Return cross-regulation conflicts.

        Reads `a.id`/`b.id` (LangChain's `add_graph_documents` stores the
        canonical entity name in the `id` property, not `name`) and includes
        value-level fields written by the dedicated conflict pass.
        Matches both CONFLICTS_WITH and STRICTER_THAN edges.
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (a)-[r:CONFLICTS_WITH|STRICTER_THAN]->(b)
                WHERE r.description IS NOT NULL
                RETURN a.id          AS source,
                       b.id          AS target,
                       type(r)       AS rel_type,
                       r.description  AS description,
                       r.source_quote AS source_quote,
                       r.concept      AS concept,
                       r.value_a      AS value_a,
                       r.value_b      AS value_b,
                       r.unit         AS unit
                """
            )
            return [dict(r) for r in result]

    # ── Step 4b: Value-level cross-regulation conflict extraction ─────────────

    # Comparable obligations the LLM should look for across regulations.
    CONFLICT_CONCEPTS = [
        "data breach notification deadline",
        "right to erasure / deletion response deadline",
        "data subject access request response deadline",
        "consent age threshold for minors",
        "data retention limits",
        "opt-out / do-not-sell response deadline",
    ]

    def _extract_value_conflicts(self, chunks: List[Dict]) -> int:
        """
        Dedicated LLM pass that finds value-level conflicts BETWEEN regulations
        (e.g. "GDPR 72 hours vs HIPAA 60 days" for breach notification) and
        writes CONFLICTS_WITH / STRICTER_THAN edges with concrete properties.

        Implemented as direct Cypher (not via LLMGraphTransformer relationship
        properties, which are unreliable with ignore_tool_usage=True) so the
        conflict data is guaranteed to carry description/source_quote/values.

        Returns the number of conflict edges written.
        """
        print("🔹 Extracting value-level cross-regulation conflicts...")

        # Group chunk text per regulation (bounded to keep the prompt sane).
        per_reg: Dict[str, str] = {}
        for c in chunks:
            per_reg.setdefault(c["regulation"], "")
            if len(per_reg[c["regulation"]]) < 16000:
                per_reg[c["regulation"]] += "\n" + c["text"]

        regs_block = "\n\n".join(
            f"=== {reg} ===\n{text[:16000]}" for reg, text in per_reg.items()
        )

        prompt = f"""You are a privacy-law analyst. Compare these regulations and find
CONCRETE, VALUE-LEVEL conflicts where two regulations impose DIFFERENT requirements
for the SAME obligation (different deadlines, thresholds, or limits).

Focus on these concepts: {", ".join(self.CONFLICT_CONCEPTS)}.

For each conflict, output one object with:
  "source":       short label incl. regulation, e.g. "GDPR breach notification"
  "target":       short label incl. the other regulation, e.g. "HIPAA breach notification"
  "concept":      one of the concepts above
  "value_a":      the source's concrete value, e.g. "72 hours"
  "value_b":      the target's concrete value, e.g. "60 days"
  "unit":         the unit being compared, e.g. "time" / "age" / "days"
  "relationship": "STRICTER_THAN" if source is strictly tighter than target, else "CONFLICTS_WITH"
  "description":  one sentence explaining the conflict
  "source_quote": a short supporting quote from the text

Return ONLY a JSON array. No markdown. If no value-level conflicts exist, return [].

REGULATION TEXT:
{regs_block}

JSON array:"""

        llm = _get_llm()
        try:
            raw = llm.invoke(prompt).content.strip()
            raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
            conflicts = json.loads(raw)
            if not isinstance(conflicts, list):
                raise ValueError("expected a list")
        except Exception as exc:
            print(f"  ⚠️ value-conflict extraction failed: {exc}")
            return 0

        written = 0
        with self.driver.session(database=self.database) as session:
            for c in conflicts:
                source = (c.get("source") or "").strip()
                target = (c.get("target") or "").strip()
                if not source or not target:
                    continue
                rel = "STRICTER_THAN" if c.get("relationship") == "STRICTER_THAN" else "CONFLICTS_WITH"
                # Relationship type can't be parameterised — it's from a fixed set.
                session.run(
                    f"""
                    MERGE (a:__Entity__ {{id: $source}})
                      ON CREATE SET a.source = 'conflict-pass'
                    MERGE (b:__Entity__ {{id: $target}})
                      ON CREATE SET b.source = 'conflict-pass'
                    MERGE (a)-[r:{rel}]->(b)
                    SET r.concept      = $concept,
                        r.value_a      = $value_a,
                        r.value_b      = $value_b,
                        r.unit         = $unit,
                        r.description  = $description,
                        r.source_quote = $source_quote
                    """,
                    source=source,
                    target=target,
                    concept=c.get("concept", ""),
                    value_a=c.get("value_a", ""),
                    value_b=c.get("value_b", ""),
                    unit=c.get("unit", ""),
                    description=c.get("description", ""),
                    source_quote=c.get("source_quote", ""),
                )
                written += 1

        print(f"✅ Wrote {written} value-level conflict edges")
        return written

    # ── Main entry point ──────────────────────────────────────────────────────

    def build_from_pdfs(
        self,
        regulations_dir: str = "data/regulations/",
        clear_first: bool = True,
    ) -> Dict:
        print("\n" + "="*60)
        print("  COMPLIANCE KNOWLEDGE GRAPH BUILDER")
        print("="*60 + "\n")

        if clear_first:
            self.clear_graph()

        chunks = self.load_pdfs(regulations_dir)

        graph_documents = _run_async(self._extract_graph_async(chunks))

        self._store_llm_transformer_graph(graph_documents)

        # Value-level cross-regulation conflicts (concrete deadlines/thresholds).
        self._extract_value_conflicts(chunks)

        self.nx_graph = self._build_networkx_graph()

        viz_path = self.export_visualization()

        analytics = self.get_analytics()

        summary = {
            "articles_processed": len(chunks),
            "graph_documents": len(graph_documents),
            "nx_nodes": self.nx_graph.number_of_nodes(),
            "nx_edges": self.nx_graph.number_of_edges(),
            "communities": analytics.get("communities_detected", 0),
            "visualization": viz_path,
            "conflicts_in_graph": len(self.get_conflicts()),
        }

        print("\n" + "="*60)
        print("  BUILD COMPLETE")
        print(f"  Articles processed   : {summary['articles_processed']}")
        print(f"  Graph nodes          : {summary['nx_nodes']}")
        print(f"  Graph edges          : {summary['nx_edges']}")
        print(f"  Communities detected : {summary['communities']}")
        print(f"  Conflicts found      : {summary['conflicts_in_graph']}")
        print(f"  Visualization        : {summary['visualization']}")
        print("="*60 + "\n")

        return summary