"""
compliance_nodes.py

Compliance-intelligence LangGraph nodes (the only flow in the app).

Node execution order:
  doc_resolver → jurisdiction_detector → kg_retriever → gap_analyzer
              → conflict_detector → risk_scorer → remediation → report_generator

Each node is a method on ComplianceNodes. Retrieval is scoped per policy
document (filename-substring targeting); cross-document queries loop the same
per-document pipeline and the report combines them.

All external calls (Azure Search, Neo4j, OpenAI) go through ComplianceRetriever,
which applies tenacity exponential-backoff retries. This layer adds per-node
timing telemetry and graceful degradation (a node failure sets a sentinel and
lets the graph continue rather than crashing the run).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

from agent.state import AgentState
from app.core.telemetry import QueryTelemetry
from pipeline.compliance_retriever import (
    ComplianceRetriever,
    JURISDICTION_KEYWORDS,
    PENALTY_CAPS,
)

logger = logging.getLogger("pdf_rag_agent.compliance_nodes")

# Matches "Article 17", "Art. 17(1)", "art 5a" etc. anywhere in a KG triple's
# source/relation/target text — used to recover an article reference for the
# gaps table even when it isn't the literal start of the source node name.
_ARTICLE_RE = re.compile(r"\b(?:article|art\.?)\s*(\d+[a-z]?(?:\(\d+\))?)", re.IGNORECASE)


# ── Risk weight per regulation ────────────────────────────────────────────────
REGULATION_WEIGHTS: Dict[str, float] = {
    "GDPR":  0.35,
    "HIPAA": 0.30,
    "CCPA":  0.20,
    "NIST":  0.15,
}

# Severity multipliers for gap risk scoring
SEVERITY_SCORES: Dict[str, float] = {
    "critical": 10.0,
    "high":     7.5,
    "medium":   5.0,
    "low":      2.5,
    "info":     1.0,
}

# Obligation types that are always high-severity if missing
CRITICAL_OBLIGATION_TYPES = {
    "breach_notification",
    "data_subject_rights",
    "lawful_basis",
    "data_retention",
    "access_control",
    "encryption",
    "dpo_appointment",
}

# Cross-document query signals
CROSS_DOC_SIGNALS = (
    "which polic", "which compan", "compare", "across", "all polic",
    "every polic", "each polic", "any polic", "these polic",
)

# Signals that the user wants to know IF/WHICH policies HAVE content on a
# topic — not a compliance gap audit. For these, the answer should be a
# yes/no coverage matrix, not a list of violations.
COVERAGE_SIGNALS = (
    "which polic", "which document", "which compan",
    "do any", "does any", "find policies", "find documents",
    "address", "mention", "cover", "discuss", "include",
    "have", "contain",
)

# Tokens to ignore when matching a query to a policy filename
_FILENAME_STOPWORDS = {
    "privacy", "policy", "policies", "latest", "pdf", "com", "the", "doc",
    "document", "data", "notice", "statement",
}

# Closed-vocabulary themes for grouping near-duplicate gaps/remediations.
GAP_THEMES = [
    "data_deletion_rights",
    "data_access_rights",
    "data_portability",
    "consent_mechanisms",
    "opt_out_mechanisms",
    "data_retention",
    "breach_notification",
    "third_party_sharing",
    "security_measures",
    "recordkeeping",
    "minor_consent",
    "regulatory_citations",
    "other",
]

# Gap-theme groups shown directly in the report; the rest go in a dropdown.
TOP_N_GAP_GROUPS = 3

GAP_THEME_LABELS: Dict[str, str] = {
    "data_deletion_rights": "Data Deletion & Erasure Rights",
    "data_access_rights":   "Data Access Rights",
    "data_portability":      "Data Portability",
    "consent_mechanisms":    "Consent Mechanisms",
    "opt_out_mechanisms":    "Opt-Out Mechanisms",
    "data_retention":        "Data Retention",
    "breach_notification":   "Breach Notification",
    "third_party_sharing":   "Third-Party Data Sharing",
    "security_measures":     "Security Measures",
    "recordkeeping":         "Recordkeeping & Documentation",
    "minor_consent":         "Minors' Privacy & Consent",
    "regulatory_citations":  "Regulatory Citations & References",
    "other":                 "Other Findings",
}


def risk_to_compliance(risk_0_10: float) -> float:
    """Convert internal risk (0-10, higher=worse) to compliance (0-100, higher=better)."""
    return round(max(0.0, min(100.0, 100.0 - risk_0_10 * 10.0)), 1)


# ═══════════════════════════════════════════════════════════════════════════════
# ComplianceNodes
# ═══════════════════════════════════════════════════════════════════════════════

class ComplianceNodes:
    """Compliance LangGraph nodes. Initialise once and pass to build_agent()."""

    def __init__(self, retriever: ComplianceRetriever | None = None) -> None:
        self._retriever = retriever   # injected or lazy-created
        self._telemetry = QueryTelemetry()

    # ── Lazy retriever ────────────────────────────────────────────────────────

    @property
    def retriever(self) -> ComplianceRetriever:
        if self._retriever is None:
            self._retriever = ComplianceRetriever()
        return self._retriever

    def _get_llm(self):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
            temperature=0,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # NODE 0 — doc_resolver  (entry point)
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _match_documents(query: str, registry: List[str]) -> List[str]:
        """Filename-substring match: return registry docs referenced by the query."""
        q = query.lower()
        matches: List[str] = []
        for doc in registry:
            stem = re.sub(r"\.[a-z0-9]+$", "", doc.lower())
            # direct: full stem appears in the query
            if stem and stem in q:
                matches.append(doc)
                continue
            # token: a significant filename token (e.g. "google") appears in query
            tokens = [t for t in re.split(r"[^a-z0-9]+", stem)
                      if len(t) >= 3 and t not in _FILENAME_STOPWORDS]
            if any(t in q for t in tokens):
                matches.append(doc)
        # de-dup, preserve order
        return list(dict.fromkeys(matches))

    def doc_resolver_node(self, state: AgentState) -> AgentState:
        """
        Resolve which policy document(s) the audit targets.

        Writes: state["target_documents"] OR state["clarification_needed"].
        """
        node_start = time.time()
        self._telemetry.start(state["query"])
        query           = state["query"]
        explicit_target = state.get("policy_document")

        try:
            registry = self.retriever.list_policy_documents()
        except Exception as exc:
            logger.error("[doc_resolver] registry fetch failed: %s", exc, exc_info=True)
            registry = []

        target_documents: List[str] = []
        clarification: str | None = None

        if explicit_target:
            # Caller named a document — validate against the registry.
            exact = [d for d in registry if d == explicit_target]
            fuzzy = self._match_documents(explicit_target, registry)
            if exact:
                target_documents = exact
            elif fuzzy:
                target_documents = fuzzy
            else:
                clarification = (
                    f"Requested document '{explicit_target}' is not in the index. "
                    f"Available: {', '.join(registry) or '(none ingested)'}"
                )
        else:
            matches = self._match_documents(query, registry)
            is_cross_doc = any(sig in query.lower() for sig in CROSS_DOC_SIGNALS)
            if matches:
                target_documents = matches
            elif is_cross_doc and registry:
                target_documents = list(registry)
            elif len(registry) == 1:
                target_documents = list(registry)
            elif not registry:
                clarification = "No policy documents have been ingested yet. Upload a policy to audit."
            else:
                clarification = (
                    "Multiple policies are available and the query doesn't name one. "
                    "Specify which document to audit (or ask a cross-document question). "
                    f"Available: {', '.join(registry)}"
                )

        # Classify intent: "coverage" = "which policies have/address X?"
        # "audit" = "does this policy comply with / what are the gaps in Y?"
        q_lower = query.lower()
        is_coverage = (
            any(sig in q_lower for sig in COVERAGE_SIGNALS)
            and not any(kw in q_lower for kw in ("comply", "complian", "gap", "violat", "miss", "fail", "audit"))
        )
        query_intent = "coverage" if is_coverage else "audit"

        state["target_documents"]    = target_documents
        state["clarification_needed"] = clarification
        state["per_doc_results"]     = {}
        state["query_intent"]        = query_intent

        elapsed = round((time.time() - node_start) * 1000)
        logger.info(
            "[doc_resolver] targets=%s | intent=%s | clarification=%s | %dms",
            target_documents, query_intent, bool(clarification), elapsed,
        )
        return state

    # ══════════════════════════════════════════════════════════════════════════
    # NODE 1 — jurisdiction_detector
    # ══════════════════════════════════════════════════════════════════════════

    def jurisdiction_detector_node(self, state: AgentState) -> AgentState:
        """
        Detect which regulatory frameworks apply (keyword scan + LLM confirmation).
        Writes state["jurisdictions"]; fallback = all frameworks.
        """
        node_start = time.time()
        query = state["query"]

        if state.get("clarification_needed"):
            state["jurisdictions"] = []
            return state

        try:
            keyword_hits = ComplianceRetriever.detect_jurisdictions(query, [])

            llm = self._get_llm()
            prompt = f"""Which privacy/security regulations are relevant to this compliance question?
Choose from: GDPR, CCPA, HIPAA, NIST, ALL, UNKNOWN.

Question: "{query}"

Rules:
- Return ONLY a JSON array of strings from the list above.
- If the question mentions "employees" or "HR data" include GDPR and CCPA.
- If health/medical data is mentioned include HIPAA.
- If the question is about technical controls or cybersecurity include NIST.
- If you are unsure, return ["GDPR", "CCPA", "HIPAA", "NIST"].
- Do NOT include markdown fences.

JSON array:"""

            response = llm.invoke(prompt)
            raw = response.content.strip().strip("`").replace("json", "").strip()
            llm_hits: List[str] = json.loads(raw) if raw.startswith("[") else keyword_hits

            allowed = set(JURISDICTION_KEYWORDS.keys())
            if "ALL" in llm_hits:
                jurisdictions = list(allowed)
            else:
                jurisdictions = list(dict.fromkeys(
                    j.upper() for j in llm_hits if j.upper() in allowed
                ))

            # Only use keyword_hits as a fallback — don't blindly append them
            # on top of a specific LLM result. When the user asks about one
            # regulation by name (e.g. "HIPAA"), the LLM returns that single
            # regulation; appending keyword_hits would add spurious frameworks.
            if not jurisdictions:
                jurisdictions = keyword_hits if keyword_hits else list(allowed)

        except Exception as exc:
            logger.warning("[jurisdiction_detector] failed (%s) — defaulting to all.", exc)
            jurisdictions = list(JURISDICTION_KEYWORDS.keys())

        state["jurisdictions"] = jurisdictions
        elapsed = round((time.time() - node_start) * 1000)
        logger.info("[jurisdiction_detector] %s | %dms", jurisdictions, elapsed)
        self._telemetry.node("node_jurisdiction_detector", jurisdictions=json.dumps(jurisdictions), latency_ms=elapsed)
        return state

    # ══════════════════════════════════════════════════════════════════════════
    # NODE 2 — kg_retriever  (loop per document)
    # ══════════════════════════════════════════════════════════════════════════

    def kg_retriever_node(self, state: AgentState) -> AgentState:
        """
        For each target document: scope a policies-index search by paper_id,
        extract entities, pull KG triples (1-hop + multi-hop), structure into
        obligations. Stores results per document in state["per_doc_results"].

        Documents are processed concurrently (each is an independent set of
        Azure Search / Neo4j / LLM calls), and within a document the
        independent Neo4j lookups (1-hop triples + multi-hop) also run
        concurrently.
        """
        node_start    = time.time()
        query         = state["query"]
        jurisdictions = state.get("jurisdictions", [])
        targets       = state.get("target_documents", [])

        if state.get("clarification_needed"):
            return state

        per_doc   = state.get("per_doc_results", {})
        retriever = self.retriever  # materialize before fanning out across threads

        def process(doc: str) -> Tuple[str, Dict]:
            try:
                chunks = retriever._azure_search(
                    query,
                    k=ComplianceRetriever.TOP_K,
                    index="policies",
                    filters=ComplianceRetriever.policy_filter(doc),
                )
                entities = retriever._extract_entities(chunks, query)

                # Triples and multi-hop both depend only on `entities` and
                # hit Neo4j independently — run them concurrently.
                with ThreadPoolExecutor(max_workers=2) as inner:
                    triples_future  = inner.submit(retriever._neo4j_triples, entities)
                    multihop_future = inner.submit(retriever.multi_hop, entities, 2)
                    triples = triples_future.result()
                    try:
                        hops = multihop_future.result()
                    except Exception as exc:
                        logger.warning("[kg_retriever] multi_hop failed for %s: %s", doc, exc)
                        hops = []

                obligations = self._triples_to_obligations(triples, jurisdictions)
            except Exception as exc:
                logger.error("[kg_retriever] failed for %s: %s", doc, exc, exc_info=True)
                chunks, triples, hops, obligations = [], [], [], []

            return doc, {
                "chunks":      chunks,
                "triples":     triples,
                "multi_hop":   hops,
                "obligations": obligations,
            }

        if targets:
            with ThreadPoolExecutor(max_workers=min(len(targets), 4)) as ex:
                for doc, data in ex.map(process, targets):
                    per_doc[doc] = data
                    logger.info(
                        "[kg_retriever] %s | chunks=%d triples=%d obligations=%d",
                        doc, len(data["chunks"]), len(data["triples"]), len(data["obligations"]),
                    )

        state["per_doc_results"] = per_doc

        # Aggregate (primary doc) for back-compat consumers.
        if targets:
            primary = per_doc.get(targets[0], {})
            state["kg_chunks"]  = primary.get("chunks", [])
            state["kg_triples"] = primary.get("triples", [])
            state["obligations"] = [
                ob for d in targets for ob in per_doc.get(d, {}).get("obligations", [])
            ]
        else:
            state["kg_chunks"], state["kg_triples"], state["obligations"] = [], [], []

        elapsed = round((time.time() - node_start) * 1000)
        logger.info("[kg_retriever] done | docs=%d | %dms", len(targets), elapsed)
        return state

    def _triples_to_obligations(
        self, triples: List[Dict], jurisdictions: List[str]
    ) -> List[Dict]:
        """Convert raw Neo4j triples into structured obligation objects."""
        TYPE_KEYWORDS = {
            "breach_notification": ["breach", "notif", "incident", "72 hour", "72-hour",
                                    "report incident", "supervisory authority"],
            "data_subject_rights": ["right to", "erasure", "deletion", "right of access",
                                    "rectification", "object", "restriction",
                                    "subject access", "forgotten"],
            "data_portability":    ["portability", "transmit", "machine-readable",
                                    "machine readable", "export"],
            "lawful_basis":        ["lawful basis", "legitimate interest", "legal basis",
                                    "contract", "legal obligation", "purpose limitation",
                                    "necessity"],
            "data_retention":      ["retention", "storage limitation", "delete", "purge",
                                    "retain", "destroy", "time limit"],
            "access_control":      ["access control", "least privilege", "authenticat",
                                    "authoris", "authoriz", "role", "permission", "credential"],
            "encryption":          ["encrypt", "cryptograph", "at rest", "in transit",
                                    "tls", "aes", "pseudonymi", "anonymi"],
            "dpo_appointment":     ["data protection officer", "dpo", "privacy officer",
                                    "representative"],
            "audit_logging":       ["audit", "logging", "monitor", "track",
                                    "record of processing", "recordkeeping",
                                    "documentation", "assessment"],
            "consent":             ["consent", "opt-in", "opt out", "opt-out", "withdrawal", "agree"],
            "third_party_sharing": ["third part", "share", "disclos", "transfer", "vendor",
                                    "sub-processor", "subprocessor", "processor agreement"],
            "minor_consent":       ["minor", "child", "parental", "under 13", "under 16",
                                    "age verif"],
        }

        obligations = []
        seen_texts: set = set()

        for i, triple in enumerate(triples):
            source   = triple.get("source", "")
            relation = triple.get("relation", "")
            target   = triple.get("target", "")
            reg      = (triple.get("regulation", "") or "").upper()

            if reg not in JURISDICTION_KEYWORDS and jurisdictions:
                reg = jurisdictions[0]

            combined = f"{source} {relation} {target}".strip()
            text = combined.lower()
            if text in seen_texts or len(text) < 15:
                continue
            seen_texts.add(text)

            # Relation labels are often UPPER_SNAKE_CASE (e.g. MUST_NOTIFY_WITHIN);
            # match against a space-separated form too so keywords like "notif"
            # line up with "must notify within".
            match_text = text.replace("_", " ")

            ob_type = "other"
            for t, keywords in TYPE_KEYWORDS.items():
                if any(kw in match_text for kw in keywords):
                    ob_type = t
                    break

            article_match = _ARTICLE_RE.search(combined.replace("_", " "))
            article = f"Article {article_match.group(1)}" if article_match else ""

            obligations.append({
                "id":         f"OBL-{i:04d}",
                "regulation": reg,
                "text":       f"{source} → [{relation}] → {target}",
                "article":    article,
                "type":       ob_type,
            })

        return obligations

    # ══════════════════════════════════════════════════════════════════════════
    # NODE 3 — gap_analyzer  (loop per document, no truncation)
    # ══════════════════════════════════════════════════════════════════════════

    def gap_analyzer_node(self, state: AgentState) -> AgentState:
        """For each document, compare its policy chunks against obligations → gaps."""
        node_start    = time.time()
        query         = state["query"]
        jurisdictions = state.get("jurisdictions", [])
        targets       = state.get("target_documents", [])

        if state.get("clarification_needed"):
            state["gaps"] = []
            return state

        per_doc = state.get("per_doc_results", {})
        all_gaps: List[Dict] = []

        docs_to_process: List[str] = []
        for doc in targets:
            data        = per_doc.get(doc, {})
            obligations = data.get("obligations", [])
            if not obligations:
                logger.warning("[gap_analyzer] %s — no obligations; skipping.", doc)
                data["gaps"] = []
                data["gap_groups"] = []
                per_doc[doc] = data
                continue
            docs_to_process.append(doc)

        def process(doc: str) -> Tuple[str, List[Dict]]:
            data        = per_doc[doc]
            chunks      = data.get("chunks", [])
            obligations = data.get("obligations", [])
            try:
                policy_text = "\n\n---\n\n".join(c["page_content"] for c in chunks) \
                    if chunks else "(No policy text retrieved)"
                logger.info("[gap_analyzer] %s | policy_text chars=%d", doc, len(policy_text))
                gaps = self._analyze_gaps_with_llm(
                    query, policy_text, obligations, jurisdictions
                )
            except Exception as exc:
                logger.error("[gap_analyzer] %s failed: %s", doc, exc, exc_info=True)
                gaps = [
                    {
                        "obligation_id": ob["id"],
                        "regulation":    ob["regulation"],
                        "description":   f"[FALLBACK] Could not verify: {ob['text'][:120]}",
                        "severity":      "high",
                        "article":       ob["article"],
                        "ob_type":       ob["type"],
                    }
                    for ob in obligations if ob["type"] in CRITICAL_OBLIGATION_TYPES
                ]

            for g in gaps:
                g["document"] = doc
            return doc, gaps

        # Each document's gap analysis is an independent set of LLM calls —
        # run documents concurrently.
        if docs_to_process:
            with ThreadPoolExecutor(max_workers=min(len(docs_to_process), 4)) as ex:
                for doc, gaps in ex.map(process, docs_to_process):
                    data = per_doc[doc]
                    data["gaps"] = gaps
                    data["gap_groups"] = self._group_gaps_by_theme(gaps)
                    per_doc[doc] = data
                    all_gaps.extend(gaps)

        state["per_doc_results"] = per_doc
        state["gaps"] = all_gaps

        elapsed = round((time.time() - node_start) * 1000)
        logger.info("[gap_analyzer] done | gaps=%d | %dms", len(all_gaps), elapsed)
        self._telemetry.node("node_gap_analyzer", gaps_found=len(all_gaps), latency_ms=elapsed)
        return state

    def _analyze_gaps_with_llm(
        self,
        query: str,
        policy_text: str,
        obligations: List[Dict],
        jurisdictions: List[str],
    ) -> List[Dict]:
        """Batch-checks obligations against the (full, untruncated) policy text.

        Each batch is an independent LLM call and they run concurrently.
        """
        BATCH = 20
        batches = [obligations[i : i + BATCH] for i in range(0, len(obligations), BATCH)]

        def process_batch(batch_idx: int, batch: List[Dict]) -> List[Dict]:
            llm = self._get_llm()
            obligations_json = json.dumps(
                [{"id": o["id"], "regulation": o["regulation"],
                  "text": o["text"], "type": o["type"]} for o in batch],
                indent=2,
            )

            prompt = f"""You are a privacy compliance auditor.

QUESTION FROM AUDITOR: {query}
APPLICABLE FRAMEWORKS: {', '.join(jurisdictions)}

POLICY DOCUMENT EXCERPTS:
{policy_text}

REGULATORY OBLIGATIONS TO CHECK (JSON):
{obligations_json}

For each obligation, determine if the policy:
  - "met"      — policy explicitly addresses this obligation
  - "partial"  — policy partially addresses it (missing details / edge cases)
  - "gap"      — policy is silent or contradicts this obligation

Return ONLY a JSON array. Include ONLY "partial" and "gap" items (skip "met").
Each item:
{{
  "obligation_id": "<id>",
  "regulation":    "<regulation>",
  "status":        "gap" | "partial",
  "description":   "<specific description of what is missing>",
  "severity":      "critical" | "high" | "medium" | "low",
  "article":       "<regulation article if known, else empty string>",
  "theme":         "<one of: {', '.join(GAP_THEMES)}>"
}}

Severity guide:
  critical — breach notification, access control, lawful basis (regulatory deadline risk)
  high     — data retention, encryption, DPO appointment
  medium   — logging, consent management, portability
  low      — documentation, minor procedural gaps

Theme guide — pick the single best-matching theme from the list above. Use
"other" only if none of the listed themes fit.

No markdown. Output only the JSON array."""

            response = llm.invoke(prompt)
            raw = response.content.strip()
            raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()

            try:
                batch_gaps = json.loads(raw)
                if not isinstance(batch_gaps, list):
                    raise ValueError("Expected list")
            except Exception as exc:
                logger.warning("[gap_analyzer] JSON parse error in batch %d: %s", batch_idx, exc)
                batch_gaps = []

            ob_type_map = {o["id"]: o["type"] for o in batch}
            for g in batch_gaps:
                g["ob_type"] = ob_type_map.get(g.get("obligation_id", ""), "other")
                if g.get("theme") not in GAP_THEMES:
                    g["theme"] = "other"

            return batch_gaps

        all_gaps: List[Dict] = []
        if batches:
            with ThreadPoolExecutor(max_workers=min(len(batches), 4)) as ex:
                futures = [ex.submit(process_batch, idx, batch) for idx, batch in enumerate(batches)]
                for fut in futures:
                    all_gaps.extend(fut.result())

        logger.info("[gap_analyzer] LLM found %d gaps across all batches.", len(all_gaps))
        return all_gaps

    @staticmethod
    def _group_gaps_by_theme(gaps: List[Dict]) -> List[Dict]:
        """Group gaps into ordered theme buckets for grouped report display."""
        SEV_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}

        buckets: Dict[str, List[Dict]] = {}
        for g in gaps:
            theme = g.get("theme") if g.get("theme") in GAP_THEMES else "other"
            buckets.setdefault(theme, []).append(g)

        groups: List[Dict] = []
        for theme in GAP_THEMES:
            items = buckets.get(theme)
            if not items:
                continue
            items = sorted(items, key=lambda g: SEV_ORDER.get(g.get("severity", "low"), 5))
            worst = min((g.get("severity", "low") for g in items),
                        key=lambda s: SEV_ORDER.get(s, 5))
            regulations = list(dict.fromkeys(
                g.get("regulation", "") for g in items if g.get("regulation")
            ))
            groups.append({
                "theme":       theme,
                "label":       GAP_THEME_LABELS.get(theme, theme.replace("_", " ").title()),
                "severity":    worst,
                "regulations": regulations,
                "gaps":        items,
            })

        groups.sort(key=lambda grp: SEV_ORDER.get(grp["severity"], 5))
        return groups

    # ══════════════════════════════════════════════════════════════════════════
    # NODE 4 — conflict_detector
    # ══════════════════════════════════════════════════════════════════════════

    def conflict_detector_node(self, state: AgentState) -> AgentState:
        """Pull value-level cross-regulation conflicts from the KG, scoped to
        the jurisdictions detected for this query."""
        node_start = time.time()
        if state.get("clarification_needed"):
            state["conflicts"] = []
            return state

        jurisdictions = state.get("jurisdictions", [])
        conflicts = [
            c for c in self._fetch_kg_conflicts()
            if self._conflict_in_scope(c, jurisdictions)
        ]
        state["conflicts"] = conflicts

        elapsed = round((time.time() - node_start) * 1000)
        logger.info("[conflict_detector] conflicts=%d | %dms", len(conflicts), elapsed)
        return state

    @staticmethod
    def _conflict_in_scope(conflict: Dict, jurisdictions: List[str]) -> bool:
        """Keep a conflict only if both regulations it names are in scope.

        Conflict source/target labels look like "GDPR breach notification" /
        "HIPAA breach notification". If neither label names a known
        regulation, keep it (can't determine scope, err on showing it)."""
        if not jurisdictions:
            return True
        text = f"{conflict.get('source', '')} {conflict.get('target', '')}".upper()
        mentioned = [reg for reg in JURISDICTION_KEYWORDS if reg in text]
        if not mentioned:
            return True
        return all(reg in jurisdictions for reg in mentioned)

    def _fetch_kg_conflicts(self) -> List[Dict]:
        """
        Read CONFLICTS_WITH / STRICTER_THAN edges from Neo4j, including the
        value-level fields. Reads a.id/b.id (LangChain stores the entity name
        in `id`, not `name`).
        """
        try:
            driver = self.retriever._get_driver()
            db     = self.retriever._neo4j_db
            with driver.session(database=db) as session:
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
                    LIMIT 50
                    """
                )
                return [dict(r) for r in result]
        except Exception as exc:
            logger.warning("[conflict_detector] conflict fetch failed: %s", exc)
            return []

    # ══════════════════════════════════════════════════════════════════════════
    # NODE 5 — risk_scorer  (0-100 compliance)
    # ══════════════════════════════════════════════════════════════════════════

    def risk_scorer_node(self, state: AgentState) -> AgentState:
        """Per-document risk → 0-100 compliance score, aggregated across documents."""
        node_start    = time.time()
        jurisdictions = state.get("jurisdictions", [])
        targets       = state.get("target_documents", [])

        if state.get("clarification_needed"):
            state["compliance_score"] = 0.0
            state["per_reg_compliance"] = {}
            state["risk_scores"] = {}
            state["overall_score"] = 0.0
            state["financial_exposure"] = ""
            return state

        per_doc = state.get("per_doc_results", {})

        for doc in targets:
            data        = per_doc.get(doc, {})
            obligations = data.get("obligations", [])
            gaps        = data.get("gaps", [])

            reg_gap_weight = {j: 0.0 for j in jurisdictions}
            reg_ob_count   = {j: 0   for j in jurisdictions}
            for ob in obligations:
                if ob["regulation"] in reg_ob_count:
                    reg_ob_count[ob["regulation"]] += 1
            for gap in gaps:
                reg = gap.get("regulation", "UNKNOWN")
                if reg in reg_gap_weight:
                    reg_gap_weight[reg] += SEVERITY_SCORES.get(gap.get("severity", "medium"), 5.0)

            risk_scores: Dict[str, float] = {}
            per_reg_compliance: Dict[str, float] = {}
            for reg in jurisdictions:
                ob_count = reg_ob_count.get(reg, 0)
                if ob_count == 0:
                    continue
                raw   = reg_gap_weight.get(reg, 0.0)
                score = min(raw / (ob_count * 10.0) * 10.0, 10.0)
                risk_scores[reg] = round(score, 2)
                per_reg_compliance[reg] = risk_to_compliance(score)

            scored_regs  = list(risk_scores.keys())
            total_weight = sum(REGULATION_WEIGHTS.get(r, 0.1) for r in scored_regs)
            overall_risk = 0.0
            if total_weight > 0 and scored_regs:
                for reg in scored_regs:
                    w = REGULATION_WEIGHTS.get(reg, 0.1) / total_weight
                    overall_risk += w * risk_scores[reg]
            overall_risk = round(overall_risk, 2)

            exposure_lines = []
            for reg in jurisdictions:
                score = risk_scores.get(reg)
                if score is None:
                    exposure_lines.append(f"- **{reg}**: no obligations retrieved — could not assess exposure")
                elif score > 0 and reg in PENALTY_CAPS:
                    exposure_lines.append(
                        f"- **{reg}** (compliance {per_reg_compliance[reg]:.0f}/100): {PENALTY_CAPS[reg]}"
                    )
            financial_exposure = "\n".join(exposure_lines) or "No material financial exposure identified."

            data["risk_scores"]        = risk_scores
            data["per_reg_compliance"] = per_reg_compliance
            data["overall_risk"]       = overall_risk
            data["compliance_score"]   = risk_to_compliance(overall_risk)
            data["financial_exposure"] = financial_exposure
            per_doc[doc] = data

        state["per_doc_results"] = per_doc

        # ── Aggregate across documents ────────────────────────────────────────
        scored = [(d, per_doc[d]) for d in targets if per_doc.get(d, {}).get("risk_scores")]
        if scored:
            n = len(scored)
            agg_risk = {}
            agg_comp = {}
            for reg in jurisdictions:
                vals = [data["risk_scores"][reg] for _, data in scored if reg in data["risk_scores"]]
                if vals:
                    agg_risk[reg] = round(sum(vals) / len(vals), 2)
                    agg_comp[reg] = risk_to_compliance(agg_risk[reg])
            overall_risk = round(sum(data["overall_risk"] for _, data in scored) / n, 2)
            state["risk_scores"]        = agg_risk
            state["per_reg_compliance"] = agg_comp
            state["overall_score"]      = overall_risk
            state["compliance_score"]   = risk_to_compliance(overall_risk)
            state["financial_exposure"] = scored[0][1]["financial_exposure"] if n == 1 else \
                "\n\n".join(f"**{doc}**\n{data['financial_exposure']}" for doc, data in scored)
        else:
            state["risk_scores"]        = {}
            state["per_reg_compliance"] = {}
            state["overall_score"]      = 0.0
            state["compliance_score"]   = 0.0
            state["financial_exposure"] = "No obligations retrieved — could not assess exposure."

        elapsed = round((time.time() - node_start) * 1000)
        logger.info("[risk_scorer] compliance=%.1f/100 | %dms",
                    state["compliance_score"], elapsed)
        self._telemetry.node(
            "node_risk_scorer",
            compliance_score=state["compliance_score"],
            per_reg_compliance=json.dumps(state["per_reg_compliance"]),
            latency_ms=elapsed,
        )
        return state

    # ══════════════════════════════════════════════════════════════════════════
    # NODE 6 — remediation
    # ══════════════════════════════════════════════════════════════════════════

    def remediation_node(self, state: AgentState) -> AgentState:
        """Generate one checklist-style remediation per gap-theme group, per document."""
        node_start = time.time()
        if state.get("clarification_needed"):
            state["remediations"] = []
            return state

        per_doc = state.get("per_doc_results", {})
        targets = state.get("target_documents", [])
        all_remediations: List[Dict] = []

        docs_to_process = [
            doc for doc in targets if per_doc.get(doc, {}).get("gap_groups")
        ]

        def process(doc: str) -> Tuple[str, Dict]:
            gap_groups = per_doc[doc]["gap_groups"]
            try:
                remediation_map = self._generate_remediations(state["query"], gap_groups)
            except Exception as exc:
                logger.error("[remediation] %s failed: %s", doc, exc, exc_info=True)
                remediation_map = {}
            return doc, remediation_map

        remediation_maps: Dict[str, Dict] = {}
        # Each document's remediation generation is an independent set of
        # LLM calls — run documents concurrently.
        if docs_to_process:
            with ThreadPoolExecutor(max_workers=min(len(docs_to_process), 4)) as ex:
                for doc, remediation_map in ex.map(process, docs_to_process):
                    remediation_maps[doc] = remediation_map

        for doc in docs_to_process:
            data            = per_doc[doc]
            gap_groups      = data["gap_groups"]
            remediation_map = remediation_maps.get(doc, {})

            for group in gap_groups:
                rec = remediation_map.get(group["theme"], {})
                group["remediation"] = {
                    "theme":          group["theme"],
                    "label":          group["label"],
                    "regulations":    group["regulations"],
                    "severity":       group["severity"],
                    "recommendation": rec.get("recommendation", ""),
                    "document":       doc,
                }
                all_remediations.append(group["remediation"])

            data["gap_groups"] = gap_groups
            per_doc[doc] = data

        state["per_doc_results"] = per_doc
        state["remediations"] = all_remediations

        elapsed = round((time.time() - node_start) * 1000)
        logger.info("[remediation] %d theme recommendations | %dms", len(all_remediations), elapsed)
        return state

    def _generate_remediations(self, query: str, gap_groups: List[Dict]) -> Dict[str, Dict]:
        """One concrete remediation per gap-theme group, keyed by theme.

        Each batch is an independent LLM call and they run concurrently.
        """
        BATCH = 10
        batches = [gap_groups[i : i + BATCH] for i in range(0, len(gap_groups), BATCH)]

        def process_batch(batch_idx: int, batch: List[Dict]) -> Dict[str, Dict]:
            llm = self._get_llm()
            groups_json = json.dumps(
                [{
                    "theme":        g["theme"],
                    "label":        g["label"],
                    "regulations":  g["regulations"],
                    "severity":     g["severity"],
                    "descriptions": [item.get("description", "") for item in g["gaps"]],
                } for g in batch],
                indent=2,
            )
            prompt = f"""You are a privacy compliance counsel. For each gap theme below,
write ONE concrete, plain-English remediation action the company can take to address
ALL the listed findings for that theme.

GAP THEMES (JSON):
{groups_json}

Return ONLY a JSON array. Each item:
{{
  "theme":          "<same theme>",
  "recommendation": "<1-3 sentence concrete action or clause to add/change, covering all findings in this theme>"
}}

No markdown. Output only the JSON array."""

            raw = llm.invoke(prompt).content.strip()
            raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
            batch_out: Dict[str, Dict] = {}
            try:
                items = json.loads(raw)
                if isinstance(items, list):
                    for item in items:
                        theme = item.get("theme")
                        if theme:
                            batch_out[theme] = item
            except Exception as exc:
                logger.warning("[remediation] parse error in batch %d: %s", batch_idx, exc)
            return batch_out

        out: Dict[str, Dict] = {}
        if batches:
            with ThreadPoolExecutor(max_workers=min(len(batches), 4)) as ex:
                futures = [ex.submit(process_batch, idx, batch) for idx, batch in enumerate(batches)]
                for fut in futures:
                    out.update(fut.result())

        return out

    # ══════════════════════════════════════════════════════════════════════════
    # NODE 7 — report_generator
    # ══════════════════════════════════════════════════════════════════════════

    def report_generator_node(self, state: AgentState) -> AgentState:
        """Render the final markdown report (handles clarification short-circuit)."""
        node_start = time.time()

        if state.get("clarification_needed"):
            report = (
                "# Clarification Needed\n\n"
                f"{state['clarification_needed']}\n"
            )
            state["compliance_report"] = report
            state["final_answer"]      = report
            self._telemetry.finish(state)
            return state

        report = self._render_report(state)
        state["compliance_report"] = report
        state["final_answer"]      = report

        elapsed = round((time.time() - node_start) * 1000)
        logger.info("[report_generator] rendered (%d chars) | %dms", len(report), elapsed)
        self._telemetry.finish(state)
        return state

    @staticmethod
    def _compliance_label(score_100: float) -> str:
        if score_100 >= 85: return "🟢 STRONG"
        if score_100 >= 65: return "🟡 MODERATE"
        if score_100 >= 40: return "🟠 WEAK"
        return "🔴 CRITICAL"

    @staticmethod
    def _render_gap_group(group: Dict) -> List[str]:
        """Render one theme-grouped set of gaps as a bullet with sub-bullets."""
        regs = ", ".join(group.get("regulations", [])) or "—"
        sev  = group.get("severity", "?").upper()
        lines = [f"- **{group.get('label', '')}** _({regs} · {sev})_"]
        for g in group.get("gaps", []):
            art = g.get("article", "")
            art_str = f" *(ref: {art})*" if art else ""
            lines.append(f"  - {g.get('description', '')}{art_str}")
        return lines

    @staticmethod
    def _render_remediation(remediation: Dict) -> str:
        """Render one theme's remediation as a checklist item."""
        regs = ", ".join(remediation.get("regulations", [])) or "—"
        recommendation = remediation.get("recommendation", "") or "(no recommendation generated)"
        return f"- [ ] **{remediation.get('label', '')}** ({regs}): {recommendation}"

    @staticmethod
    def _render_conflict_note(conflict: Dict) -> str:
        """Render one cross-regulation conflict as a plain-language note."""
        src, tgt = conflict.get("source", ""), conflict.get("target", "")
        va, vb   = conflict.get("value_a"), conflict.get("value_b")
        desc     = conflict.get("description", "")
        if va and vb:
            note = f"- **{src}** requires {va}, while **{tgt}** requires {vb}."
        else:
            note = f"- **{src}** vs **{tgt}**"
        if desc:
            note += f" {desc}"
        return note

    def _render_doc_section(self, doc: str, data: Dict, jurisdictions: List[str]) -> List[str]:
        """Render one document's grouped findings + remediation checklist."""
        gap_groups = data.get("gap_groups", [])
        per_reg    = data.get("per_reg_compliance", {})
        score      = data.get("compliance_score", 0.0)

        lines = [
            f"### 📄 {doc} — {score:.0f}/100 {self._compliance_label(score)}",
            "",
            "| Framework | Compliance | Rating |",
            "|-----------|-----------|--------|",
        ]
        for reg in jurisdictions:
            comp = per_reg.get(reg)
            if comp is None:
                lines.append(f"| {reg} | N/A | ⚪ NO DATA |")
            else:
                lines.append(f"| {reg} | {comp:.0f}/100 | {self._compliance_label(comp)} |")

        lines += ["", "**Findings:**", ""]
        if not gap_groups:
            lines.append("✅ No compliance gaps identified for the queried scope.")
        else:
            top, rest = gap_groups[:TOP_N_GAP_GROUPS], gap_groups[TOP_N_GAP_GROUPS:]
            for group in top:
                lines += self._render_gap_group(group)
            if rest:
                lines += ["", "<details>",
                          f"<summary>Show {len(rest)} more finding(s)</summary>", ""]
                for group in rest:
                    lines += self._render_gap_group(group)
                lines += ["", "</details>"]

        remediations = [
            g["remediation"] for g in gap_groups
            if g.get("remediation", {}).get("recommendation")
        ]
        if remediations:
            lines += ["", "**Remediation checklist:**", ""]
            for r in remediations:
                lines.append(self._render_remediation(r))

        lines.append("")
        return lines

    def _render_coverage_report(self, state: AgentState) -> str:
        """
        Render a coverage/discovery answer for queries like
        'Which policies address HIPAA breach notification timelines?'
        The question is not 'what gaps exist?' but 'which documents have
        content on this topic?' — so the answer is a yes/no matrix per policy.
        """
        query         = state["query"]
        jurisdictions = state.get("jurisdictions", [])
        targets       = state.get("target_documents", [])
        per_doc       = state.get("per_doc_results", {})

        lines = [
            "# Policy Coverage Report",
            "",
            f"**Question:** {query}",
            f"**Frameworks:** {', '.join(jurisdictions) or 'Auto-detected'}",
            "",
            "---",
            "",
            "## Which policies address this topic?",
            "",
        ]

        addressed, not_addressed = [], []
        for doc in targets:
            data     = per_doc.get(doc, {})
            chunks   = data.get("chunks", [])
            gaps     = data.get("gaps", [])
            gap_grps = data.get("gap_groups", [])

            # A document "addresses" the topic if it has relevant chunks retrieved
            # (i.e., the policy contains text about this subject) — regardless of
            # whether that coverage is complete. Gaps show HOW WELL it addresses it.
            has_content = len(chunks) > 0
            if has_content:
                addressed.append((doc, gaps, gap_grps))
            else:
                not_addressed.append(doc)

        if addressed:
            lines.append("### ✅ Policies that address this topic")
            lines.append("")
            for doc, gaps, gap_grps in addressed:
                critical = sum(1 for g in gaps if g.get("severity") == "critical")
                high     = sum(1 for g in gaps if g.get("severity") == "high")
                if not gaps:
                    coverage_note = "Fully addresses this requirement — no gaps identified."
                elif critical:
                    coverage_note = f"Has relevant language but with **{critical} critical** gap{'s' if critical > 1 else ''}{f' and {high} high' if high else ''} — coverage is incomplete."
                elif high:
                    coverage_note = f"Partially addresses this requirement — **{high} high-severity** gap{'s' if high > 1 else ''} identified."
                else:
                    coverage_note = f"Covers this topic with {len(gaps)} minor gap{'s' if len(gaps) != 1 else ''}."
                lines.append(f"**{doc}** — {coverage_note}")
                if gap_grps:
                    for grp in gap_grps[:2]:
                        lines.append(f"  - *Missing:* {grp.get('label', '')} ({', '.join(grp.get('regulations', []))})")
                lines.append("")

        if not_addressed:
            lines.append("### ❌ Policies that do NOT address this topic")
            lines.append("")
            for doc in not_addressed:
                lines.append(f"**{doc}** — No relevant content found in the indexed policy text.")
            lines.append("")

        if not addressed and not not_addressed:
            lines.append("No policy documents were found. Please ingest policies first.")

        lines += ["---", "", "*Report generated by Privacy Compliance Intelligence Platform.*"]
        return "\n".join(lines)

    def _render_report(self, state: AgentState) -> str:
        """Render the full compliance audit report (single- or multi-document)."""
        # Route to the coverage renderer for discovery-style questions
        if state.get("query_intent") == "coverage":
            return self._render_coverage_report(state)

        query         = state["query"]
        jurisdictions = state.get("jurisdictions", [])
        targets       = state.get("target_documents", [])
        per_doc       = state.get("per_doc_results", {})
        conflicts     = state.get("conflicts", [])
        overall_comp  = state.get("compliance_score", 0.0)

        multi = len(targets) > 1

        lines = [
            "# Privacy Compliance Audit Report",
            "",
            f"**Query:** {query}",
            f"**Documents Audited:** {', '.join(targets) or '(none)'}",
            f"**Frameworks Analysed:** {', '.join(jurisdictions)}",
            f"**Overall Compliance Score:** {overall_comp:.0f}/100  {self._compliance_label(overall_comp)}",
            "",
            "---",
            "",
        ]

        if multi:
            lines += ["## Document Comparison", "",
                      "| Document | Compliance | Rating |",
                      "|----------|-----------|--------|"]
            for doc in targets:
                s = per_doc.get(doc, {}).get("compliance_score", 0.0)
                lines.append(f"| {doc} | {s:.0f}/100 | {self._compliance_label(s)} |")
            lines += ["", "---", ""]

        lines += ["## Per-Document Findings", ""]
        for doc in targets:
            lines += self._render_doc_section(doc, per_doc.get(doc, {}), jurisdictions)

        # ── Cross-regulation notes ───────────────────────────────────────────
        lines += ["---", "", "## Cross-Regulation Notes", ""]
        if not conflicts:
            lines.append("No conflicting requirements found between the audited frameworks.")
        else:
            lines.append(
                f"The frameworks in this audit ({', '.join(jurisdictions)}) differ on "
                "the following points — make sure your policy satisfies the stricter "
                "of the two:"
            )
            lines.append("")
            for c in conflicts:
                lines.append(self._render_conflict_note(c))

        lines += ["", "---", "",
                  "*Report generated by Privacy Compliance Intelligence Platform.*"]

        return "\n".join(lines)


# ── Router function (used by graph.py) ────────────────────────────────────────

def compliance_router(state: AgentState) -> str:
    """After report_generator, the run is complete."""
    return "done"


def doc_resolver_router(state: AgentState) -> str:
    """Route to report_generator (clarification) or continue the audit."""
    return "clarify" if state.get("clarification_needed") else "audit"
