from typing import TypedDict, List, Dict, Optional


class AgentState(TypedDict, total=False):
    # ── Core ──────────────────────────────────────────────────────────────────
    query:            str
    final_answer:     str

    # ── Document targeting (doc_resolver) ─────────────────────────────────────
    policy_document:     Optional[str]   # caller-supplied target filename (optional)
    target_documents:    List[str]       # resolved policy filenames to audit
    clarification_needed: Optional[str]  # set when query is ambiguous; short-circuits

    # ── jurisdiction_detector output ──────────────────────────────────────────
    jurisdictions: List[str]             # e.g. ["GDPR", "HIPAA"]

    # ── kg_retriever output ───────────────────────────────────────────────────
    kg_chunks:    List[Dict]             # aggregate Azure Search docs (primary doc)
    kg_triples:   List[Dict]             # Neo4j triples
    obligations:  List[Dict]             # structured obligations {id, regulation, text, article, type}

    # Per-document results, keyed by filename. Each value holds that document's
    # chunks / obligations / gaps / risk_scores / per_reg_compliance /
    # compliance_score / financial_exposure / remediations.
    per_doc_results: Dict[str, Dict]

    # ── gap_analyzer output ───────────────────────────────────────────────────
    gaps:         List[Dict]             # aggregate {obligation_id, regulation, description, severity}

    # ── conflict_detector output ──────────────────────────────────────────────
    conflicts:    List[Dict]             # value-level cross-regulation conflicts from KG

    # ── risk_scorer output (0-100 compliance; higher = better) ────────────────
    risk_scores:        Dict             # {regulation: risk 0-10} (internal)
    per_reg_compliance: Dict             # {regulation: compliance 0-100}
    overall_score:      float            # internal risk 0-10
    compliance_score:   float            # headline 0-100 (100 = fully compliant)
    financial_exposure: str              # human-readable penalty summary

    # ── remediation output ────────────────────────────────────────────────────
    remediations: List[Dict]            # {obligation_id, regulation, recommendation, severity}

    # ── query intent (doc_resolver) ──────────────────────────────────────────
    # "audit"    — user wants a gap/compliance analysis ("does X comply with Y?")
    # "coverage" — user wants to know which policies mention/address a topic
    #              ("which policies address X?", "do any policies cover Y?")
    query_intent: str

    # ── report_generator output ───────────────────────────────────────────────
    compliance_report: str              # final markdown report rendered to final_answer
