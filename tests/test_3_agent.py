"""
tests/test_3_agent.py

Layer 3 — Compliance node logic and routing.
All external services (Azure, Neo4j, LLM) are mocked.

Run:
    python -m pytest tests/test_3_agent.py -v
"""

from unittest.mock import MagicMock, patch


def _make_nodes():
    from agent.compliance_nodes import ComplianceNodes
    return ComplianceNodes(retriever=MagicMock())


# ── doc_resolver: filename-substring matching ─────────────────────────────────

class TestDocMatching:

    def test_company_name_matches_filename(self):
        from agent.compliance_nodes import ComplianceNodes
        registry = ["google_privacy_policy_latest.pdf", "meta_privacy_policy.pdf"]
        matches = ComplianceNodes._match_documents("Does the Google policy cover erasure?", registry)
        assert matches == ["google_privacy_policy_latest.pdf"]

    def test_no_match_returns_empty(self):
        from agent.compliance_nodes import ComplianceNodes
        registry = ["google_privacy_policy_latest.pdf"]
        assert ComplianceNodes._match_documents("Does the policy cover encryption?", registry) == []

    def test_doc_resolver_router(self):
        from agent.compliance_nodes import doc_resolver_router
        assert doc_resolver_router({"clarification_needed": "ambiguous"}) == "clarify"
        assert doc_resolver_router({"clarification_needed": None}) == "audit"
        assert doc_resolver_router({}) == "audit"


# ── compliance_router ─────────────────────────────────────────────────────────

class TestComplianceRouter:

    def test_always_returns_done(self):
        from agent.compliance_nodes import compliance_router
        assert compliance_router({"compliance_score": 100.0}) == "done"
        assert compliance_router({"compliance_score": 0.0}) == "done"


# ── risk_scorer math (0-100 compliance) ───────────────────────────────────────

class TestRiskScorerMath:

    def _state(self, obligations, gaps):
        return {
            "query": "test",
            "jurisdictions": ["GDPR"],
            "target_documents": ["doc.pdf"],
            "per_doc_results": {"doc.pdf": {"obligations": obligations, "gaps": gaps}},
        }

    def _run(self, nodes, state):
        with patch.object(nodes, "_telemetry") as tel:
            tel.logger = MagicMock()
            return nodes.risk_scorer_node(state)

    def test_no_gaps_is_fully_compliant(self):
        nodes = _make_nodes()
        obs = [{"id": "OBL-0000", "regulation": "GDPR", "text": "x", "article": "", "type": "consent"}]
        result = self._run(nodes, self._state(obs, []))
        assert result["compliance_score"] == 100.0
        assert result["per_reg_compliance"]["GDPR"] == 100.0

    def test_critical_gap_is_zero_compliance(self):
        nodes = _make_nodes()
        obs = [{"id": "OBL-0000", "regulation": "GDPR", "text": "x", "article": "", "type": "breach_notification"}]
        gaps = [{"obligation_id": "OBL-0000", "regulation": "GDPR", "severity": "critical",
                 "description": "missing", "article": "", "ob_type": "breach_notification"}]
        result = self._run(nodes, self._state(obs, gaps))
        assert result["per_reg_compliance"]["GDPR"] == 0.0
        assert result["compliance_score"] == 0.0

    def test_compliance_bounded_0_100(self):
        nodes = _make_nodes()
        obs = [{"id": f"OBL-{i:04d}", "regulation": "GDPR", "text": "x", "article": "", "type": "other"} for i in range(2)]
        gaps = [{"obligation_id": f"OBL-{i:04d}", "regulation": "GDPR", "severity": "critical",
                 "description": "gap", "article": "", "ob_type": "other"} for i in range(10)]
        result = self._run(nodes, self._state(obs, gaps))
        assert 0.0 <= result["compliance_score"] <= 100.0


# ── _triples_to_obligations (unchanged logic) ─────────────────────────────────

class TestTriplesToObligations:

    def test_basic_triple_becomes_obligation(self):
        nodes = _make_nodes()
        triples = [{"source": "Article 33", "relation": "requires",
                    "target": "breach notification within 72 hours", "regulation": "GDPR"}]
        obs = nodes._triples_to_obligations(triples, ["GDPR"])
        assert len(obs) == 1
        assert obs[0]["regulation"] == "GDPR"
        assert obs[0]["type"] == "breach_notification"

    def test_deduplication(self):
        nodes = _make_nodes()
        triple = {"source": "Article 17", "relation": "grants",
                  "target": "right to erasure", "regulation": "GDPR"}
        assert len(nodes._triples_to_obligations([triple, triple], ["GDPR"])) == 1

    def test_short_triples_skipped(self):
        nodes = _make_nodes()
        triples = [{"source": "A", "relation": "B", "target": "C", "regulation": "GDPR"}]
        assert len(nodes._triples_to_obligations(triples, ["GDPR"])) == 0


# ── report rendering (state-based, 0-100) ─────────────────────────────────────

class TestRenderReport:

    def _state(self, gaps, conflicts, compliance):
        from agent.compliance_nodes import ComplianceNodes
        gap_groups = ComplianceNodes._group_gaps_by_theme(gaps)
        for group in gap_groups:
            group["remediation"] = {
                "theme":          group["theme"],
                "label":          group["label"],
                "regulations":    group["regulations"],
                "severity":       group["severity"],
                "recommendation": "Add a clause addressing this.",
                "document":       "google.pdf",
            }
        return {
            "query": "Test query",
            "jurisdictions": ["GDPR", "HIPAA"],
            "target_documents": ["google.pdf"],
            "per_doc_results": {"google.pdf": {
                "gaps": gaps,
                "gap_groups": gap_groups,
                "per_reg_compliance": {"GDPR": 70.0, "HIPAA": 30.0},
                "compliance_score": compliance,
            }},
            "conflicts": conflicts,
            "compliance_score": compliance,
        }

    def test_report_contains_required_sections(self):
        nodes = _make_nodes()
        gaps = [{"obligation_id": "OBL-0000", "regulation": "GDPR", "severity": "high",
                 "description": "Missing DPO appointment", "article": "Article 37",
                 "ob_type": "dpo_appointment", "theme": "recordkeeping"}]
        report = nodes._render_report(self._state(gaps, [], 55.0))
        assert "# Privacy Compliance Audit Report" in report
        assert "Overall Compliance Score" in report
        assert "## Per-Document Findings" in report
        assert "## Cross-Regulation Notes" in report
        assert "Missing DPO appointment" in report
        assert "55/100" in report
        assert "Remediation checklist" in report

    def test_value_level_conflict_rendered(self):
        nodes = _make_nodes()
        conflicts = [{"source": "GDPR breach notification", "target": "HIPAA breach notification",
                      "rel_type": "STRICTER_THAN", "value_a": "72 hours", "value_b": "60 days",
                      "description": "GDPR is tighter"}]
        report = nodes._render_report(self._state([], conflicts, 80.0))
        assert "72 hours" in report and "60 days" in report

    def test_no_conflicts_message(self):
        nodes = _make_nodes()
        report = nodes._render_report(self._state([], [], 90.0))
        assert "No conflicting requirements found between the audited frameworks." in report
