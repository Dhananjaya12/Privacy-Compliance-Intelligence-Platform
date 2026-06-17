"""
tests/test_1_imports.py

Layer 1 — Static checks. No credentials, no network, no GPU.
Verifies the entire import graph is consistent after the refactor.

Run:
    python -m pytest tests/test_1_imports.py -v
"""

import importlib
import sys


# ── Helpers ───────────────────────────────────────────────────────────────────

def _import(module_path: str):
    """Import a module and return it. Raises on failure."""
    return importlib.import_module(module_path)


# ── 1. No duplicate entry points ──────────────────────────────────────────────

def test_old_app_config_deleted():
    """app/config.py must be gone — only app/core/config.py should exist."""
    try:
        _import("app.config")
        raise AssertionError("app.config still exists — should have been deleted")
    except ModuleNotFoundError:
        pass  # expected


def test_old_app_schemas_deleted():
    """app/schemas.py must be gone — only app/models/schemas.py should exist."""
    try:
        _import("app.schemas")
        raise AssertionError("app.schemas still exists — should have been deleted")
    except ModuleNotFoundError:
        pass  # expected


def test_old_routers_deleted():
    """app/routers/ must be gone — only app/api/v1/endpoints/ should exist."""
    for mod in ("app.routers.query", "app.routers.ingest", "app.routers.health"):
        try:
            _import(mod)
            raise AssertionError(f"{mod} still exists — old routers should be deleted")
        except ModuleNotFoundError:
            pass  # expected


# ── 2. Canonical modules import cleanly ───────────────────────────────────────

def test_canonical_config_imports():
    mod = _import("app.core.config")
    assert hasattr(mod, "Settings")
    assert hasattr(mod, "get_settings")


def test_settings_has_all_required_fields():
    """Settings must include every Azure service field used in the codebase."""
    from app.core.config import Settings
    required = [
        "AZURE_SEARCH_ENDPOINT", "AZURE_SEARCH_KEY", "AZURE_SEARCH_INDEX_NAME",
        "AZURE_DOC_INTEL_ENDPOINT", "AZURE_DOC_INTEL_KEY",
        "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_KEY", "AZURE_OPENAI_DEPLOYMENT",
        "NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "NEO4J_DATABASE",
        "APPLICATIONINSIGHTS_CONNECTION_STRING",
        "TAVILY_API_KEY",
    ]
    fields = Settings.model_fields
    for f in required:
        assert f in fields, f"Settings missing field: {f}"


def test_canonical_schemas_import():
    mod = _import("app.models.schemas")
    for cls in ("QueryRequest", "QueryResponse", "SourceChunk",
                "IngestResponse", "HealthResponse", "ReadinessResponse"):
        assert hasattr(mod, cls), f"app.models.schemas missing: {cls}"


def test_canonical_router_imports():
    mod = _import("app.api.v1.router")
    assert hasattr(mod, "api_router")


def test_endpoint_modules_import():
    for mod_path in (
        "app.api.v1.endpoints.health",
        "app.api.v1.endpoints.query",
        "app.api.v1.endpoints.ingest",
    ):
        mod = _import(mod_path)
        assert hasattr(mod, "router"), f"{mod_path} missing 'router'"


def test_app_main_uses_canonical_imports():
    """app/main.py must import from app.core.config, not app.config."""
    import inspect
    import app.main as main_mod
    src = inspect.getsource(main_mod)
    assert "from app.core.config import" in src, \
        "app/main.py must import from app.core.config, not app.config"
    assert "from app.core.lifespan import" in src, \
        "app/main.py must use app.core.lifespan"
    assert "from app.api.v1.router import" in src, \
        "app/main.py must use app.api.v1.router"
    # Confirm old routers not referenced
    assert "app.routers" not in src, \
        "app/main.py still references deleted app.routers"


# ── 3. Agent layer imports ─────────────────────────────────────────────────────

def test_agent_state_imports():
    mod = _import("agent.state")
    assert hasattr(mod, "AgentState")


def test_legacy_rag_nodes_deleted():
    """agent/nodes.py and agent/websearch.py are removed in the compliance-only app."""
    for mod in ("agent.nodes", "agent.websearch"):
        try:
            _import(mod)
            raise AssertionError(f"{mod} still exists — should have been deleted")
        except ModuleNotFoundError:
            pass  # expected


def test_compliance_nodes_have_new_node_methods():
    """The split compliance nodes must all be present."""
    from agent.compliance_nodes import ComplianceNodes
    for m in (
        "doc_resolver_node", "jurisdiction_detector_node", "kg_retriever_node",
        "gap_analyzer_node", "conflict_detector_node", "risk_scorer_node",
        "remediation_node", "report_generator_node",
    ):
        assert hasattr(ComplianceNodes, m), f"ComplianceNodes missing {m}"


def test_compliance_nodes_import_re_at_top():
    """import re must be a top-level import, not buried inside a method."""
    import inspect
    import agent.compliance_nodes as cn
    src = inspect.getsource(cn)

    # Find position of 'import re' vs first def/class
    re_pos    = src.find("import re")
    class_pos = src.find("class ComplianceNodes")
    assert re_pos != -1, "'import re' not found in compliance_nodes.py"
    assert re_pos < class_pos, \
        "'import re' appears after class definition — must be at module top-level"


def test_compliance_nodes_exports():
    mod = _import("agent.compliance_nodes")
    assert hasattr(mod, "ComplianceNodes")
    assert hasattr(mod, "compliance_router")
    assert hasattr(mod, "doc_resolver_router")


def test_agent_graph_imports():
    mod = _import("agent.graph")
    assert hasattr(mod, "build_agent")


# ── 4. Pipeline layer imports ─────────────────────────────────────────────────

def test_pipeline_extractor_no_dead_code():
    """extractor.py must not contain the massive commented-out block."""
    import inspect
    import pipeline.extractor as ext
    src = inspect.getsource(ext)
    # The old file had >200 lines of comments starting with '# import json'
    # and multiple nested '# def _extract_with_azure' blocks
    commented_lines = [l for l in src.splitlines() if l.strip().startswith("#")]
    assert len(commented_lines) < 10, \
        f"extractor.py still has {len(commented_lines)} commented lines — dead code not removed"


def test_pipeline_extractor_exports():
    mod = _import("pipeline.extractor")
    assert hasattr(mod, "PDFTextExtractor")
    assert hasattr(mod, "_is_scanned")


def test_pipeline_no_faiss_or_chroma():
    """No pipeline file should reference FAISS, Chroma, or pgvector."""
    import inspect
    import pipeline.rag_pipeline as rp
    import pipeline.vectorstore as vs
    for mod, name in ((rp, "rag_pipeline"), (vs, "vectorstore")):
        src = inspect.getsource(mod).lower()
        for dead in ("faiss", "chroma", "pgvector"):
            assert dead not in src, \
                f"{name}.py still references '{dead}' — should be Azure Search only"


def test_rag_pipeline_exports():
    mod = _import("pipeline.rag_pipeline")
    assert hasattr(mod, "RAGPipeline")


def test_rag_service_no_duplicate_build_vectorstore():
    """ingest_pdfs must call build_vectorstore exactly once."""
    import inspect
    import app.services.rag_service as svc
    src = inspect.getsource(svc)
    count = src.count("build_vectorstore")
    assert count == 1, \
        f"rag_service.py calls build_vectorstore {count} times — expected exactly 1"


def test_chunker_bnb_indentation():
    """BitsAndBytesConfig must be instantiated inside _get_llm, not at module level."""
    import inspect
    import pipeline.chunker as ch
    src = inspect.getsource(ch)

    # If bnb_config is at module scope, it appears before 'class ChunkingManager'
    bnb_pos   = src.find("bnb_config = BitsAndBytesConfig")
    class_pos = src.find("class ChunkingManager")
    assert bnb_pos > class_pos, \
        "bnb_config instantiation appears before ChunkingManager — indentation bug"


def test_generator_no_hardcoded_model_check():
    """generator.py must not have hardcoded model-name if/elif dispatch."""
    import inspect
    import pipeline.generator as gen
    src = inspect.getsource(gen)
    assert 'if self.llm_model_name == "meta-llama' not in src, \
        "generator.py still has hardcoded model name check — should be removed"


# ── 5. Single config JSON ─────────────────────────────────────────────────────

def test_only_default_json_exists():
    """config/config.json must be deleted; config/default.json must exist."""
    from pathlib import Path
    assert not Path("config/config.json").exists(), \
        "config/config.json still exists — should be deleted"
    assert Path("config/default.json").exists(), \
        "config/default.json is missing"


def test_as_pipeline_config_structure():
    """as_pipeline_config() must return the keys all downstream code expects."""
    from unittest.mock import patch
    # Patch env so Settings can be constructed without real credentials
    with patch.dict("os.environ", {
        "AZURE_SEARCH_ENDPOINT": "https://fake.search.windows.net",
        "AZURE_SEARCH_KEY": "fakekey",
        "NEO4J_URI": "neo4j+s://fake.databases.neo4j.io",
        "NEO4J_PASSWORD": "fakepassword",
    }):
        from app.core.config import Settings
        s = Settings()
        cfg = s.as_pipeline_config()

    required_top_keys = {"paths", "chunking", "vectorstore", "azure_openai", "neo4j", "llm"}
    assert required_top_keys.issubset(cfg.keys()), \
        f"as_pipeline_config() missing keys: {required_top_keys - cfg.keys()}"

    # vectorstore config must NOT contain faiss/chroma/pgvector keys
    vs = cfg["vectorstore"]
    for dead_key in ("type", "pgvector_connection"):
        assert dead_key not in vs, \
            f"vectorstore config still has dead key '{dead_key}'"
