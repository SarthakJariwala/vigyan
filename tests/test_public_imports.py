from __future__ import annotations

import importlib

import pytest


def test_root_package_exposes_metadata_only() -> None:
    import vigyan

    assert vigyan.__version__ == "0.0.1"
    assert not hasattr(vigyan, "ingest_pdf")
    assert not hasattr(vigyan, "query")
    assert not hasattr(vigyan, "GrobidParser")
    assert not hasattr(vigyan, "LanceDBVectorStore")


def test_public_models_and_interfaces_import_from_named_modules() -> None:
    from vigyan.interfaces import DocumentParser, VectorStore
    from vigyan.models import Chunk, Document, Paragraph, QueryHit
    from vigyan.parsers import GrobidParser
    from vigyan.vectordb import LanceDBVectorStore

    assert Document.__name__ == "Document"
    assert Paragraph.__name__ == "Paragraph"
    assert Chunk.__name__ == "Chunk"
    assert QueryHit.__name__ == "QueryHit"
    assert DocumentParser.__name__ == "DocumentParser"
    assert VectorStore.__name__ == "VectorStore"
    assert GrobidParser.__name__ == "GrobidParser"
    assert LanceDBVectorStore.__name__ == "LanceDBVectorStore"


def test_agent_package_exports_only_reusable_research_types() -> None:
    import vigyan.agent

    assert vigyan.agent.__all__ == ["ResearchCapability", "ResearchRetriever"]
    assert {
        "ResearchAgentDeps",
        "agent",
        "build_deps",
        "build_deps_from_env",
        "run_research_query",
    }.isdisjoint(vars(vigyan.agent))


def test_legacy_research_agent_module_is_absent() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("vigyan.agent.research_agent")
