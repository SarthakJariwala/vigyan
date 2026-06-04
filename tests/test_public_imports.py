from __future__ import annotations


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
