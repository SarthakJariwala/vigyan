from __future__ import annotations

from typing import Any

from vigyan.corpus import CorpusRetriever
from vigyan.models import Chunk, Document, DocumentReference, QueryHit
from vigyan.vectordb.lancedb_store import LanceDBVectorStore


class FakeStore:
    model_name = "fake-embedding-model"
    dim = 3

    def __init__(self, hits: list[QueryHit]) -> None:
        self.hits = hits
        self.open_count = 0
        self.search_calls: list[dict[str, Any]] = []

    def create_or_open(self) -> None:
        self.open_count += 1

    def upsert_documents(self, docs: list[Document]) -> None:
        raise AssertionError("Corpus retrieval must not upsert documents")

    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        raise AssertionError("Corpus retrieval must not upsert chunks")

    def upsert_references(self, references: list[DocumentReference]) -> None:
        raise AssertionError("Corpus retrieval must not upsert references")

    def search(
        self,
        query: str,
        top_k: int = 8,
        filters: str | None = None,
    ) -> list[QueryHit]:
        self.search_calls.append(
            {"query": query, "top_k": top_k, "filters": filters}
        )
        return self.hits


def test_retrieve_opens_store_and_delegates_to_vector_store_search() -> None:
    hits = [
        QueryHit(
            doc_id="doc-1",
            title="A Test Paper",
            year=2026,
            doi="10.1234/example",
            arxiv_id=None,
            page_span=(2, 3),
            text="Relevant evidence.",
            citation="Tester A Test Paper (2026), pp. 2-3",
            distance=0.1,
        )
    ]
    store = FakeStore(hits)
    retriever = CorpusRetriever(store=store)

    result = retriever.retrieve(
        "protein folding",
        top_k=4,
        filters="year >= 2020",
    )

    assert result == hits
    assert store.open_count == 1
    assert store.search_calls == [
        {"query": "protein folding", "top_k": 4, "filters": "year >= 2020"}
    ]


def test_lancedb_format_hit_preserves_chunk_type_section_path_and_caption() -> None:
    hit = LanceDBVectorStore._format_hit(
        {
            "_distance": 0.2,
            "doc_id": "doc-1",
            "text": "[TABLE]\nCaption: Table 1 Device metrics.",
            "title": "A Test Paper",
            "authors": ["Tester A", "Tester B"],
            "year": 2026,
            "doi": "10.1234/example",
            "arxiv_id": None,
            "page_start": 5,
            "page_end": 5,
            "section_path": ["Results", "Device performance"],
            "chunk_type": "table",
            "caption": "Table 1 Device metrics.",
            "cited_ref_ids": ["b10"],
        },
        cited_references=[
            DocumentReference(
                reference_id="doc-1:b10",
                source_doc_id="doc-1",
                ref_id="b10",
                title="External primary source",
                authors=["Primary Author", "Second Author"],
                year=2025,
                doi="10.1234/primary?urlappend=%3Fref%3DPDF",
                raw_text="Primary Author, External primary source",
                in_corpus=False,
            )
        ],
    )

    assert hit.chunk_type == "table"
    assert hit.section_path == ["Results", "Device performance"]
    assert hit.caption == "Table 1 Device metrics."
    assert hit.cited_ref_ids == ["b10"]
    cited_reference = hit.cited_references[0]
    assert cited_reference.model_dump() == {
        "ref_id": "b10",
        "title": "External primary source",
        "authors": "Primary Author et al.",
        "doi": "10.1234/primary",
        "year": 2025,
        "journal": None,
        "in_corpus": False,
        "resolved_doc_id": None,
    }
    assert not hasattr(cited_reference, "reference_id")
    assert not hasattr(cited_reference, "source_doc_id")
    assert not hasattr(cited_reference, "raw_text")
    assert hit.citation == "Tester A et al. A Test Paper (2026), p. 5"
