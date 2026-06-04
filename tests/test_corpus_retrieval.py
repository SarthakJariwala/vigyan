from __future__ import annotations

from typing import Any

from vigyan.corpus import CorpusRetriever
from vigyan.models import Chunk, Document, QueryHit


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
