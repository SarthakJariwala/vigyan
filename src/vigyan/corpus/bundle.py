from __future__ import annotations

from dataclasses import dataclass, field

from ..interfaces import DocumentParser, VectorStore
from .ingestion import CorpusIngestor
from .retrieval import CorpusRetriever


@dataclass(frozen=True)
class Corpus:
    """Compose PDF ingestion and retrieval over one parser and vector store."""

    parser: DocumentParser
    store: VectorStore
    ingestor: CorpusIngestor = field(init=False)
    retriever: CorpusRetriever = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "ingestor",
            CorpusIngestor(parser=self.parser, store=self.store),
        )
        object.__setattr__(self, "retriever", CorpusRetriever(store=self.store))
