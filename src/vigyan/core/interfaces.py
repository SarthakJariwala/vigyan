from typing import Protocol, runtime_checkable

from .models import Chunk, Document, Paragraph, QueryHit


@runtime_checkable
class Embedder(Protocol):
    """Abstract embedding provider."""

    @property
    def model_name(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class VectorStore(Protocol):
    """Abstract vector store with document/chunk upsert and search."""

    def create_or_open(self) -> None: ...

    def upsert_documents(self, docs: list[Document]) -> None: ...

    def upsert_chunks(self, chunks: list[Chunk]) -> None: ...

    def search(
        self, query: str, top_k: int = 8, filters: str | None = None
    ) -> list[QueryHit]: ...


@runtime_checkable
class DocumentParser(Protocol):
    """Abstract document parser that returns paragraphs and optional raw representation."""

    def parse(self, pdf_bytes: bytes) -> tuple[list[Paragraph], str | None]: ...

    def extract_metadata(self, pdf_bytes: bytes) -> Document: ...
