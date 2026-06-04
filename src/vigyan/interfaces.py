from typing import Protocol, runtime_checkable

from .models import Chunk, Document, Paragraph, QueryHit


@runtime_checkable
class VectorStore(Protocol):
    """Vector store adapter seam for document/chunk upsert and search."""

    @property
    def model_name(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def create_or_open(self) -> None: ...

    def upsert_documents(self, docs: list[Document]) -> None: ...

    def upsert_chunks(self, chunks: list[Chunk]) -> None: ...

    def search(
        self, query: str, top_k: int = 8, filters: str | None = None
    ) -> list[QueryHit]: ...


@runtime_checkable
class DocumentParser(Protocol):
    """Document parser adapter seam for PDF parsing and metadata extraction."""

    def parse(self, pdf_bytes: bytes) -> tuple[list[Paragraph], str | None]: ...

    def extract_metadata(self, pdf_bytes: bytes) -> Document: ...
