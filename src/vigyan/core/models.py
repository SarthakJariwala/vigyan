from datetime import datetime, timezone

from pydantic import BaseModel, Field


class Document(BaseModel):
    """Domain model describing a scientific paper/document.

    This is storage-agnostic and used across the pipeline.
    """

    doc_id: str
    title: str
    authors: list[str]
    venue: str | None = None
    year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    url: str | None = None
    pdf_sha256: str | None = None
    n_pages: int = 0
    tei_xml: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Paragraph(BaseModel):
    """A parsed paragraph with citation-relevant metadata."""

    text: str
    page_start: int
    page_end: int
    para_id: str | None = None
    coords: str | None = None


class Chunk(BaseModel):
    """A chunk suitable for embedding and indexing.

    Note: The vector field is not stored here; embeddings are computed
    automatically by the vector store (e.g., LanceDB).
    """

    chunk_id: str
    doc_id: str
    text: str
    page_start: int
    page_end: int
    para_ids: list[str] = []
    section_path: list[str] = []
    char_start: int | None = None
    char_end: int | None = None
    coords: list[str] = []
    # Denormalized doc fields for fast filters & citations
    title: str
    authors: list[str]
    venue: str | None = None
    year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    source_url: str | None = None
    # Versioning
    embedding_model: str
    embedding_dims: int
    embedding_ts: datetime
    parser: str


class QueryHit(BaseModel):
    """Result item for semantic search, including a citation string."""

    doc_id: str
    title: str
    year: int | None
    doi: str | None
    arxiv_id: str | None
    page_span: tuple[int, int]
    text: str
    citation: str
    distance: float | None = None
