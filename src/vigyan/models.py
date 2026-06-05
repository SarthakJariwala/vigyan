from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class Document(BaseModel):
    """Domain model describing a scientific paper/document."""

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
    """A parsed text/table block with citation-relevant metadata."""

    text: str
    page_start: int
    page_end: int
    para_id: str | None = None
    coords: str | None = None
    section_path: list[str] = Field(default_factory=list)
    cited_ref_ids: list[str] = Field(default_factory=list)
    block_type: Literal["paragraph", "table"] = "paragraph"
    caption: str | None = None
    cells: list[list[str]] | None = None


class DocumentReference(BaseModel):
    """Bibliography entry cited by a source document."""

    reference_id: str
    source_doc_id: str
    ref_id: str
    label: str | None = None
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    venue: str | None = None
    year: int | None = None
    doi: str | None = None
    url: str | None = None
    raw_text: str = ""
    in_corpus: bool = False
    resolved_doc_id: str | None = None


class CitedReference(BaseModel):
    """Compact bibliography entry exposed on retrieval hits."""

    ref_id: str
    title: str | None = None
    authors: str | None = None
    doi: str | None = None
    year: int | None = None
    journal: str | None = None
    in_corpus: bool = False
    resolved_doc_id: str | None = None


class Chunk(BaseModel):
    """A chunk suitable for embedding and indexing.

    Note: The vector field is not stored here; embeddings are computed
    automatically by the vector store adapter.
    """

    chunk_id: str
    doc_id: str
    text: str
    page_start: int
    page_end: int
    para_ids: list[str] = Field(default_factory=list)
    section_path: list[str] = Field(default_factory=list)
    char_start: int | None = None
    char_end: int | None = None
    coords: list[str] = Field(default_factory=list)
    cited_ref_ids: list[str] = Field(default_factory=list)
    chunk_type: str = "paragraph"
    caption: str | None = None
    title: str
    authors: list[str]
    venue: str | None = None
    year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    source_url: str | None = None
    embedding_model: str
    embedding_dims: int
    embedding_ts: datetime
    parser: str


class QueryHit(BaseModel):
    """Result item for Corpus retrieval, including a citation string."""

    doc_id: str
    title: str
    year: int | None
    doi: str | None
    arxiv_id: str | None
    page_span: tuple[int, int]
    section_path: list[str] = Field(default_factory=list)
    chunk_type: str = "paragraph"
    caption: str | None = None
    cited_ref_ids: list[str] = Field(default_factory=list)
    cited_references: list[CitedReference] = Field(default_factory=list)
    text: str
    citation: str
    distance: float | None = None
