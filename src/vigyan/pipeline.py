from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone

from .core.interfaces import DocumentParser, VectorStore
from .core.models import Chunk, Document, QueryHit


def _hash_pdf(pdf_bytes: bytes) -> str:
    return hashlib.sha256(pdf_bytes).hexdigest()


def ingest_pdf(
    pdf_bytes: bytes,
    meta: Document | None,
    parser: DocumentParser,
    store: VectorStore,
    source_url: str | None = None,
) -> Document:
    """Parse a PDF, create chunks, and index them.

    - Parses the PDF to paragraphs via `parser`.
    - Builds chunks (simple paragraph-based policy).
    - Upserts document and chunks into `store` (embeddings computed automatically).
    """

    # Determine metadata first (avoids double calls if parser can reuse work)
    if meta is None:
        meta = parser.extract_metadata(pdf_bytes)

    paragraphs, _ = parser.parse(pdf_bytes)
    pdf_sha = _hash_pdf(pdf_bytes)

    base = meta.model_dump(
        exclude={"pdf_sha256", "n_pages", "tei_xml", "created_at"},
        exclude_none=True,
    )
    doc = Document(
        **base,
        pdf_sha256=pdf_sha,
        n_pages=max((p.page_end for p in paragraphs), default=0),
        tei_xml=None,  # avoid storing large TEI by default; caller can persist it
        created_at=meta.created_at or datetime.now(timezone.utc),
    )

    # Build chunks (embeddings are computed automatically by the store)
    chunks: list[Chunk] = []
    for p in paragraphs:
        chunk_id = str(uuid.uuid4())
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                text=p.text,
                page_start=p.page_start,
                page_end=p.page_end,
                para_ids=[p.para_id] if p.para_id else [],
                section_path=[],
                char_start=None,
                char_end=None,
                coords=[p.coords] if p.coords else [],
                title=doc.title,
                authors=doc.authors,
                venue=doc.venue,
                year=doc.year,
                doi=doc.doi,
                arxiv_id=doc.arxiv_id,
                source_url=source_url or doc.url,
                embedding_model=store.model_name,
                embedding_dims=store.dim,
                embedding_ts=datetime.now(timezone.utc),
                parser=type(parser).__name__,
            )
        )

    # Persist
    store.create_or_open()
    store.upsert_documents([doc])
    store.upsert_chunks(chunks)
    return doc


def query(
    text: str,
    store: VectorStore,
    top_k: int = 8,
    filters: str | None = None,
) -> list[QueryHit]:
    store.create_or_open()
    return store.search(text, top_k=top_k, filters=filters)
