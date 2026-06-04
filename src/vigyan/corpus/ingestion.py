from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from ..interfaces import DocumentParser, VectorStore
from ..models import Chunk, Document


def _hash_pdf(pdf_bytes: bytes) -> str:
    return hashlib.sha256(pdf_bytes).hexdigest()


@dataclass
class CorpusIngestor:
    """Ingest scientific document PDFs into a Corpus."""

    parser: DocumentParser
    store: VectorStore

    def ingest_pdf(
        self,
        pdf_bytes: bytes,
        meta: Document | None = None,
        source_url: str | None = None,
    ) -> Document:
        """Parse a PDF, create chunks, and index them in the Corpus."""
        if meta is None:
            meta = self.parser.extract_metadata(pdf_bytes)

        paragraphs, _ = self.parser.parse(pdf_bytes)
        pdf_sha = _hash_pdf(pdf_bytes)

        base = meta.model_dump(
            exclude={"pdf_sha256", "n_pages", "tei_xml", "created_at"},
            exclude_none=True,
        )
        doc = Document(
            **base,
            pdf_sha256=pdf_sha,
            n_pages=max((p.page_end for p in paragraphs), default=0),
            tei_xml=None,
            created_at=meta.created_at or datetime.now(timezone.utc),
        )

        self.store.create_or_open()

        chunks: list[Chunk] = []
        for p in paragraphs:
            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid4()),
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
                    embedding_model=self.store.model_name,
                    embedding_dims=self.store.dim,
                    embedding_ts=datetime.now(timezone.utc),
                    parser=type(self.parser).__name__,
                )
            )

        self.store.upsert_documents([doc])
        self.store.upsert_chunks(chunks)
        return doc
