from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from ..interfaces import DocumentParser, VectorStore
from ..models import Chunk, Document, Paragraph


MAX_CHUNK_CHARS = 1600
TARGET_MERGED_CHUNK_CHARS = 800


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

        chunks = self._build_chunks(doc, paragraphs, source_url)

        self.store.upsert_documents([doc])
        self.store.upsert_chunks(chunks)
        return doc

    def _build_chunks(
        self,
        doc: Document,
        blocks: list[Paragraph],
        source_url: str | None,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        pending: list[Paragraph] = []

        def flush_pending() -> None:
            nonlocal pending
            if not pending:
                return
            chunks.append(self._chunk_from_blocks(doc, pending, source_url))
            pending = []

        for block in blocks:
            if block.block_type == "table":
                flush_pending()
                chunks.append(self._chunk_from_blocks(doc, [block], source_url))
                continue

            split_texts = _split_text_by_sentence(block.text, MAX_CHUNK_CHARS)
            if len(split_texts) > 1:
                flush_pending()
                for text in split_texts:
                    chunks.append(
                        self._chunk_from_blocks(
                            doc,
                            [block],
                            source_url,
                            override_text=text,
                        )
                    )
                continue

            if pending and _can_merge_blocks(pending, block):
                pending.append(block)
                if _joined_text_len(pending) >= TARGET_MERGED_CHUNK_CHARS:
                    flush_pending()
                continue

            flush_pending()
            pending = [block]

        flush_pending()
        return chunks

    def _chunk_from_blocks(
        self,
        doc: Document,
        blocks: list[Paragraph],
        source_url: str | None,
        override_text: str | None = None,
    ) -> Chunk:
        first = blocks[0]
        text = override_text if override_text is not None else "\n\n".join(
            block.text for block in blocks
        )
        para_ids = [block.para_id for block in blocks if block.para_id]
        coords = [block.coords for block in blocks if block.coords]
        return Chunk(
            chunk_id=str(uuid.uuid4()),
            doc_id=doc.doc_id,
            text=text,
            page_start=min(block.page_start for block in blocks),
            page_end=max(block.page_end for block in blocks),
            para_ids=para_ids,
            section_path=list(first.section_path),
            char_start=None,
            char_end=None,
            coords=coords,
            chunk_type=first.block_type,
            caption=first.caption if first.block_type == "table" else None,
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


def _joined_text_len(blocks: list[Paragraph]) -> int:
    return len("\n\n".join(block.text for block in blocks))


def _can_merge_blocks(pending: list[Paragraph], block: Paragraph) -> bool:
    first = pending[0]
    return (
        block.block_type == "paragraph"
        and first.block_type == "paragraph"
        and first.section_path == block.section_path
        and first.page_start == block.page_start
        and first.page_end == block.page_end
        and _joined_text_len([*pending, block]) <= MAX_CHUNK_CHARS
    )


def _split_text_by_sentence(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    sentences = [
        part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()
    ]
    if len(sentences) <= 1:
        return [
            text[i : i + max_chars].strip() for i in range(0, len(text), max_chars)
        ]

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(
                sentence[i : i + max_chars].strip()
                for i in range(0, len(sentence), max_chars)
            )
            continue

        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)
    return chunks
