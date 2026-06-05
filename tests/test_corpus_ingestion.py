from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from vigyan.corpus import CorpusIngestor
from vigyan.models import Chunk, Document, DocumentReference, Paragraph


class FakeParser:
    def __init__(self) -> None:
        self.metadata_calls: list[bytes] = []
        self.parse_calls: list[bytes] = []

    def extract_metadata(self, pdf_bytes: bytes) -> Document:
        self.metadata_calls.append(pdf_bytes)
        return Document(
            doc_id="doc-from-parser",
            title="Parsed Paper",
            authors=["Ada Lovelace", "Grace Hopper"],
            venue="Journal of Tests",
            year=2026,
            doi="10.1234/parser",
            url="https://example.test/parser",
        )

    def parse(self, pdf_bytes: bytes) -> tuple[list[Paragraph], str | None]:
        self.parse_calls.append(pdf_bytes)
        return (
            [
                Paragraph(
                    text="First paragraph.",
                    page_start=1,
                    page_end=1,
                    para_id="p1",
                    coords="1,10,10,50,50",
                ),
                Paragraph(
                    text="Second paragraph.",
                    page_start=2,
                    page_end=3,
                    para_id=None,
                    coords=None,
                ),
            ],
            "<TEI />",
        )


class FakeStore:
    model_name = "fake-embedding-model"

    def __init__(self) -> None:
        self.open_count = 0
        self.documents: list[Document] = []
        self.chunks: list[Chunk] = []
        self.references: list[DocumentReference] = []

    @property
    def dim(self) -> int:
        assert self.open_count > 0, "CorpusIngestor must open the store before reading dim"
        return 3

    def create_or_open(self) -> None:
        self.open_count += 1

    def upsert_documents(self, docs: list[Document]) -> None:
        self.documents.extend(docs)

    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        self.chunks.extend(chunks)

    def upsert_references(self, references: list[DocumentReference]) -> None:
        self.references.extend(references)

    def search(
        self,
        query: str,
        top_k: int = 8,
        filters: str | None = None,
    ) -> list[Any]:
        raise AssertionError("Corpus ingestion must not search")


def test_ingest_pdf_extracts_metadata_opens_store_builds_chunks_and_persists() -> None:
    parser = FakeParser()
    store = FakeStore()
    ingestor = CorpusIngestor(parser=parser, store=store)

    doc = ingestor.ingest_pdf(b"pdf bytes", meta=None, source_url="https://source.test/paper.pdf")

    assert parser.metadata_calls == [b"pdf bytes"]
    assert parser.parse_calls == [b"pdf bytes"]
    assert store.open_count == 1
    assert store.documents == [doc]
    assert doc.doc_id == "doc-from-parser"
    assert doc.pdf_sha256 == hashlib.sha256(b"pdf bytes").hexdigest()
    assert doc.n_pages == 3
    assert doc.tei_xml is None

    assert len(store.chunks) == 2
    first, second = store.chunks
    assert first.doc_id == doc.doc_id
    assert first.text == "First paragraph."
    assert first.page_start == 1
    assert first.page_end == 1
    assert first.para_ids == ["p1"]
    assert first.coords == ["1,10,10,50,50"]
    assert first.cited_ref_ids == []
    assert first.title == "Parsed Paper"
    assert first.authors == ["Ada Lovelace", "Grace Hopper"]
    assert first.source_url == "https://source.test/paper.pdf"
    assert first.embedding_model == "fake-embedding-model"
    assert first.embedding_dims == 3
    assert first.embedding_ts.tzinfo is not None
    assert first.parser == "FakeParser"

    assert second.para_ids == []
    assert second.coords == []
    assert second.page_start == 2
    assert second.page_end == 3
    assert store.references == []


def test_ingest_pdf_uses_explicit_metadata_without_extracting_metadata() -> None:
    parser = FakeParser()
    store = FakeStore()
    ingestor = CorpusIngestor(parser=parser, store=store)
    created_at = datetime(2025, 1, 2, tzinfo=timezone.utc)
    meta = Document(
        doc_id="explicit-doc",
        title="Explicit Paper",
        authors=["Test Author"],
        created_at=created_at,
        url="https://example.test/meta",
    )

    doc = ingestor.ingest_pdf(b"pdf bytes", meta=meta, source_url=None)

    assert parser.metadata_calls == []
    assert parser.parse_calls == [b"pdf bytes"]
    assert doc.doc_id == "explicit-doc"
    assert doc.title == "Explicit Paper"
    assert doc.created_at == created_at
    assert store.chunks[0].source_url == "https://example.test/meta"


def test_ingest_pdf_persists_bibliography_references_and_chunk_citation_links() -> None:
    class Parser(FakeParser):
        def parse(self, pdf_bytes: bytes) -> tuple[list[Paragraph], str | None]:
            self.parse_calls.append(pdf_bytes)
            return (
                [
                    Paragraph(
                        text="External record claim [11].",
                        page_start=2,
                        page_end=2,
                        para_id="p1",
                        coords="2,10,10,50,50",
                        cited_ref_ids=["b10"],
                    )
                ],
                "<TEI />",
            )

        def parse_references(
            self,
            tei_xml: str,
            source_doc_id: str,
        ) -> list[DocumentReference]:
            return [
                DocumentReference(
                    reference_id=f"{source_doc_id}:b10",
                    source_doc_id=source_doc_id,
                    ref_id="b10",
                    title="Record tandem solar cells",
                    authors=["Sara Record"],
                    year=2025,
                    doi="10.1234/record",
                    raw_text="Sara Record, Record tandem solar cells",
                )
            ]

    parser = Parser()
    store = FakeStore()
    ingestor = CorpusIngestor(parser=parser, store=store)

    doc = ingestor.ingest_pdf(b"pdf bytes", meta=None, source_url=None)

    assert store.references == [
        DocumentReference(
            reference_id=f"{doc.doc_id}:b10",
            source_doc_id=doc.doc_id,
            ref_id="b10",
            title="Record tandem solar cells",
            authors=["Sara Record"],
            year=2025,
            doi="10.1234/record",
            raw_text="Sara Record, Record tandem solar cells",
        )
    ]
    assert store.chunks[0].cited_ref_ids == ["b10"]


def test_ingest_pdf_merges_short_adjacent_paragraphs_and_keeps_tables_standalone() -> None:
    class Parser(FakeParser):
        def parse(self, pdf_bytes: bytes) -> tuple[list[Paragraph], str | None]:
            self.parse_calls.append(pdf_bytes)
            return (
                [
                    Paragraph(
                        text="Short result one.",
                        page_start=2,
                        page_end=2,
                        para_id="p1",
                        coords="2,10,10,50,50",
                        section_path=["Results"],
                    ),
                    Paragraph(
                        text="Short result two.",
                        page_start=2,
                        page_end=2,
                        para_id="p2",
                        coords="2,20,10,50,50",
                        section_path=["Results"],
                    ),
                    Paragraph(
                        text="[TABLE]\nCaption: Table 1 Device metrics.\n| Metric | Value |\n| --- | --- |\n| PCE | 25.1% |",
                        page_start=2,
                        page_end=2,
                        para_id="tab_1",
                        coords="2,30,10,50,50",
                        section_path=["Results"],
                        block_type="table",
                        caption="Table 1 Device metrics.",
                        cells=[["Metric", "Value"], ["PCE", "25.1%"]],
                    ),
                    Paragraph(
                        text="Short result after table.",
                        page_start=2,
                        page_end=2,
                        para_id="p3",
                        coords="2,40,10,50,50",
                        section_path=["Results"],
                    ),
                ],
                "<TEI />",
            )

    parser = Parser()
    store = FakeStore()
    ingestor = CorpusIngestor(parser=parser, store=store)

    ingestor.ingest_pdf(b"pdf bytes", meta=None, source_url=None)

    assert len(store.chunks) == 3
    merged, table, after_table = store.chunks
    assert merged.text == "Short result one.\n\nShort result two."
    assert merged.para_ids == ["p1", "p2"]
    assert merged.coords == ["2,10,10,50,50", "2,20,10,50,50"]
    assert merged.section_path == ["Results"]
    assert merged.chunk_type == "paragraph"

    assert table.text.startswith("[TABLE]\nCaption: Table 1 Device metrics.")
    assert table.para_ids == ["tab_1"]
    assert table.section_path == ["Results"]
    assert table.chunk_type == "table"
    assert table.caption == "Table 1 Device metrics."

    assert after_table.text == "Short result after table."
    assert after_table.para_ids == ["p3"]
    assert after_table.chunk_type == "paragraph"


def test_ingest_pdf_splits_very_long_paragraphs_by_sentence() -> None:
    class Parser(FakeParser):
        def parse(self, pdf_bytes: bytes) -> tuple[list[Paragraph], str | None]:
            self.parse_calls.append(pdf_bytes)
            long_text = f"{'A' * 900}. {'B' * 900}."
            return (
                [
                    Paragraph(
                        text=long_text,
                        page_start=4,
                        page_end=4,
                        para_id="p-long",
                        coords="4,10,10,50,50",
                        section_path=["Discussion"],
                    )
                ],
                "<TEI />",
            )

    parser = Parser()
    store = FakeStore()
    ingestor = CorpusIngestor(parser=parser, store=store)

    ingestor.ingest_pdf(b"pdf bytes", meta=None, source_url=None)

    assert len(store.chunks) == 2
    first, second = store.chunks
    assert first.text == f"{'A' * 900}."
    assert second.text == f"{'B' * 900}."
    assert first.para_ids == ["p-long"]
    assert second.para_ids == ["p-long"]
    assert first.page_start == second.page_start == 4
    assert first.section_path == second.section_path == ["Discussion"]
