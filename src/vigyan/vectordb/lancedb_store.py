from datetime import datetime
from pathlib import Path

import lancedb
import platformdirs
from lancedb.pydantic import LanceModel, Vector

from ..core.interfaces import Embedder, VectorStore
from ..core.models import Chunk, Document, QueryHit


def default_lancedb_path() -> str:
    """Return the default LanceDB path in user data directory."""
    return str(Path(platformdirs.user_data_dir("vigyan")) / "lancedb")


class DocumentRecord(LanceModel):
    doc_id: str
    title: str
    authors: list[str]
    venue: str | None = None
    year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    url: str | None = None
    pdf_sha256: str | None = None
    n_pages: int
    tei_xml: str | None = None
    created_at: datetime


def make_chunk_record_model(vector_dim: int):
    class ChunkRecord(LanceModel):
        chunk_id: str
        doc_id: str
        text: str
        vector: Vector(vector_dim)  # type: ignore[valid-type]
        # Citation & structure
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

    return ChunkRecord


class LanceDBVectorStore(VectorStore):
    """LanceDB-based VectorStore implementation.

    This store requires an `Embedder` to compute embeddings and fix the vector
    dimension at table creation time.
    """

    def __init__(self, embedder: Embedder, uri: str | None = None) -> None:
        self._uri = uri or default_lancedb_path()
        self._embedder = embedder
        self._db: lancedb.DBConnection | None = None
        self._docs_tbl = None
        self._chunks_tbl = None

    def create_or_open(self) -> None:
        db = lancedb.connect(self._uri)

        # Create or open documents table
        if "documents" in db.table_names():
            docs_tbl = db.open_table("documents")
        else:
            docs_tbl = db.create_table(
                "documents", schema=DocumentRecord, mode="create"
            )

        ChunkRecord = make_chunk_record_model(self._embedder.dim)
        # Create or open chunks table
        if "chunks" in db.table_names():
            chunks_tbl = db.open_table("chunks")
        else:
            chunks_tbl = db.create_table("chunks", schema=ChunkRecord, mode="create")

        # Basic FTS support for hybrid strategies if needed
        try:
            chunks_tbl.create_fts_index("text", use_tantivy=True)
        except Exception as e:
            print(e)
            # Index may already exist; ignore
            pass

        self._db = db
        self._docs_tbl = docs_tbl
        self._chunks_tbl = chunks_tbl

    def upsert_documents(self, docs: list[Document]) -> None:
        assert self._docs_tbl is not None, "Call create_or_open() first"
        rows = [
            DocumentRecord(**d.model_dump()).model_dump()  # ensure schema
            for d in docs
        ]
        if rows:
            # Simple add; for true upsert you'd need a merge strategy
            self._docs_tbl.add(rows)

    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        assert self._chunks_tbl is not None, "Call create_or_open() first"
        rows = [c.model_dump() for c in chunks]
        if rows:
            self._chunks_tbl.add(rows)

    def search(
        self, query: str, top_k: int = 8, filters: str | None = None
    ) -> list[QueryHit]:
        assert self._chunks_tbl is not None, "Call create_or_open() first"
        qvec = self._embedder.embed([query])[0]
        q = self._chunks_tbl.search(qvec)
        if filters:
            q = q.where(filters)
        hits = (
            q.limit(top_k)
            .select(
                [
                    "_distance",
                    "chunk_id",
                    "doc_id",
                    "text",
                    "title",
                    "authors",
                    "venue",
                    "year",
                    "doi",
                    "arxiv_id",
                    "page_start",
                    "page_end",
                    "para_ids",
                    "section_path",
                    "coords",
                ]
            )
            .to_list()
        )
        return [self._format_hit(h) for h in hits]

    @staticmethod
    def _format_hit(h: dict) -> QueryHit:
        authors = h["authors"]
        short_auth = (
            (authors[0] + " et al.")
            if authors and len(authors) > 1
            else (authors[0] if authors else "Unknown")
        )
        title = h["title"]
        year = h.get("year") or ""
        pp = (
            f"p. {h['page_start']}"
            if h["page_start"] == h["page_end"]
            else f"pp. {h['page_start']}-{h['page_end']}"
        )
        cite = f"{short_auth} {title} ({year}), {pp}"

        return QueryHit(
            doc_id=h["doc_id"],
            title=title,
            year=h.get("year"),
            doi=h.get("doi"),
            arxiv_id=h.get("arxiv_id"),
            page_span=(h["page_start"], h["page_end"]),
            text=h["text"],
            citation=cite,
            distance=h.get("_distance"),
        )
