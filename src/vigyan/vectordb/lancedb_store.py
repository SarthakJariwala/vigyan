from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

import lancedb
import platformdirs
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector

from ..interfaces import VectorStore
from ..models import Chunk, CitedReference, Document, DocumentReference, QueryHit


def default_lancedb_path() -> str:
    """Return the default LanceDB path in user data directory."""
    return str(Path(platformdirs.user_data_dir("vigyan")) / "lancedb")


def _normalize_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    value = doi.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if value.startswith(prefix):
            value = value[len(prefix) :]
    value = value.split("?", 1)[0].strip()
    return value or None


def _normalize_title(title: str | None) -> str | None:
    if not title:
        return None
    return " ".join(title.lower().split()) or None


def _format_reference_authors(authors: list[str]) -> str | None:
    if not authors:
        return None
    return authors[0] if len(authors) == 1 else f"{authors[0]} et al."


def _compact_cited_reference(reference: DocumentReference) -> CitedReference:
    return CitedReference(
        ref_id=reference.ref_id,
        title=reference.title,
        authors=_format_reference_authors(reference.authors),
        doi=_normalize_doi(reference.doi),
        year=reference.year,
        journal=reference.venue,
        in_corpus=reference.in_corpus,
        resolved_doc_id=reference.resolved_doc_id,
    )


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


class ReferenceRecord(LanceModel):
    reference_id: str
    source_doc_id: str
    ref_id: str
    label: str | None = None
    title: str | None = None
    authors: list[str] = []
    venue: str | None = None
    year: int | None = None
    doi: str | None = None
    url: str | None = None
    raw_text: str = ""
    in_corpus: bool = False
    resolved_doc_id: str | None = None


def make_chunk_record_model(embedding_fn):
    """Create a ChunkRecord model with auto-embedding via LanceDB."""

    class ChunkRecord(LanceModel):
        chunk_id: str
        doc_id: str
        text: str = embedding_fn.SourceField()
        vector: Vector(embedding_fn.ndims()) = embedding_fn.VectorField()  # type: ignore[valid-type]
        # Citation & structure
        page_start: int
        page_end: int
        para_ids: list[str] = []
        section_path: list[str] = []
        char_start: int | None = None
        char_end: int | None = None
        coords: list[str] = []
        cited_ref_ids: list[str] = []
        chunk_type: str = "paragraph"
        caption: str | None = None
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
    """LanceDB-based VectorStore with auto-embedding.

    Uses LanceDB's built-in embedding registry to automatically embed text
    on insert and query.
    """

    def __init__(
        self,
        uri: str | None = None,
        embedding_model: str = "text-embedding-3-large",
        embedding_provider: str = "openai",
        dim: int | None = None,
        base_url: str | None = None,
        api_key_env: str | None = None,
    ) -> None:
        self._uri = uri or default_lancedb_path()
        self._embedding_model = embedding_model
        self._embedding_provider = embedding_provider
        self._dim = dim
        self._base_url = base_url
        self._api_key_env = api_key_env
        self._db: lancedb.DBConnection | None = None
        self._docs_tbl = None
        self._chunks_tbl = None
        self._refs_tbl = None
        self._embedding_fn = None

    @property
    def model_name(self) -> str:
        return self._embedding_model

    @property
    def dim(self) -> int:
        if self._embedding_fn is not None:
            return self._embedding_fn.ndims()
        raise RuntimeError("Call create_or_open() first to initialize embedding function")

    def _create_embedding_fn(self):
        """Create the embedding function from the LanceDB registry."""
        registry = get_registry()
        provider = registry.get(self._embedding_provider)
        kwargs = {"name": self._embedding_model}
        if self._dim is not None:
            kwargs["dim"] = self._dim
        if self._base_url:
            kwargs["base_url"] = self._base_url
        if self._api_key_env:
            kwargs["api_key_env"] = self._api_key_env
        return provider.create(**kwargs)

    def create_or_open(self) -> None:
        if self._embedding_fn is None:
            self._embedding_fn = self._create_embedding_fn()

        db = lancedb.connect(self._uri)

        # Create or open documents table
        if "documents" in db.table_names():
            docs_tbl = db.open_table("documents")
        else:
            docs_tbl = db.create_table(
                "documents", schema=DocumentRecord, mode="create"
            )

        ChunkRecord = make_chunk_record_model(self._embedding_fn)
        # Create or open references table
        if "references" in db.table_names():
            refs_tbl = db.open_table("references")
        else:
            refs_tbl = db.create_table("references", schema=ReferenceRecord, mode="create")

        # Create or open chunks table
        if "chunks" in db.table_names():
            chunks_tbl = db.open_table("chunks")
        else:
            chunks_tbl = db.create_table("chunks", schema=ChunkRecord, mode="create")

        # Basic FTS support for hybrid strategies if needed
        try:
            chunks_tbl.create_fts_index("text", use_tantivy=True)
        except Exception:
            pass

        self._db = db
        self._docs_tbl = docs_tbl
        self._chunks_tbl = chunks_tbl
        self._refs_tbl = refs_tbl

    def upsert_documents(self, docs: list[Document]) -> None:
        assert self._docs_tbl is not None, "Call create_or_open() first"
        rows = [
            DocumentRecord(**d.model_dump()).model_dump()
            for d in docs
        ]
        if rows:
            self._docs_tbl.add(rows)

    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        assert self._chunks_tbl is not None, "Call create_or_open() first"
        rows = [c.model_dump() for c in chunks]
        if rows:
            self._chunks_tbl.add(rows)

    def upsert_references(self, references: list[DocumentReference]) -> None:
        assert self._refs_tbl is not None, "Call create_or_open() first"
        rows = [ReferenceRecord(**r.model_dump()).model_dump() for r in references]
        if rows:
            self._refs_tbl.add(rows)

    def search(
        self, query: str, top_k: int = 8, filters: str | None = None
    ) -> list[QueryHit]:
        assert self._chunks_tbl is not None, "Call create_or_open() first"
        q = self._chunks_tbl.search(query)
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
                    "cited_ref_ids",
                    "chunk_type",
                    "caption",
                ]
            )
            .to_list()
        )
        references_by_hit = self._cited_references_by_hit(hits)
        return [
            self._format_hit(
                h,
                cited_references=references_by_hit.get(
                    (h["doc_id"], tuple(h.get("cited_ref_ids") or [])), []
                ),
            )
            for h in hits
        ]

    def _cited_references_by_hit(
        self,
        hits: list[dict],
    ) -> dict[tuple[str, tuple[str, ...]], list[DocumentReference]]:
        if self._refs_tbl is None or self._docs_tbl is None:
            return {}

        needed: set[tuple[str, str]] = set()
        for hit in hits:
            for ref_id in hit.get("cited_ref_ids") or []:
                needed.add((hit["doc_id"], ref_id))
        if not needed:
            return {}

        reference_rows = self._refs_tbl.to_arrow().to_pylist()
        document_rows = self._docs_tbl.to_arrow().to_pylist()
        documents = [DocumentRecord(**row) for row in document_rows]

        refs_by_source: dict[tuple[str, str], DocumentReference] = {}
        for row in reference_rows:
            key = (row["source_doc_id"], row["ref_id"])
            if key not in needed or key in refs_by_source:
                continue
            reference = DocumentReference(**row)
            refs_by_source[key] = self._resolve_reference(reference, documents)

        references_by_hit: dict[tuple[str, tuple[str, ...]], list[DocumentReference]] = {}
        for hit in hits:
            ref_ids = tuple(hit.get("cited_ref_ids") or [])
            references_by_hit[(hit["doc_id"], ref_ids)] = [
                refs_by_source[(hit["doc_id"], ref_id)]
                for ref_id in ref_ids
                if (hit["doc_id"], ref_id) in refs_by_source
            ]
        return references_by_hit

    @staticmethod
    def _resolve_reference(
        reference: DocumentReference,
        documents: list[DocumentRecord],
    ) -> DocumentReference:
        reference_doi = _normalize_doi(reference.doi)
        if reference_doi:
            for document in documents:
                if _normalize_doi(document.doi) == reference_doi:
                    return reference.model_copy(
                        update={"in_corpus": True, "resolved_doc_id": document.doc_id}
                    )

        reference_title = _normalize_title(reference.title)
        if reference_title:
            best_doc_id: str | None = None
            best_score = 0.0
            for document in documents:
                document_title = _normalize_title(document.title)
                if not document_title:
                    continue
                score = SequenceMatcher(None, reference_title, document_title).ratio()
                if score > best_score:
                    best_doc_id = document.doc_id
                    best_score = score
            if best_doc_id and best_score >= 0.92:
                return reference.model_copy(
                    update={"in_corpus": True, "resolved_doc_id": best_doc_id}
                )

        return reference.model_copy(update={"in_corpus": False, "resolved_doc_id": None})

    @staticmethod
    def _format_hit(
        h: dict,
        cited_references: list[DocumentReference] | None = None,
    ) -> QueryHit:
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
            section_path=h.get("section_path") or [],
            chunk_type=h.get("chunk_type") or "paragraph",
            caption=h.get("caption"),
            cited_ref_ids=h.get("cited_ref_ids") or [],
            cited_references=[
                _compact_cited_reference(reference)
                for reference in cited_references or []
            ],
            text=h["text"],
            citation=cite,
            distance=h.get("_distance"),
        )
