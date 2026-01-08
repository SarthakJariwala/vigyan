from __future__ import annotations

from pathlib import Path
from typing import Annotated

from cyclopts import Parameter

from ..parsers.grobid import GrobidParser
from ..pipeline import ingest_pdf
from ..vectordb.lancedb_store import LanceDBVectorStore, default_lancedb_path
from .app import app


@app.command
def ingest(
    pdf: Annotated[Path, Parameter(help="Path to PDF file")],
    *,
    db: Annotated[str, Parameter(help="LanceDB URI")] = default_lancedb_path(),
    grobid: Annotated[str, Parameter(help="GROBID server URL")] = "http://localhost:8070",
    embed_model: Annotated[str, Parameter(help="OpenAI embedding model")] = "text-embedding-3-small",
) -> None:
    """Ingest a PDF into the vector store."""
    pdf_bytes = pdf.read_bytes()
    store = LanceDBVectorStore(uri=db, embedding_model=embed_model)
    parser = GrobidParser(server_url=grobid)
    doc = ingest_pdf(
        pdf_bytes=pdf_bytes,
        meta=None,
        parser=parser,
        store=store,
    )
    print("Ingested:", doc.doc_id)
