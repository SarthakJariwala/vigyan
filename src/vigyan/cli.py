from __future__ import annotations

import argparse
import json
from pathlib import Path

from .embedders.openai_embedder import OpenAIEmbedder
from .parsers.grobid import GrobidParser
from .pipeline import ingest_pdf, query
from .vectordb.lancedb_store import LanceDBVectorStore, default_lancedb_path


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vigyan")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest a PDF into the vector store")
    p_ingest.add_argument(
        "--db", default=default_lancedb_path(), help="LanceDB URI (./db)"
    )
    p_ingest.add_argument("--pdf", required=True, help="Path to PDF file")
    p_ingest.add_argument(
        "--grobid", default="http://localhost:8070", help="GROBID server URL"
    )
    p_ingest.add_argument(
        "--embed-model",
        default="text-embedding-3-small",
        help="OpenAI embedding model",
    )

    p_query = sub.add_parser("query", help="Run a semantic query against the store")
    p_query.add_argument(
        "--db", default=default_lancedb_path(), help="LanceDB URI (./db)"
    )
    p_query.add_argument("--q", required=True, help="Query string")
    p_query.add_argument("--top-k", type=int, default=8)
    p_query.add_argument("--filter")
    p_query.add_argument(
        "--embed-model",
        default="text-embedding-3-small",
        help="OpenAI embedding model",
    )

    args = parser.parse_args(argv)

    if args.cmd == "ingest":
        pdf_path = Path(args.pdf)
        pdf_bytes = pdf_path.read_bytes()
        embedder = OpenAIEmbedder(model=args.embed_model)
        store = LanceDBVectorStore(uri=args.db, embedder=embedder)
        parser_obj = GrobidParser(server_url=args.grobid)
        ret_doc = ingest_pdf(
            pdf_bytes=pdf_bytes,
            meta=None,
            parser=parser_obj,
            embedder=embedder,
            store=store,
        )
        print("Ingested:", ret_doc.doc_id)
        return 0

    if args.cmd == "query":
        embedder = OpenAIEmbedder(model=args.embed_model)
        store = LanceDBVectorStore(uri=args.db, embedder=embedder)
        hits = query(text=args.q, store=store, top_k=args.top_k, filters=args.filter)
        print(json.dumps([h.model_dump() for h in hits], indent=2))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(cli())
