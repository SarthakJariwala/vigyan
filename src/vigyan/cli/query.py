from __future__ import annotations

import json
from typing import Annotated

from cyclopts import Parameter

from ..pipeline import query as run_query
from ..vectordb.lancedb_store import LanceDBVectorStore, default_lancedb_path
from .app import app


@app.command
def query(
    q: Annotated[str, Parameter(help="Query string")],
    *,
    db: Annotated[str, Parameter(help="LanceDB URI")] = default_lancedb_path(),
    top_k: Annotated[int, Parameter(help="Number of results to return")] = 8,
    filter: Annotated[str | None, Parameter(help="Filter expression")] = None,
    embed_model: Annotated[str, Parameter(help="OpenAI embedding model")] = "text-embedding-3-small",
) -> None:
    """Run a semantic query against the store."""
    store = LanceDBVectorStore(uri=db, embedding_model=embed_model)
    hits = run_query(text=q, store=store, top_k=top_k, filters=filter)
    print(json.dumps([h.model_dump() for h in hits], indent=2))
