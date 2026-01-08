Vigyan — Agentic search on scientific documents with citations
==============================================================

Overview
--------

Vigyan provides a small, clean Python API to parse scientific PDFs, embed the content, and index it in a vector database with citation-aware metadata (paper, page range, paragraph ids, etc.).

Design Principles
-----------------

- Clear interfaces: `VectorStore` and `DocumentParser` decouple concerns.
- Storage-agnostic domain models: `Document`, `Chunk`, and `QueryHit`.
- Adapter implementations: LanceDB vector store with built-in embedding, GROBID parser.
- Simple pipeline: `ingest_pdf` and `query` orchestrate the workflow.

Install
-------

Requires Python 3.12+.

Dependencies include `lancedb`, `httpx`, `lxml`, and `pydantic` (declared in `pyproject.toml`).

Quick Start (Code)
------------------

```python
from vigyan import (
    Document,
    LanceDBVectorStore,
    GrobidParser,
    ingest_pdf,
    query,
)

# Configure components
store = LanceDBVectorStore(embedding_model="text-embedding-3-small")
parser = GrobidParser(server_url="http://localhost:8070")  # GROBID must be running

# Ingest a PDF with automatic metadata (via GROBID)
pdf_bytes = open("paper.pdf", "rb").read()
ingest_pdf(pdf_bytes, meta=None, parser=parser, store=store)

# Query
hits = query("protein folding with attention", store=store, top_k=5)
for h in hits:
    print(h.citation, "-", h.title)
    print(h.text)
```

CLI
---

```
vigyan ingest --db ./vigyan_db --pdf ./paper.pdf

vigyan query --db ./vigyan_db --q "protein folding with attention"
```

Notes
-----

- OpenAI-compatible key must be available in the environment for embedding.
- GROBID must be running for parsing and metadata extraction. You can swap in a different `DocumentParser` implementation if preferred.
- The LanceDB store uses auto-embedding via the LanceDB registry, supporting OpenAI and other providers.
