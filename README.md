Vigyan — SDK for agentic search on scientific documents with citations
======================================================================

Overview
--------

Vigyan provides a small, clean Python SDK to parse scientific PDFs, embed the content, index it in a vector database, and answer research questions with citation-aware metadata (paper, page range, paragraph ids, etc.).

Design Principles
-----------------

- Clear interfaces: `VectorStore` and `DocumentParser` decouple concerns.
- Storage-agnostic domain models: `Document`, `Chunk`, and `QueryHit`.
- Adapter implementations: LanceDB vector store with built-in embedding, GROBID parser.
- Simple SDK pipeline: `ingest_pdf`, `query`, and `run_research_query` orchestrate the workflow.

Install
-------

Requires Python 3.12+.

Dependencies include `lancedb`, `httpx`, `lxml`, and `pydantic` (declared in `pyproject.toml`).

Quick Start
-----------

```python
from vigyan import LanceDBVectorStore, GrobidParser, ingest_pdf, query
from vigyan.agent import run_research_query

# Configure components
store = LanceDBVectorStore(embedding_model="text-embedding-3-small")
parser = GrobidParser(server_url="http://localhost:8070")  # GROBID must be running

# Ingest a PDF with automatic metadata (via GROBID)
pdf_bytes = open("paper.pdf", "rb").read()
ingest_pdf(pdf_bytes, meta=None, parser=parser, store=store)

# Retrieve relevant passages directly
hits = query("protein folding with attention", store=store, top_k=5)
for h in hits:
    print(h.citation, "-", h.title)
    print(h.text)

# Or run the research agent for a cited answer
answer = run_research_query(
    "What does this corpus say about protein folding with attention?",
    db_uri="./vigyan_db",
    embed_model="text-embedding-3-small",
)
print(answer.answer)
for citation in answer.citations:
    print(f"[{citation.index}] {citation.citation}")
```

Notes
-----

- OpenAI-compatible key must be available in the environment for embedding.
- GROBID must be running for parsing and metadata extraction. You can swap in a different `DocumentParser` implementation if preferred.
- The LanceDB store uses auto-embedding via the LanceDB registry, supporting OpenAI and other providers.
