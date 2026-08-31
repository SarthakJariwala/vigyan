# External services

Vigyan has three external boundaries. Exercise only the service named in the verification request and keep data in the isolated run.

## GROBID

`GrobidParser` defaults to `http://localhost:8070`. A live ingestion check needs a reachable GROBID service and a disposable or redistributable PDF.

Capture:

- the GROBID base URL without credentials or sensitive query parameters;
- the PDF SHA-256, not private document bytes;
- extracted title, authors, paragraphs, tables, page spans, and references;
- the returned `Document` and a retrieval read-back if storage is also in scope.

A mocked `httpx.post` proves request and TEI parsing contracts, not the running GROBID deployment.

## Embeddings and LanceDB

Opening a `LanceDBVectorStore` can create local tables and indexes. Use the DB path returned by the helper. A real semantic retrieval check also calls the configured embedding provider.

Capture:

- provider and embedding model names;
- whether the endpoint is production, local, or a named protocol double;
- indexed document and chunk counts through supported APIs where available;
- the query, filters, `top_k`, and serialized `QueryHit` read-back.

Never print API keys. A local OpenAI-compatible double proves request compatibility only. It does not prove the production embedding account or model quality.

## Model-backed answers

The global web agent is configured for `anthropic:claude-opus-4-8`. `run_research_query` may use that model or an explicit override. A cited-answer check needs a populated isolated corpus, working embeddings, and an approved model credential.

Capture:

- the exact public entry point and model name;
- the user question;
- retrieved `QueryHit` evidence and page ranges;
- the complete visible answer and citations;
- provider or validation errors, including the exit or HTTP status.

Check that each scientific claim points to retrieved text. If a chunk cites a source that is absent from the corpus, the answer must identify that secondary-support limitation. A successful model response without retrieval evidence does not prove citation grounding.

## Credentials

Use a separate run launched with `--allow-provider-credentials` for approved live embedding or model checks. Keep values in the process environment. Do not write secrets into evidence, commands, notes, or repository files. If credentials are not already available and the user asks for a live check, request them through the secure secrets workflow.
