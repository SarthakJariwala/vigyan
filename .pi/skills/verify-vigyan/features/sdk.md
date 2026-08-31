# SDK behavior

Vigyan's main user interface is its Python API. Verify calls through public imports and label every parser, store, embedding, and model boundary.

## Public entry points

- `vigyan.models`: `Document`, `Paragraph`, `Chunk`, `DocumentReference`, `CitedReference`, and `QueryHit`
- `vigyan.corpus`: `CorpusIngestor` and `CorpusRetriever`
- `vigyan.parsers`: `GrobidParser`
- `vigyan.vectordb`: `LanceDBVectorStore`
- `vigyan.agent`: `ResearchAgentDeps`, `agent`, and `run_research_query`

## Deterministic baseline

Capture the full suite:

```bash
"$VERIFY_VIGYAN" run "$RUN_ID" \
  --evidence sdk/pytest.txt --timeout 300 -- \
  uv run --no-sync python -m pytest -q -p no:cacheprovider
```

The suite uses protocol fakes and mocked HTTP/model boundaries. It covers:

- ingestion order, metadata, PDF hashing, page counts, references, chunk merging, sentence splitting, and standalone tables;
- GROBID request fields, coordinate fallback, table extraction, and bibliography parsing;
- retrieval delegation and `QueryHit` citation formatting;
- public imports and environment-backed agent dependency resolution.

## Focused public-SDK proof

When the change affects orchestration, write a short temporary script under the run state and invoke it through `verify-vigyan run`. Use `CorpusIngestor` or `CorpusRetriever`, not private helpers. Save structured output such as `Document.model_dump_json()` or `QueryHit.model_dump_json()` in the transcript.

For a mutation, show both sides:

1. ingest or upsert through the public API;
2. retrieve through `CorpusRetriever` and capture the resulting hit.

If the script uses a fake parser or store, say so in `notes.txt`. That proof ends at the protocol seam.

## Stable observations

Prefer domain fields over implementation files:

- document ID, title, SHA-256, and page count;
- chunk text, page span, section path, type, caption, and cited reference IDs;
- hit text, distance, citation, DOI, page span, and cited reference metadata;
- forwarded query, `top_k`, and filter expression.

UUIDs and timestamps are run-specific. LanceDB's directory layout and table internals are not user contracts.
