# Vigyan verification map

Choose the smallest feature set that reaches the changed user behavior. Start every recipe from a new run and require `HEALTHY` before reporting success.

## Baseline

- Run from the repository root.
- Use a unique `RUN_ID` and `verify-vigyan launch`.
- Keep runtime state under `.pi/verification/vigyan/.state/$RUN_ID/` and proof under `.pi/verification/vigyan/$RUN_ID/`.
- Use the isolated DB path returned by `verify-vigyan path "$RUN_ID" db`.
- Capture each command or request in its own evidence file.
- Run `verify-vigyan cleanup` even when a check fails. Cleanup preserves evidence.

## Feature map

- [SDK behavior](sdk.md) covers public imports, ingestion and chunking orchestration, retrieval delegation, citation metadata, and environment-backed agent dependencies.
- [CLai web](clai-web.md) covers the documented web command, server readiness, agent configuration, and the local UI route.
- [External services](external-services.md) defines the additional proof needed for GROBID, embeddings, LanceDB retrieval, and model-backed cited answers.

## Reporting skips

Name the skipped path and missing precondition. Examples:

- `GROBID ingestion skipped: no reachable GROBID URL was supplied.`
- `Live retrieval skipped: no embedding provider credential was approved for this run.`
- `Cited answer skipped: the isolated corpus is empty and no model-backed request was made.`

Do not substitute a lower-level check and claim the skipped behavior.
