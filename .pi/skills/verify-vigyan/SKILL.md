---
name: verify-vigyan
description: Launch and verify Vigyan through its public Python SDK and checkout-only contributor CLai agent. Use after changes to ingestion, GROBID parsing, chunking, LanceDB retrieval, research capabilities, citations, environment configuration, or contributor web verification, and whenever user-visible evidence is needed.
compatibility: Linux checkout with uv, Python 3.12+, and /proc available for safe process cleanup.
---

# Verify Vigyan

Vigyan is a Python SDK. It does not own a CLI or a ready-made agent. Contributors can run the checkout-only `vigyan_dev.clai_agent:agent` host through CLai.

The helper starts that command on a loopback-only random port with isolated HOME and XDG directories. Every run gets separate state, a LanceDB path, a server log, and a retained evidence directory. Never point verification at the developer's normal corpus.

Read [`features/README.md`](features/README.md) before choosing a recipe. It separates local proof from checks that need GROBID, an embedding provider, or an LLM.

## Start an isolated run

From the repository root:

```bash
export RUN_ID="vigyan-$(date -u +%Y%m%dT%H%M%SZ)-$$"
export VERIFY_VIGYAN=".pi/skills/verify-vigyan/bin/verify-vigyan"
"$VERIFY_VIGYAN" launch "$RUN_ID"
```

Readiness is the line `READY Vigyan <version>`. Launch runs `uv sync --locked`, imports the checkout-only contributor agent through CLai, serves a run-specific local marker page instead of downloading UI HTML, and waits for `/api/health`, the marker page, and the expected model in `/api/configure`. Dependency sync may contact the configured Python package index when the locked packages are not cached.

By default, the run removes inherited `VIGYAN_*` settings, hides `OPENAI_API_KEY`, and gives CLai a non-working Anthropic placeholder so it can build `/api/configure` without using the operator's credential. Launch does not send model or embedding requests.

For an explicitly approved live model or embedding check, start a separate run with `--allow-provider-credentials`. This passes inherited provider credentials to the server without recording their values:

```bash
"$VERIFY_VIGYAN" launch "$RUN_ID" --allow-provider-credentials
```

## Check health

```bash
"$VERIFY_VIGYAN" doctor "$RUN_ID"
```

Require `HEALTHY`. Doctor checks the checkout revision, project version, recorded process marker, health response, local marker page, and `/api/configure`. It writes `doctor.txt` to the evidence directory.

Resolve run paths through the helper:

```bash
EVIDENCE=$("$VERIFY_VIGYAN" path "$RUN_ID" evidence)
DB_URI=$("$VERIFY_VIGYAN" path "$RUN_ID" db)
SERVER_LOG=$("$VERIFY_VIGYAN" path "$RUN_ID" server-log)
```

## Capture checks

Run the test suite in the isolated environment and save the command, stdout, stderr, timeout, and exit code together:

```bash
"$VERIFY_VIGYAN" run "$RUN_ID" \
  --evidence sdk/pytest.txt --timeout 300 -- \
  uv run --no-sync python -m pytest -q -p no:cacheprovider
```

Capture the contributor web check separately because unit tests do not prove that CLai can import the checkout host:

```bash
"$VERIFY_VIGYAN" request "$RUN_ID" /api/health \
  --evidence clai-web/health.txt
"$VERIFY_VIGYAN" request "$RUN_ID" /api/configure \
  --evidence clai-web/configure.txt
"$VERIFY_VIGYAN" request "$RUN_ID" / \
  --evidence clai-web/root.txt
```

Evidence paths are relative to `.pi/verification/vigyan/$RUN_ID/`. The `run` and `request` commands refuse absolute paths, parent traversal, and overwrites. `doctor.txt` is the current health snapshot and is replaced by later doctor calls.

Use `run` for a focused public-SDK script when a change needs proof beyond tests. Import from `vigyan`, `vigyan.agent`, `vigyan.corpus`, `vigyan.parsers`, or `vigyan.vectordb`, as a user would. Record which adapter is real and which boundary is replaced by a fake.

## Claim only what ran

A valid proof names its boundary:

- The test suite proves deterministic SDK contracts at the parser, store, HTTP, and model seams. It does not prove a live service.
- `/api/health`, `/api/configure`, and `/` prove that the locked development tools can import and serve the checkout-only agent. They do not prove chat, retrieval, or citations.
- A fake `DocumentParser` or `VectorStore` proves SDK orchestration. It does not prove GROBID, LanceDB, or embeddings.
- A local OpenAI-compatible embedding double proves integration against that named protocol boundary. It does not prove the configured provider account or production model.
- A real PDF ingestion claim requires a reachable GROBID service. A real vector retrieval claim requires the configured embedding provider. A cited answer requires both populated retrieval data and a successful LLM response.

Never label unit tests or a health endpoint as end-to-end scientific search. Do not send a document, query, or credential to an external service unless the verification request calls for that service.

## Evidence contract

Keep proof under `.pi/verification/vigyan/$RUN_ID/`. A useful set contains:

- `launch.json`, which records the revision, initial dirty-worktree state, package versions, exact contributor server command, port, health response, and configuration response;
- `server.log`, which captures CLai and Uvicorn output;
- one transcript per check, with the command or request and immediate result;
- `doctor.txt` from the final health check;
- `notes.txt` when a fake, external service, skipped path, or manual observation needs explanation;
- `cleanup.txt` after teardown.

For ingestion or retrieval mutations, capture a second read through the public SDK. Do not treat files under the LanceDB directory as a stable user contract.

## Clean up

```bash
"$VERIFY_VIGYAN" cleanup "$RUN_ID"
test ! -e ".pi/verification/vigyan/.state/$RUN_ID"
test -f ".pi/verification/vigyan/$RUN_ID/launch.json"
test -f ".pi/verification/vigyan/$RUN_ID/cleanup.txt"
```

On Linux, cleanup checks `/proc` for the process environment marker before signaling its process group. It removes only that run's isolated state and preserves evidence. Run cleanup after failed checks too.

## Helper commands

```text
verify-vigyan launch RUN_ID [--timeout SECONDS] [--allow-provider-credentials]
verify-vigyan doctor RUN_ID
verify-vigyan run RUN_ID --evidence RELATIVE_PATH [--timeout SECONDS] -- COMMAND...
verify-vigyan request RUN_ID /PATH --evidence RELATIVE_PATH
verify-vigyan path RUN_ID {state,evidence,db,server-log}
verify-vigyan cleanup RUN_ID
```

Use a new `RUN_ID` for every run. Launch refuses existing state or evidence so old output cannot mix with new proof.
