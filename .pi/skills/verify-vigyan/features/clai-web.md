# Contributor CLai web

Contributors can run the checkout-only agent with this command:

```bash
uv run clai web --agent vigyan_dev.clai_agent:agent
```

`verify-vigyan launch` runs that entry point with `--no-sync`, a randomly selected loopback port, and a run-specific local HTML marker. The extra flags reduce port collision risk and avoid network access for UI assets without changing the imported agent. Readiness fails if the selected port serves any page other than this run's marker.

## Recipe

```bash
"$VERIFY_VIGYAN" doctor "$RUN_ID"
"$VERIFY_VIGYAN" request "$RUN_ID" /api/health \
  --evidence clai-web/health.txt
"$VERIFY_VIGYAN" request "$RUN_ID" /api/configure \
  --evidence clai-web/configure.txt
"$VERIFY_VIGYAN" request "$RUN_ID" / \
  --evidence clai-web/root.txt
```

Expected observations:

- `/api/health` returns HTTP 200 and `{"ok":true}`.
- `/api/configure` returns HTTP 200 and identifies `anthropic:claude-opus-4-8`.
- `/` returns HTTP 200 and contains the marker for this `RUN_ID`.
- `server.log` shows Uvicorn bound to the recorded loopback port without an import error.

## Claim boundary

This recipe proves that the locked development tools import the checkout-only agent and serve its HTTP shell. It does not call `/api/chat`, invoke `semantic_search`, open LanceDB, create embeddings, contact Anthropic, or validate answer citations.

A chat proof must capture the request, streamed response, retrieved evidence, and resulting citations. Follow [external-services.md](external-services.md) before making that request.
