# Vigyan

Vigyan parses scientific PDFs, indexes them in a vector store, and adds citation-grounded corpus research to Pydantic AI agents.

## Install Vigyan

Vigyan requires Python 3.12 or later. Install the package with your project package manager.

```bash
uv add vigyan
```

GROBID must be running when you parse PDFs. Your embedding provider credentials must be available when you create or query a LanceDB store.

## Ingest and query one corpus

Configure one `Corpus` with a parser and a store. The corpus builds its ingestor and retriever over those same components.

```python
from pathlib import Path

from vigyan.corpus import Corpus
from vigyan.parsers import GrobidParser
from vigyan.vectordb import LanceDBVectorStore

store = LanceDBVectorStore(
    uri="./vigyan_db",
    embedding_model="text-embedding-3-small",
)
parser = GrobidParser(server_url="http://localhost:8070")

corpus = Corpus(parser=parser, store=store)
corpus.ingestor.ingest_pdf(Path("paper.pdf").read_bytes())

retriever = corpus.retriever
for hit in retriever.retrieve("protein folding with attention", top_k=5):
    print(hit.citation)
    print(hit.text)
```

`DocumentParser` and `VectorStore` are protocols. You can replace GROBID or LanceDB without changing `Corpus`. `CorpusIngestor` and `CorpusRetriever` remain available for callers that need to compose those parts separately.

## Add research to an existing agent

`ResearchCapability` installs the citation instructions and `semantic_search` tool as one unit. It closes over the retriever, so the host agent keeps its own dependency type.

```python
from dataclasses import dataclass

from pydantic_ai import Agent

from vigyan.agent import ResearchCapability


@dataclass
class AppDeps:
    project_id: str


assistant = Agent(
    "anthropic:claude-opus-4-8",
    deps_type=AppDeps,
    instructions="Answer questions for the current research project.",
    capabilities=[ResearchCapability(retriever)],
)

result = assistant.run_sync(
    "What accuracy did the indexed papers report?",
    deps=AppDeps(project_id="protein-folding"),
)
print(result.output)
```

Research loads eagerly by default. In a general-purpose agent that rarely needs the corpus, use `ResearchCapability(retriever, defer_loading=True)`. Pydantic AI then loads the research instructions and tool together when the model selects the capability.

The constructor accepts any object that implements `ResearchRetriever`. A custom retriever needs this method:

```python
from vigyan.models import QueryHit


class MyRetriever:
    def retrieve(
        self,
        text: str,
        top_k: int = 8,
        filters: str | None = None,
    ) -> list[QueryHit]:
        ...
```

## Use the compatibility research agent

`run_research_query()` keeps the original convenience API. It creates a LanceDB-backed retriever and returns the model's answer as a string.

```python
from vigyan.agent import run_research_query

answer = run_research_query(
    "What does this corpus say about protein folding with attention?",
    db_uri="./vigyan_db",
    embed_model="text-embedding-3-small",
)
print(answer)
```

## Run the CLai web agent

CLai cannot pass Pydantic AI dependencies to an imported agent. The compatibility agent reads its retriever settings from the environment when CLai supplies `deps=None`.

```bash
export VIGYAN_DB_URI=./vigyan_db
export VIGYAN_EMBED_MODEL=text-embedding-3-small
export VIGYAN_TOP_K=8
export VIGYAN_FILTERS="year >= 2020"

uv run clai web --agent src.vigyan.agent.research_agent:agent
```

Explicit `ResearchAgentDeps` still take precedence when you call the compatibility agent from Python.
