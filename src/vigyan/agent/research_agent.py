from __future__ import annotations

import os
from dataclasses import dataclass

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

from ..core.interfaces import VectorStore
from ..core.models import QueryHit
from ..pipeline import query as pipeline_query
from ..vectordb.lancedb_store import LanceDBVectorStore

SYSTEM_PROMPT = """\
You are Vigyan, a scientific research assistant helping researchers answer
questions based on a corpus of scientific papers indexed in a vector store.

Your responsibilities:

1. ALWAYS base your answers on the results returned by `semantic_search`.
   - If the tool returns no relevant results, say so explicitly.
   - Do NOT fabricate papers, results, or citations.

2. Citations:
   - Every specific scientific claim, numerical value, or experimental detail
     MUST be supported by at least one citation.
   - Use numbered citations like [1], [2], [3] in the answer text.
   - The numbering [1], [2], ... corresponds to the `citations` list in your output.
   - Reference the exact page range from the retrieved chunks.

   Example inline style:
     "The authors report an accuracy of 93% on CIFAR-10 [1, pp. 3-4]."

3. Intellectual honesty:
   - If evidence is weak, conflicting, or incomplete, state this explicitly.
   - Distinguish between what is directly supported by the text and interpretation.

4. Scope:
   - Prefer direct quotes or close paraphrases for key numerical results.
   - If asked about something outside the corpus, state that limitation.
"""


@dataclass
class VigyanDeps:
    store: VectorStore
    default_top_k: int = 8
    default_filters: str | None = None


class Citation(BaseModel):
    index: int
    doc_id: str
    title: str
    year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    page_start: int
    page_end: int
    snippet: str
    citation: str


class AgentAnswer(BaseModel):
    answer: str
    citations: list[Citation]


DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_TOP_K = 8


def _env_value(name: str) -> str | None:
    value = os.getenv(name)
    return value or None


def _env_int(name: str, default: int) -> int:
    value = _env_value(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def build_deps_from_env() -> VigyanDeps:
    """Create dependencies for CLai/web runs that cannot pass deps explicitly."""
    return build_deps(
        db_uri=_env_value("VIGYAN_DB_URI"),
        embed_model=(
            _env_value("VIGYAN_EMBED_MODEL")
            or _env_value("VIGYAN_EMBEDDING_MODEL")
            or DEFAULT_EMBED_MODEL
        ),
        top_k=_env_int("VIGYAN_TOP_K", DEFAULT_TOP_K),
        filters=_env_value("VIGYAN_FILTERS"),
        embedding_provider=_env_value("VIGYAN_EMBED_PROVIDER") or "openai",
        dim=(
            _env_int("VIGYAN_EMBED_DIM", 0)
            if _env_value("VIGYAN_EMBED_DIM") is not None
            else None
        ),
        base_url=_env_value("VIGYAN_EMBED_BASE_URL"),
        api_key_env=_env_value("VIGYAN_EMBED_API_KEY_ENV"),
    )


def resolve_deps(deps: VigyanDeps | None) -> VigyanDeps:
    """Prefer explicit SDK deps; fall back to env-backed deps for CLai/web."""
    return deps if deps is not None else build_deps_from_env()


agent: Agent[VigyanDeps | None, AgentAnswer] = Agent(
    "anthropic:claude-opus-4-8",
    deps_type=VigyanDeps,
    system_prompt=SYSTEM_PROMPT,
    defer_model_check=True,
)


@agent.tool
def semantic_search(
    ctx: RunContext[VigyanDeps | None],
    query: str,
    top_k: int | None = None,
    filters: str | None = None,
) -> list[QueryHit]:
    """Search the scientific paper vector store for text related to the query.

    Args:
        ctx: The run context with dependencies
        query: Natural-language query describing what to look for
        top_k: Maximum number of chunks to return (uses default if omitted)
        filters: Optional filter expression to restrict documents/chunks

    Returns:
        A list of QueryHit objects with relevant chunks including page_span
        and formatted citation strings.
    """
    deps = resolve_deps(ctx.deps)
    k = top_k or deps.default_top_k
    f = filters if filters is not None else deps.default_filters
    return pipeline_query(text=query, store=deps.store, top_k=k, filters=f)


def build_deps(
    *,
    db_uri: str | None = None,
    embed_model: str = DEFAULT_EMBED_MODEL,
    top_k: int = DEFAULT_TOP_K,
    filters: str | None = None,
    embedding_provider: str = "openai",
    dim: int | None = None,
    base_url: str | None = None,
    api_key_env: str | None = None,
) -> VigyanDeps:
    """Create Vigyan dependencies for an agent run."""
    store = LanceDBVectorStore(
        uri=db_uri,
        embedding_model=embed_model,
        embedding_provider=embedding_provider,
        dim=dim,
        base_url=base_url,
        api_key_env=api_key_env,
    )
    store.create_or_open()
    return VigyanDeps(
        store=store,
        default_top_k=top_k,
        default_filters=filters,
    )


def run_research_query(
    question: str,
    *,
    db_uri: str,
    embed_model: str,
    top_k: int = 8,
    filters: str | None = None,
    llm_model: str | None = None,
) -> AgentAnswer:
    """Run the Vigyan research agent to answer a scientific question.

    Args:
        question: The user's scientific question
        db_uri: LanceDB URI
        embed_model: OpenAI embedding model name
        top_k: Number of results to retrieve per search
        filters: Optional filter expression
        llm_model: Optional LLM model override

    Returns:
        AgentAnswer with answer text and structured citations
    """
    deps = build_deps(
        db_uri=db_uri,
        embed_model=embed_model,
        top_k=top_k,
        filters=filters,
    )

    if llm_model:
        result = agent.run_sync(question, deps=deps, model=llm_model)
    else:
        result = agent.run_sync(question, deps=deps)

    return result.output
