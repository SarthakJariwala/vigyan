from __future__ import annotations

import os
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext

from ..corpus import CorpusRetriever
from ..vectordb import LanceDBVectorStore
from .capability import ResearchCapability, ResearchRetriever

_AGENT_INSTRUCTIONS = """\
You are Vigyan, a scientific research assistant. Answer questions from the
configured corpus of scientific papers.
"""


@dataclass
class ResearchAgentDeps:
    retriever: ResearchRetriever
    default_top_k: int = 8
    default_filters: str | None = None


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


def build_deps_from_env() -> ResearchAgentDeps:
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


def resolve_deps(deps: ResearchAgentDeps | None) -> ResearchAgentDeps:
    """Prefer explicit SDK deps; fall back to env-backed deps for CLai/web."""
    return deps if deps is not None else build_deps_from_env()


def _research_capability_for_run(
    ctx: RunContext[ResearchAgentDeps | None],
) -> ResearchCapability:
    deps = resolve_deps(ctx.deps)
    return ResearchCapability(
        deps.retriever,
        default_top_k=deps.default_top_k,
        default_filters=deps.default_filters,
    )


agent: Agent[ResearchAgentDeps | None] = Agent(
    "anthropic:claude-opus-4-8",
    deps_type=ResearchAgentDeps,
    instructions=_AGENT_INSTRUCTIONS,
    capabilities=[_research_capability_for_run],
    defer_model_check=True,
)


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
) -> ResearchAgentDeps:
    """Create Vigyan dependencies for an agent run."""
    store = LanceDBVectorStore(
        uri=db_uri,
        embedding_model=embed_model,
        embedding_provider=embedding_provider,
        dim=dim,
        base_url=base_url,
        api_key_env=api_key_env,
    )
    retriever = CorpusRetriever(store=store)
    return ResearchAgentDeps(
        retriever=retriever,
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
) -> str:
    """Run the Vigyan research agent to answer a scientific question.

    Args:
        question: The user's scientific question
        db_uri: LanceDB URI
        embed_model: OpenAI embedding model name
        top_k: Number of results to retrieve per search
        filters: Optional filter expression
        llm_model: Optional LLM model override

    Returns:
        str with answer text and citations
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
