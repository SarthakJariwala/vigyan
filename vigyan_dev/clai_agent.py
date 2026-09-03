from __future__ import annotations

import os as _os
from collections.abc import Mapping as _Mapping
from dataclasses import dataclass as _dataclass

from pydantic_ai import Agent as _Agent
from pydantic_ai import RunContext as _RunContext

from vigyan.agent import ResearchCapability as _ResearchCapability
from vigyan.corpus import CorpusRetriever as _CorpusRetriever
from vigyan.vectordb import LanceDBVectorStore as _LanceDBVectorStore

_AGENT_INSTRUCTIONS = """\
You are Vigyan, a scientific research assistant. Answer questions from the
configured corpus of scientific papers.
"""
_DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
_DEFAULT_TOP_K = 8


@_dataclass(frozen=True)
class _Settings:
    db_uri: str | None
    embedding_model: str
    top_k: int
    filters: str | None
    embedding_provider: str
    embedding_dim: int | None
    embedding_base_url: str | None
    embedding_api_key_env: str | None


def _environment_value(
    environment: _Mapping[str, str],
    name: str,
) -> str | None:
    return environment.get(name) or None


def _parse_integer(name: str, value: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _settings_from_env(environment: _Mapping[str, str]) -> _Settings:
    top_k = _environment_value(environment, "VIGYAN_TOP_K")
    embedding_dim = _environment_value(environment, "VIGYAN_EMBED_DIM")
    return _Settings(
        db_uri=_environment_value(environment, "VIGYAN_DB_URI"),
        embedding_model=(
            _environment_value(environment, "VIGYAN_EMBED_MODEL")
            or _environment_value(environment, "VIGYAN_EMBEDDING_MODEL")
            or _DEFAULT_EMBEDDING_MODEL
        ),
        top_k=(
            _parse_integer("VIGYAN_TOP_K", top_k)
            if top_k is not None
            else _DEFAULT_TOP_K
        ),
        filters=_environment_value(environment, "VIGYAN_FILTERS"),
        embedding_provider=(
            _environment_value(environment, "VIGYAN_EMBED_PROVIDER") or "openai"
        ),
        embedding_dim=(
            _parse_integer("VIGYAN_EMBED_DIM", embedding_dim)
            if embedding_dim is not None
            else None
        ),
        embedding_base_url=_environment_value(
            environment, "VIGYAN_EMBED_BASE_URL"
        ),
        embedding_api_key_env=_environment_value(
            environment, "VIGYAN_EMBED_API_KEY_ENV"
        ),
    )


def _research_capability_for_run(
    _ctx: _RunContext[None],
) -> _ResearchCapability:
    settings = _settings_from_env(_os.environ)
    store = _LanceDBVectorStore(
        uri=settings.db_uri,
        embedding_model=settings.embedding_model,
        embedding_provider=settings.embedding_provider,
        dim=settings.embedding_dim,
        base_url=settings.embedding_base_url,
        api_key_env=settings.embedding_api_key_env,
    )
    retriever = _CorpusRetriever(store=store)
    return _ResearchCapability(
        retriever,
        default_top_k=settings.top_k,
        default_filters=settings.filters,
    )


agent: _Agent[None, str] = _Agent(
    "anthropic:claude-opus-4-8",
    instructions=_AGENT_INSTRUCTIONS,
    capabilities=[_research_capability_for_run],
    defer_model_check=True,
)

__all__ = ["agent"]
