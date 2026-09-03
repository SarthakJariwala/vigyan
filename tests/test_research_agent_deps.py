from __future__ import annotations

from typing import Any

from pydantic_ai import Agent, ModelResponse
from pydantic_ai.messages import TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from vigyan.agent import research_agent
from vigyan.models import Chunk, Document, QueryHit


class FakeVectorStore:
    model_name = "fake-embedding-model"
    dim = 3

    def __init__(self) -> None:
        self.open_count = 0
        self.search_calls: list[dict[str, Any]] = []

    def create_or_open(self) -> None:
        self.open_count += 1

    def upsert_documents(self, docs: list[Document]) -> None:
        raise AssertionError("test store should not ingest documents")

    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        raise AssertionError("test store should not ingest chunks")

    def search(
        self,
        query: str,
        top_k: int = 8,
        filters: str | None = None,
    ) -> list[QueryHit]:
        self.search_calls.append({"query": query, "top_k": top_k, "filters": filters})
        return [
            QueryHit(
                doc_id="doc-1",
                title="A Test Paper",
                year=2024,
                doi="10.1234/example",
                arxiv_id=None,
                page_span=(2, 3),
                text="Relevant evidence from the paper.",
                citation="Tester et al. A Test Paper (2024), pp. 2-3",
                distance=0.1,
            )
        ]


class FakeCorpusRetriever:
    def __init__(self) -> None:
        self.retrieve_calls: list[dict[str, Any]] = []

    def retrieve(
        self,
        text: str,
        top_k: int = 8,
        filters: str | None = None,
    ) -> list[QueryHit]:
        self.retrieve_calls.append({"text": text, "top_k": top_k, "filters": filters})
        return [
            QueryHit(
                doc_id="doc-1",
                title="A Test Paper",
                year=2024,
                doi="10.1234/example",
                arxiv_id=None,
                page_span=(2, 3),
                text="Relevant evidence from the paper.",
                citation="Tester et al. A Test Paper (2024), pp. 2-3",
                distance=0.1,
            )
        ]


def _search_model(**tool_args: Any) -> FunctionModel:
    def respond(messages: list[Any], info: AgentInfo) -> ModelResponse:
        search_returns = [
            part
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart)
            and part.tool_name == "semantic_search"
        ]
        if not search_returns:
            assert "semantic_search" in {tool.name for tool in info.function_tools}
            assert info.instructions is not None
            assert "You are Vigyan" in info.instructions
            assert "Every specific scientific claim" in info.instructions
            return ModelResponse(
                parts=[ToolCallPart(tool_name="semantic_search", args=tool_args)]
            )
        return ModelResponse(parts=[TextPart(content="answer [1, pp. 2-3]")])

    return FunctionModel(respond)


def test_agent_package_exports_agent_and_deps_helpers() -> None:
    from vigyan.agent import (
        ResearchCapability,
        ResearchRetriever,
        agent,
        build_deps,
        build_deps_from_env,
    )

    assert isinstance(agent, Agent)
    assert agent is research_agent.agent
    assert build_deps is research_agent.build_deps
    assert build_deps_from_env is research_agent.build_deps_from_env


def test_clai_agent_path_loads_the_compatibility_agent() -> None:
    from pydantic_ai._cli import load_agent

    loaded_agent = load_agent("src.vigyan.agent.research_agent:agent")

    assert isinstance(loaded_agent, Agent)


def test_agent_builds_default_deps_when_clai_provides_none(
    monkeypatch,
) -> None:
    fake_store = FakeVectorStore()
    store_kwargs: dict[str, Any] = {}

    def fake_store_factory(**kwargs: Any) -> FakeVectorStore:
        store_kwargs.update(kwargs)
        return fake_store

    monkeypatch.setenv("VIGYAN_DB_URI", "/tmp/vigyan-test-db")
    monkeypatch.setenv("VIGYAN_EMBED_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("VIGYAN_TOP_K", "4")
    monkeypatch.setenv("VIGYAN_FILTERS", "year >= 2020")
    monkeypatch.setattr(research_agent, "LanceDBVectorStore", fake_store_factory)

    with research_agent.agent.override(model=_search_model(query="protein folding")):
        result = research_agent.agent.run_sync("Research protein folding", deps=None)

    assert result.output == "answer [1, pp. 2-3]"
    assert store_kwargs["uri"] == "/tmp/vigyan-test-db"
    assert store_kwargs["embedding_model"] == "text-embedding-3-small"
    assert fake_store.open_count == 1
    assert fake_store.search_calls == [
        {"query": "protein folding", "top_k": 4, "filters": "year >= 2020"}
    ]


def test_agent_prefers_explicit_deps_and_tool_arguments(
    monkeypatch,
) -> None:
    unused_default_store = FakeVectorStore()
    explicit_retriever = FakeCorpusRetriever()

    monkeypatch.setattr(
        research_agent,
        "LanceDBVectorStore",
        lambda **kwargs: unused_default_store,
    )

    deps = research_agent.ResearchAgentDeps(
        retriever=explicit_retriever,
        default_top_k=3,
        default_filters="year >= 2020",
    )

    with research_agent.agent.override(
        model=_search_model(
            query="attention mechanisms",
            top_k=2,
            filters="doi = '10.1234/example'",
        )
    ):
        research_agent.agent.run_sync("Research attention", deps=deps)

    assert explicit_retriever.retrieve_calls == [
        {
            "text": "attention mechanisms",
            "top_k": 2,
            "filters": "doi = '10.1234/example'",
        }
    ]
    assert unused_default_store.search_calls == []


def test_agent_builds_a_fresh_capability_for_each_run() -> None:
    first = FakeCorpusRetriever()
    second = FakeCorpusRetriever()

    with research_agent.agent.override(model=_search_model(query="first")):
        research_agent.agent.run_sync(
            "First run", deps=research_agent.ResearchAgentDeps(first)
        )
    with research_agent.agent.override(model=_search_model(query="second")):
        research_agent.agent.run_sync(
            "Second run", deps=research_agent.ResearchAgentDeps(second)
        )

    assert first.retrieve_calls == [{"text": "first", "top_k": 8, "filters": None}]
    assert second.retrieve_calls == [
        {"text": "second", "top_k": 8, "filters": None}
    ]


def test_run_research_query_returns_the_compatibility_agent_output(
    monkeypatch,
) -> None:
    retriever = FakeCorpusRetriever()
    build_kwargs: dict[str, Any] = {}

    def fake_build_deps(**kwargs: Any) -> research_agent.ResearchAgentDeps:
        build_kwargs.update(kwargs)
        return research_agent.ResearchAgentDeps(
            retriever,
            default_top_k=kwargs["top_k"],
            default_filters=kwargs["filters"],
        )

    monkeypatch.setattr(research_agent, "build_deps", fake_build_deps)

    with research_agent.agent.override(model=_search_model(query="wrapper")):
        answer = research_agent.run_research_query(
            "Research through the wrapper",
            db_uri="/tmp/vigyan-test-db",
            embed_model="text-embedding-3-small",
            top_k=5,
            filters="year >= 2020",
        )

    assert answer == "answer [1, pp. 2-3]"
    assert build_kwargs == {
        "db_uri": "/tmp/vigyan-test-db",
        "embed_model": "text-embedding-3-small",
        "top_k": 5,
        "filters": "year >= 2020",
    }
    assert retriever.retrieve_calls == [
        {"text": "wrapper", "top_k": 5, "filters": "year >= 2020"}
    ]


def test_agent_can_run_with_test_model_without_explicit_deps(monkeypatch) -> None:
    fake_store = FakeVectorStore()

    monkeypatch.setenv("VIGYAN_DB_URI", "/tmp/vigyan-test-db")
    monkeypatch.setenv("VIGYAN_EMBED_MODEL", "text-embedding-3-small")
    monkeypatch.setattr(
        research_agent,
        "LanceDBVectorStore",
        lambda **kwargs: fake_store,
    )

    with research_agent.agent.override(model=TestModel()):
        result = research_agent.agent.run_sync(
            "Summarize the indexed papers", deps=None
        )

    assert fake_store.search_calls
