from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import pytest
from pydantic_ai import Agent, ModelResponse
from pydantic_ai.messages import TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from vigyan.models import QueryHit


class FakeVectorStore:
    model_name = "fake-embedding-model"
    dim = 3

    def __init__(self) -> None:
        self.open_count = 0
        self.search_calls: list[dict[str, Any]] = []

    def create_or_open(self) -> None:
        self.open_count += 1

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


def _search_model(query: str) -> FunctionModel:
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
                parts=[ToolCallPart(tool_name="semantic_search", args={"query": query})]
            )
        return ModelResponse(parts=[TextPart(content="answer [1, pp. 2-3]")])

    return FunctionModel(respond)


def test_clai_module_exports_only_an_agent() -> None:
    from vigyan_dev import clai_agent

    assert clai_agent.__all__ == ["agent"]
    assert isinstance(clai_agent.agent, Agent)
    assert clai_agent.agent.deps_type is type(None)
    assert {
        "ResearchAgentDeps",
        "build_deps",
        "build_deps_from_env",
        "run_research_query",
    }.isdisjoint(vars(clai_agent))


def test_clai_loads_the_checkout_agent() -> None:
    from pydantic_ai._cli import load_agent

    loaded_agent = load_agent("vigyan_dev.clai_agent:agent")

    assert isinstance(loaded_agent, Agent)


def test_settings_parse_an_explicit_environment_mapping() -> None:
    from vigyan_dev import clai_agent

    settings = clai_agent._settings_from_env(
        {
            "VIGYAN_DB_URI": "/tmp/vigyan-test-db",
            "VIGYAN_EMBED_MODEL": "text-embedding-3-large",
            "VIGYAN_TOP_K": "4",
            "VIGYAN_FILTERS": "year >= 2020",
            "VIGYAN_EMBED_PROVIDER": "openai",
            "VIGYAN_EMBED_DIM": "3072",
            "VIGYAN_EMBED_BASE_URL": "https://embeddings.example.test/v1",
            "VIGYAN_EMBED_API_KEY_ENV": "TEST_EMBED_API_KEY",
        }
    )

    assert settings == clai_agent._Settings(
        db_uri="/tmp/vigyan-test-db",
        embedding_model="text-embedding-3-large",
        top_k=4,
        filters="year >= 2020",
        embedding_provider="openai",
        embedding_dim=3072,
        embedding_base_url="https://embeddings.example.test/v1",
        embedding_api_key_env="TEST_EMBED_API_KEY",
    )
    with pytest.raises(FrozenInstanceError):
        settings.top_k = 2


def test_settings_accept_legacy_embedding_model_name() -> None:
    from vigyan_dev import clai_agent

    settings = clai_agent._settings_from_env(
        {"VIGYAN_EMBEDDING_MODEL": "text-embedding-legacy"}
    )

    assert settings.embedding_model == "text-embedding-legacy"


@pytest.mark.parametrize("name", ["VIGYAN_TOP_K", "VIGYAN_EMBED_DIM"])
def test_settings_reject_non_integer_values(name: str) -> None:
    from vigyan_dev import clai_agent

    with pytest.raises(ValueError, match=rf"{name} must be an integer"):
        clai_agent._settings_from_env({name: "many"})


def test_agent_builds_fresh_research_objects_for_each_run(monkeypatch) -> None:
    from vigyan_dev import clai_agent

    stores: list[FakeVectorStore] = []
    retrievers: list[Any] = []
    capabilities: list[Any] = []
    store_kwargs: list[dict[str, Any]] = []
    real_retriever = clai_agent._CorpusRetriever
    real_capability = clai_agent._ResearchCapability

    def fake_store_factory(**kwargs: Any) -> FakeVectorStore:
        store = FakeVectorStore()
        stores.append(store)
        store_kwargs.append(kwargs)
        return store

    def recording_retriever(**kwargs: Any) -> Any:
        retriever = real_retriever(**kwargs)
        retrievers.append(retriever)
        return retriever

    def recording_capability(*args: Any, **kwargs: Any) -> Any:
        capability = real_capability(*args, **kwargs)
        capabilities.append(capability)
        return capability

    monkeypatch.setenv("VIGYAN_DB_URI", "/tmp/vigyan-test-db")
    monkeypatch.setenv("VIGYAN_EMBED_MODEL", "text-embedding-test")
    monkeypatch.setenv("VIGYAN_TOP_K", "4")
    monkeypatch.setenv("VIGYAN_FILTERS", "year >= 2020")
    monkeypatch.setattr(clai_agent, "_LanceDBVectorStore", fake_store_factory)
    monkeypatch.setattr(clai_agent, "_CorpusRetriever", recording_retriever)
    monkeypatch.setattr(clai_agent, "_ResearchCapability", recording_capability)

    with clai_agent.agent.override(model=_search_model("first query")):
        first_result = clai_agent.agent.run_sync("First run")
    monkeypatch.setenv("VIGYAN_TOP_K", "2")
    monkeypatch.setenv("VIGYAN_FILTERS", "year >= 2024")
    with clai_agent.agent.override(model=_search_model("second query")):
        second_result = clai_agent.agent.run_sync("Second run")

    assert first_result.output == "answer [1, pp. 2-3]"
    assert second_result.output == "answer [1, pp. 2-3]"
    assert len(stores) == 2
    assert stores[0] is not stores[1]
    assert len(retrievers) == 2
    assert retrievers[0] is not retrievers[1]
    assert len(capabilities) == 2
    assert capabilities[0] is not capabilities[1]
    assert store_kwargs == [
        {
            "uri": "/tmp/vigyan-test-db",
            "embedding_model": "text-embedding-test",
            "embedding_provider": "openai",
            "dim": None,
            "base_url": None,
            "api_key_env": None,
        },
        {
            "uri": "/tmp/vigyan-test-db",
            "embedding_model": "text-embedding-test",
            "embedding_provider": "openai",
            "dim": None,
            "base_url": None,
            "api_key_env": None,
        },
    ]
    assert stores[0].open_count == 1
    assert stores[1].open_count == 1
    assert stores[0].search_calls == [
        {"query": "first query", "top_k": 4, "filters": "year >= 2020"}
    ]
    assert stores[1].search_calls == [
        {"query": "second query", "top_k": 2, "filters": "year >= 2024"}
    ]
