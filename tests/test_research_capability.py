from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent, ModelResponse, RunContext
from pydantic_ai.capabilities import Capability
from pydantic_ai.messages import (
    LoadCapabilityReturnPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from vigyan.agent import ResearchCapability
from vigyan.models import QueryHit


def _query_hit() -> QueryHit:
    return QueryHit(
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


class RecordingRetriever:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def retrieve(
        self,
        text: str,
        top_k: int = 8,
        filters: str | None = None,
    ) -> list[QueryHit]:
        self.calls.append({"text": text, "top_k": top_k, "filters": filters})
        return [_query_hit()]


def _tool_returns(messages: list[Any], name: str) -> list[ToolReturnPart]:
    return [
        part
        for message in messages
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == name
    ]


@dataclass
class HostDeps:
    tenant: str


def test_capability_composes_with_host_deps_and_executes_search() -> None:
    retriever = RecordingRetriever()
    host_capability = Capability[HostDeps]()

    @host_capability.tool
    def tenant_name(ctx: RunContext[HostDeps]) -> str:
        return ctx.deps.tenant

    request_count = 0

    def respond(messages: list[Any], info: AgentInfo) -> ModelResponse:
        nonlocal request_count
        request_count += 1
        if request_count == 1:
            assert {tool.name for tool in info.function_tools} == {
                "semantic_search",
                "tenant_name",
            }
            assert info.instructions is not None
            normalized_instructions = " ".join(info.instructions.split())
            assert "Answer for the current tenant." in info.instructions
            assert "Every specific scientific claim" in info.instructions
            assert "in_corpus=True" in info.instructions
            assert "doc_id = '<resolved_doc_id>'" in info.instructions
            assert "does not contain the primary source" in normalized_instructions
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="semantic_search",
                        args={"query": "attention mechanisms"},
                    ),
                    ToolCallPart(tool_name="tenant_name", args={}),
                ]
            )

        research_returns = _tool_returns(messages, "semantic_search")
        tenant_returns = _tool_returns(messages, "tenant_name")
        research_content = research_returns[0].content
        assert isinstance(research_content, list)
        assert isinstance(research_content[0], QueryHit)
        assert research_content[0].doc_id == "doc-1"
        assert tenant_returns[0].content == "acme"
        return ModelResponse(parts=[TextPart(content="Grounded answer [1, pp. 2-3].")])

    agent = Agent(
        FunctionModel(respond),
        deps_type=HostDeps,
        instructions="Answer for the current tenant.",
        capabilities=[
            ResearchCapability(
                retriever,
                default_top_k=4,
                default_filters="year >= 2020",
            ),
            host_capability,
        ],
    )

    result = agent.run_sync("What did the paper report?", deps=HostDeps("acme"))

    assert result.output == "Grounded answer [1, pp. 2-3]."
    assert retriever.calls == [
        {
            "text": "attention mechanisms",
            "top_k": 4,
            "filters": "year >= 2020",
        }
    ]


def test_search_arguments_override_defaults_and_preserve_zero_behavior() -> None:
    retriever = RecordingRetriever()
    call_args = {
        "query": "protein folding",
        "top_k": 0,
        "filters": "doi = '10.1234/example'",
    }

    def respond(messages: list[Any], info: AgentInfo) -> ModelResponse:
        if not _tool_returns(messages, "semantic_search"):
            return ModelResponse(
                parts=[ToolCallPart(tool_name="semantic_search", args=call_args)]
            )
        return ModelResponse(parts=[TextPart(content="done")])

    agent = Agent(
        FunctionModel(respond),
        capabilities=[
            ResearchCapability(
                retriever,
                default_top_k=6,
                default_filters="year >= 2020",
            )
        ],
    )

    agent.run_sync("Search")

    assert retriever.calls == [
        {
            "text": "protein folding",
            "top_k": 6,
            "filters": "doi = '10.1234/example'",
        }
    ]


def test_deferred_capability_loads_tools_and_instructions_together() -> None:
    retriever = RecordingRetriever()
    tool_snapshots: list[dict[str, bool]] = []
    loaded_instructions: list[str] = []

    def respond(messages: list[Any], info: AgentInfo) -> ModelResponse:
        tool_snapshots.append(
            {tool.name: tool.defer_loading for tool in info.function_tools}
        )
        load_returns = [
            part
            for message in messages
            for part in message.parts
            if isinstance(part, LoadCapabilityReturnPart)
        ]
        for part in load_returns:
            instructions = part.content.get("instructions")
            if isinstance(instructions, str):
                loaded_instructions.append(instructions)

        if not load_returns:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="load_capability",
                        args={"id": "vigyan-research"},
                    )
                ]
            )
        if not _tool_returns(messages, "semantic_search"):
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="semantic_search",
                        args={"query": "stars"},
                    )
                ]
            )
        return ModelResponse(parts=[TextPart(content="done")])

    agent = Agent(
        FunctionModel(respond),
        capabilities=[ResearchCapability(retriever, defer_loading=True)],
    )

    result = agent.run_sync("Use the paper corpus")

    assert result.output == "done"
    assert tool_snapshots[0]["semantic_search"] is True
    assert any(snapshot["semantic_search"] is False for snapshot in tool_snapshots[1:])
    assert any(
        "Every specific scientific claim" in instructions
        for instructions in loaded_instructions
    )
    assert retriever.calls == [{"text": "stars", "top_k": 8, "filters": None}]
