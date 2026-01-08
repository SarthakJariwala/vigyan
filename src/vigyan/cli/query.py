from __future__ import annotations

from typing import Annotated

from cyclopts import Parameter

from ..agent import AgentAnswer, run_research_query
from ..vectordb.lancedb_store import default_lancedb_path
from .app import app


def _print_human_readable(answer: AgentAnswer) -> None:
    print("\n=== Answer ===\n")
    print(answer.answer.strip())
    print("\n=== Citations ===\n")
    for cit in answer.citations:
        pages = f"pp. {cit.page_start}-{cit.page_end}"
        doi = f", doi={cit.doi}" if cit.doi else ""
        arxiv = f", arxiv={cit.arxiv_id}" if cit.arxiv_id else ""
        print(f"[{cit.index}] {cit.citation} ({pages}{doi}{arxiv})")
        print(f"    -> {cit.snippet}\n")


@app.command
def query(
    q: Annotated[str, Parameter(help="Your scientific question")],
    *,
    db: Annotated[str, Parameter(help="LanceDB URI")] = default_lancedb_path(),
    top_k: Annotated[int, Parameter(help="Number of results to retrieve")] = 8,
    filter: Annotated[str | None, Parameter(help="Filter expression")] = None,
    embed_model: Annotated[
        str, Parameter(help="OpenAI embedding model")
    ] = "text-embedding-3-large",
    llm_model: Annotated[str | None, Parameter(help="LLM model for the agent")] = None,
    json_output: Annotated[bool, Parameter(help="Print JSON instead of text")] = False,
) -> None:
    """Ask a scientific question over the indexed corpus using the Vigyan agent."""
    result = run_research_query(
        q,
        db_uri=db,
        embed_model=embed_model,
        top_k=top_k,
        filters=filter,
        llm_model=llm_model,
    )

    if json_output:
        print(result.model_dump_json(indent=2))
    else:
        _print_human_readable(result)
