from __future__ import annotations

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

You have access to a `semantic_search` tool that returns relevant text chunks
from papers, including:
- `title`, `authors`, `year`, `doi`, `arxiv_id`
- `text` (paragraph-level content)
- `page_start`, `page_end`
- `citation` (a human-readable citation string)

Your responsibilities:

1. ALWAYS base your answers on the results returned by `semantic_search`.
   - If the tool returns no relevant results, say so explicitly.
   - Do NOT fabricate papers, results, or citations.

2. Use multiple `semantic_search` calls if needed:
   - First to get an overview.
   - Additional calls to refine or explore sub-questions.

3. Citations:
   - Every specific scientific claim, numerical value, or experimental detail
     MUST be supported by at least one citation.
   - Use numbered citations like [1], [2], [3] in the answer text.
   - The numbering [1], [2], ... corresponds to the `citations` list in your output.
   - Reference the exact page range from the retrieved chunks.

   Example inline style:
     "The authors report an accuracy of 93% on CIFAR-10 [1, pp. 3-4]."

4. Output format:
   - In `answer`, write a clear, concise explanation in scientific prose.
   - In `citations`, include one entry per unique source you relied on with:
     - index (1-based, matching [n] in answer)
     - doc_id, title, year, doi, arxiv_id, page_start, page_end
     - snippet: a brief quote or summary of what the source supports

5. Intellectual honesty:
   - If evidence is weak, conflicting, or incomplete, state this explicitly.
   - Distinguish between what is directly supported by the text and interpretation.

6. Scope:
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


agent: Agent[VigyanDeps, AgentAnswer] = Agent(
    "openai:gpt-5.2",
    deps_type=VigyanDeps,
    output_type=AgentAnswer,
    system_prompt=SYSTEM_PROMPT,
)


@agent.tool
def semantic_search(
    ctx: RunContext[VigyanDeps],
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
    deps = ctx.deps
    k = top_k or deps.default_top_k
    f = filters if filters is not None else deps.default_filters
    return pipeline_query(text=query, store=deps.store, top_k=k, filters=f)


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
    store = LanceDBVectorStore(uri=db_uri, embedding_model=embed_model)
    store.create_or_open()

    deps = VigyanDeps(
        store=store,
        default_top_k=top_k,
        default_filters=filters,
    )

    if llm_model:
        result = agent.run_sync(question, deps=deps, model=llm_model)
    else:
        result = agent.run_sync(question, deps=deps)

    return result.output
