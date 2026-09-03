from __future__ import annotations

from typing import Protocol, final

from pydantic_ai import Tool
from pydantic_ai.capabilities import Capability

from ..models import QueryHit

_CAPABILITY_ID = "vigyan-research"
_CAPABILITY_DESCRIPTION = (
    "Search ingested scientific papers and cite page-level evidence."
)
_RESEARCH_INSTRUCTIONS = """\
Always base research answers on results from `semantic_search`.
If the tool returns no relevant results, say so. Do not fabricate papers,
results, or citations.

Citations and support levels:
- Every specific scientific claim, numerical value, or experimental detail
  must cite at least one retrieved chunk.
- Use numbered citations such as [1], [2], and [3] in the answer text.
- Include the exact page range from each retrieved chunk.
- Inspect each QueryHit's `cited_ref_ids` and `cited_references` fields.
  If a retrieved chunk cites another paper for a claim, treat the chunk as
  secondary support rather than primary evidence.
- If a cited reference has `in_corpus=True`, run another `semantic_search`
  scoped to `doc_id = '<resolved_doc_id>'` before presenting that paper as
  verified primary evidence.
- If a cited reference has `in_corpus=False`, name the referenced paper or
  DOI when available. State that the corpus verifies the citation but does
  not contain the primary source.

Use this inline citation style:
"The authors report an accuracy of 93% on CIFAR-10 [1, pp. 3-4]."

State when evidence is weak, conflicting, incomplete, or outside the corpus.
Distinguish direct support from interpretation. Prefer direct quotes or close
paraphrases for key numerical results.
"""


class ResearchRetriever(Protocol):
    """The retrieval operation required by `ResearchCapability`."""

    def retrieve(
        self,
        text: str,
        top_k: int = 8,
        filters: str | None = None,
    ) -> list[QueryHit]: ...


def _resolve_search_args(
    *,
    top_k: int | None,
    filters: str | None,
    default_top_k: int,
    default_filters: str | None,
) -> tuple[int, str | None]:
    return (
        top_k or default_top_k,
        filters if filters is not None else default_filters,
    )


@final
class ResearchCapability(Capability[object]):
    """Install citation-grounded corpus research on a Pydantic AI agent."""

    def __init__(
        self,
        retriever: ResearchRetriever,
        *,
        default_top_k: int = 8,
        default_filters: str | None = None,
        defer_loading: bool = False,
    ) -> None:
        self._retriever = retriever
        self._default_top_k = default_top_k
        self._default_filters = default_filters
        super().__init__(
            id=_CAPABILITY_ID,
            description=_CAPABILITY_DESCRIPTION,
            defer_loading=defer_loading,
            instructions=_RESEARCH_INSTRUCTIONS,
            tools=[Tool(self._semantic_search, name="semantic_search")],
        )

    def _semantic_search(
        self,
        query: str,
        top_k: int | None = None,
        filters: str | None = None,
    ) -> list[QueryHit]:
        """Search the scientific paper corpus for citation-ready evidence.

        Args:
            query: Natural-language text that describes the requested evidence.
            top_k: Maximum number of chunks to return. Uses the configured default when omitted.
            filters: Optional filter expression for documents or chunks.

        Returns:
            Citation-ready corpus hits with page spans and source metadata.
        """
        effective_top_k, effective_filters = _resolve_search_args(
            top_k=top_k,
            filters=filters,
            default_top_k=self._default_top_k,
            default_filters=self._default_filters,
        )
        return self._retriever.retrieve(
            text=query,
            top_k=effective_top_k,
            filters=effective_filters,
        )
