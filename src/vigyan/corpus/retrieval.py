from __future__ import annotations

from dataclasses import dataclass

from ..interfaces import VectorStore
from ..models import QueryHit


@dataclass
class CorpusRetriever:
    """Retrieve citation-ready evidence from a Corpus."""

    store: VectorStore

    def retrieve(
        self,
        text: str,
        top_k: int = 8,
        filters: str | None = None,
    ) -> list[QueryHit]:
        """Search indexed Corpus content for text related to the input text."""
        self.store.create_or_open()
        return self.store.search(text, top_k=top_k, filters=filters)
