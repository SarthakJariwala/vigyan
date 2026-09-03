from unittest.mock import Mock

from vigyan.corpus import Corpus
from vigyan.interfaces import DocumentParser, VectorStore


def test_corpus_builds_ingestion_and_retrieval_from_shared_components() -> None:
    parser = Mock(spec=DocumentParser)
    store = Mock(spec=VectorStore)

    corpus = Corpus(parser=parser, store=store)

    assert corpus.ingestor.parser is parser
    assert corpus.ingestor.store is store
    assert corpus.retriever.store is store
