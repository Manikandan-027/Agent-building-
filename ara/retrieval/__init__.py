"""Retrieval package: ingestion, engines, vector stores, colpali client."""
from ara.retrieval.colpali_client import ColPaliClient, InProcessMockColPali, make_colpali
from ara.retrieval.engine import RetrievalEngine, RetrievalResult
from ara.retrieval.ingestion import IngestionPipeline
from ara.retrieval.vectors import (
    InMemoryMultiVectorStore,
    QdrantMultiVectorStore,
    make_vector_store,
)

__all__ = [
    "ColPaliClient", "InProcessMockColPali", "make_colpali", "RetrievalEngine", "RetrievalResult",
    "IngestionPipeline", "InMemoryMultiVectorStore", "QdrantMultiVectorStore", "make_vector_store",
]
