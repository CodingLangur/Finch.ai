"""Retrieval subsystem for Finch.ai: vector embeddings, cosine ranking (sqlite-vec), and Reciprocal Rank Fusion."""
from .embeddings import BaseEmbedder, MockEmbedder, OllamaEmbedder, get_embedder
from .search import (
    SearchResult,
    TranscriptTurn,
    SessionTranscript,
    HybridSearchResult,
    search_keyword,
    load_session_transcript,
    search_hybrid,
    search_hybrid_sync,
)

__all__ = [
    "BaseEmbedder",
    "MockEmbedder",
    "OllamaEmbedder",
    "get_embedder",
    "SearchResult",
    "TranscriptTurn",
    "SessionTranscript",
    "HybridSearchResult",
    "search_keyword",
    "load_session_transcript",
    "search_hybrid",
    "search_hybrid_sync",
]
