"""Unit and integration tests for Phase 5 Semantic & Hybrid Search (sqlite-vec + RRF)."""
import asyncio
import os
import unittest

from ai_assistant.storage.sqlite_archive import SQLiteArchive
from ai_assistant.storage.search import (
    HybridSearchResult,
    search_hybrid,
    search_hybrid_sync,
)
from ai_assistant.embeddings.embedder import MockEmbedder, OllamaEmbedder


class TestSemanticAndHybridSearch(unittest.IsolatedAsyncioTestCase):
    """Test suite for sqlite-vec vector storage and Reciprocal Rank Fusion hybrid retrieval."""

    async def asyncSetUp(self):
        self.archive = SQLiteArchive(db_path=":memory:", embedding_dim=768)
        self.embedder = MockEmbedder(dimension=768)

        # Seed test sessions
        self.s1 = self.archive.create_session(
            session_id="sess_k8s",
            title="Kubernetes & Container DevOps",
        )
        self.archive.update_session_summary(
            session_id="sess_k8s",
            title="Kubernetes & Container DevOps",
            summary="Managing microservices, cluster deployment, ingress networking, and Docker containers.",
        )
        self.archive.add_message(
            session_id="sess_k8s",
            role="user",
            content="How do I inspect pods using `kubectl get pods`?",
        )

        self.s2 = self.archive.create_session(
            session_id="sess_python",
            title="Python Concurrency & Algorithms",
        )
        self.archive.update_session_summary(
            session_id="sess_python",
            title="Python Concurrency & Algorithms",
            summary="Asynchronous task processing, event loops, and recursive algorithmic structures.",
        )
        self.archive.add_message(
            session_id="sess_python",
            role="user",
            content="Can you write def calculate_fibonacci(n): in python?",
        )

        # Generate and store embeddings for both sessions
        v1 = await self.embedder.embed_text("Kubernetes & Container DevOps: Managing microservices and Docker")
        v2 = await self.embedder.embed_text("Python Concurrency & Algorithms: Asynchronous task processing and algorithms")

        self.archive.store_session_embedding("sess_k8s", v1)
        self.archive.store_session_embedding("sess_python", v2)

    async def asyncTearDown(self):
        self.archive.close()

    def test_sqlite_vec_support_active(self):
        """Verify that sqlite-vec extension is loaded and active."""
        self.assertTrue(self.archive.has_vec_support())

    async def test_semantic_search_cosine_ranking(self):
        """Test semantic vector query against sessions_vec using cosine distance."""
        query_v = await self.embedder.embed_text("deploying docker containers on cluster")
        results = self.archive.search_sessions_semantic(query_v, limit=5)

        self.assertGreaterEqual(len(results), 1)
        # Kubernetes session should have highest similarity / lowest distance
        self.assertEqual(results[0]["session_id"], "sess_k8s")
        self.assertGreater(results[0]["similarity"], 0.0)

    async def test_hybrid_search_rrf_scoring(self):
        """Test unified search_hybrid combining lexical FTS5 and semantic vector ranks."""
        # Query that matches sess_python both lexically ("calculate_fibonacci") and semantically
        results = await search_hybrid(
            query="def calculate_fibonacci(n):",
            limit=5,
            archive=self.archive,
            embedder=self.embedder,
        )

        self.assertGreaterEqual(len(results), 1)
        top_match = results[0]
        self.assertEqual(top_match.session_id, "sess_python")
        self.assertGreater(top_match.rrf_score, 0.0)
        self.assertIsNotNone(top_match.lexical_rank)

    async def test_hybrid_search_thematic_discovery(self):
        """Test that hybrid search retrieves semantically relevant session when keywords differ."""
        # Query with words not literally in the messages ("microservices")
        results = await search_hybrid(
            query="managing microservices and docker",
            limit=5,
            archive=self.archive,
            embedder=self.embedder,
        )

        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0].session_id, "sess_k8s")

    def test_search_hybrid_sync_wrapper(self):
        """Verify synchronous search_hybrid_sync wrapper."""
        results = search_hybrid_sync(
            query="calculate_fibonacci",
            limit=2,
            archive=self.archive,
            embedder=self.embedder,
        )
        self.assertIsInstance(results, list)
        self.assertGreaterEqual(len(results), 1)

    async def test_delete_session_cleans_up_vector(self):
        """Ensure deleting a session purges its vector from sessions_vec."""
        q_vec = await self.embedder.embed_text("Kubernetes & Container DevOps")
        matches_before = self.archive.search_sessions_semantic(q_vec, limit=5)
        self.assertTrue(any(m["session_id"] == "sess_k8s" for m in matches_before))

        # Delete session
        deleted = self.archive.delete_session("sess_k8s")
        self.assertTrue(deleted)

        matches_after = self.archive.search_sessions_semantic(q_vec, limit=5)
        self.assertFalse(any(m["session_id"] == "sess_k8s" for m in matches_after))


if __name__ == "__main__":
    unittest.main()
