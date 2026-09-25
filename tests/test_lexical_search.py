"""Unit tests for Phase 4 Lexical Search (FTS5) and Transcript Fetching."""
import os
import tempfile
import unittest

from dotenv import load_dotenv
load_dotenv()

from ai_assistant.storage.sqlite_archive import SQLiteArchive
from ai_assistant.storage.search import (
    SearchResult,
    SessionTranscript,
    load_session_transcript,
    search_keyword,
)


class TestLexicalSearchFTS5(unittest.TestCase):
    """Test suite for SQLite FTS5 lexical search and transcript fetching."""

    def setUp(self):
        self.archive = SQLiteArchive(db_path=":memory:")

        # Create sample test sessions
        self.session_algo = self.archive.create_session(
            session_id="sess_algorithms",
            title="Algorithm Design & Fibonacci",
            model="Gemma4-26000-ctx:latest",
            mode="chatbot",
        )
        self.session_ops = self.archive.create_session(
            session_id="sess_devops",
            title="DevOps & Deployment Log",
            model="gemini-2.5-flash",
            mode="agent",
        )
        self.session_arch = self.archive.create_session(
            session_id="sess_architecture",
            title="Finch.ai Storage Architecture",
            model="gemini-2.5-flash",
            mode="chatbot",
        )

        # Seed conversation turns
        self.archive.add_message(
            session_id=self.session_algo.id,
            role="user",
            content="Can you provide def calculate_fibonacci(n): in python?",
            timestamp="2026-09-24T10:15:00Z",
        )
        self.archive.add_message(
            session_id=self.session_algo.id,
            role="assistant",
            content="""Here is the optimized recursive implementation:
```python
def calculate_fibonacci(n):
    if n <= 1:
        return n
    return calculate_fibonacci(n - 1) + calculate_fibonacci(n - 2)
```
Ensure you memoize with `@functools.lru_cache` for O(N) complexity.""",
            timestamp="2026-09-24T10:15:05Z",
        )

        self.archive.add_message(
            session_id=self.session_ops.id,
            role="user",
            content="When was Finch.ai version 0.1.0 deployed?",
            timestamp="2026-09-25T08:00:00Z",
        )
        self.archive.add_message(
            session_id=self.session_ops.id,
            role="assistant",
            content="Finch.ai was successfully deployed on 2026-09-24 across all nodes.",
            timestamp="2026-09-25T08:00:10Z",
        )

        self.archive.add_message(
            session_id=self.session_arch.id,
            role="user",
            content="Explain how SQLite WAL mode and FTS5 virtual table work in Finch.ai.",
            timestamp="2026-09-25T09:30:00Z",
        )
        self.archive.add_message(
            session_id=self.session_arch.id,
            role="assistant",
            content="""SQLite WAL mode provides concurrency for simultaneous readers and writers.
The messages_fts virtual table uses FTS5 for fast sub-millisecond lexical retrieval.
Configuration is defined in config.py and ai_assistant/storage/sqlite_archive.py.""",
            timestamp="2026-09-25T09:30:15Z",
        )

    def tearDown(self):
        self.archive.close()

    def test_fts5_virtual_table_and_triggers_exist(self):
        """Verify messages_fts table and synchronization triggers exist."""
        conn = self.archive._get_connection()
        cursor = conn.cursor()

        # Check virtual table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'")
        self.assertIsNotNone(cursor.fetchone())

        # Check insert, delete, update triggers
        cursor.execute("SELECT name FROM sqlite_master WHERE type='trigger' AND name='messages_ai'")
        self.assertIsNotNone(cursor.fetchone())

        cursor.execute("SELECT name FROM sqlite_master WHERE type='trigger' AND name='messages_ad'")
        self.assertIsNotNone(cursor.fetchone())

        cursor.execute("SELECT name FROM sqlite_master WHERE type='trigger' AND name='messages_au'")
        self.assertIsNotNone(cursor.fetchone())

    def test_search_code_snippets(self):
        """Test exact-term queries on code snippets and function definitions."""
        # 1. Search exact function definition
        results = search_keyword("def calculate_fibonacci(n):", archive=self.archive)
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0].session_id, "sess_algorithms")
        self.assertIn("calculate_fibonacci", results[0].content)
        self.assertIn("[MATCH]", results[0].snippet)

        # 2. Search code decorator
        results_decorator = search_keyword("functools.lru_cache", archive=self.archive)
        self.assertEqual(len(results_decorator), 1)
        self.assertEqual(results_decorator[0].session_id, "sess_algorithms")

        # 3. Search filename in code reference
        results_file = search_keyword("config.py", archive=self.archive)
        self.assertEqual(len(results_file), 1)
        self.assertEqual(results_file[0].session_id, "sess_architecture")

    def test_search_specific_dates(self):
        """Test exact queries on specific dates containing hyphens."""
        results = search_keyword("2026-09-24", archive=self.archive)
        self.assertGreaterEqual(len(results), 1)

        # Matched session
        session_ids = [r.session_id for r in results]
        self.assertIn("sess_devops", session_ids)
        matched_ops = next(r for r in results if r.session_id == "sess_devops")
        self.assertIn("2026-09-24", matched_ops.content)
        self.assertIn("[MATCH]2026-09-24[/MATCH]", matched_ops.snippet)

    def test_search_technical_terms(self):
        """Test exact queries on technical domain terms."""
        # Term: WAL mode
        wal_results = search_keyword("WAL mode", archive=self.archive)
        self.assertGreaterEqual(len(wal_results), 1)
        self.assertEqual(wal_results[0].session_id, "sess_architecture")

        # Term: FTS5
        fts_results = search_keyword("FTS5", archive=self.archive)
        self.assertGreaterEqual(len(fts_results), 1)
        self.assertEqual(fts_results[0].session_id, "sess_architecture")

        # Term: Finch.ai
        finch_results = search_keyword("Finch.ai", archive=self.archive)
        self.assertGreaterEqual(len(finch_results), 1)

    def test_search_with_session_filter(self):
        """Test restricting search to a specific session."""
        # Query appears in sess_devops and sess_architecture
        all_results = search_keyword("Finch.ai", archive=self.archive)
        self.assertGreaterEqual(len(all_results), 2)

        # Filtered to sess_devops only
        filtered = search_keyword(
            "Finch.ai",
            session_id="sess_devops",
            archive=self.archive,
        )
        self.assertTrue(all(r.session_id == "sess_devops" for r in filtered))

    def test_search_triggers_insert_update_delete(self):
        """Verify that FTS5 is kept in sync during INSERT, UPDATE, and DELETE."""
        # Insert dynamic message
        msg_id = self.archive.add_message(
            session_id=self.session_algo.id,
            role="user",
            content="Let us test asynchronous generator pipeline with asyncio.Queue",
        )
        res = search_keyword("asyncio.Queue", archive=self.archive)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].message_id, msg_id)

        # Update message
        conn = self.archive._get_connection()
        with conn:
            conn.execute(
                "UPDATE messages SET content = ? WHERE id = ?",
                ("Updated to use threading.BoundedSemaphore instead", msg_id),
            )

        # Old term should no longer match
        res_old = search_keyword("asyncio.Queue", archive=self.archive)
        self.assertEqual(len(res_old), 0)

        # New term should match
        res_new = search_keyword("threading.BoundedSemaphore", archive=self.archive)
        self.assertEqual(len(res_new), 1)

        # Delete message
        with conn:
            conn.execute("DELETE FROM messages WHERE id = ?", (msg_id,))

        res_deleted = search_keyword("threading.BoundedSemaphore", archive=self.archive)
        self.assertEqual(len(res_deleted), 0)

    def test_cascade_delete_session_removes_fts_entries(self):
        """Verify cascade delete on session removes all associated FTS entries."""
        res_before = search_keyword("calculate_fibonacci", archive=self.archive)
        self.assertGreater(len(res_before), 0)

        # Delete session
        deleted = self.archive.delete_session("sess_algorithms")
        self.assertTrue(deleted)

        res_after = search_keyword("calculate_fibonacci", archive=self.archive)
        self.assertEqual(len(res_after), 0)

    def test_special_characters_graceful_handling(self):
        """Ensure queries with punctuation, symbols, or unclosed quotes don't crash."""
        queries = [
            '"""',
            "'''",
            "- - -",
            "calculate_fibonacci(n):",
            "@#$$%^&*()",
            "OR AND NOT",
            "nonexistent_term_xyz_12345",
        ]
        for q in queries:
            try:
                results = search_keyword(q, archive=self.archive)
                self.assertIsInstance(results, list)
            except Exception as e:
                self.fail(f"search_keyword crashed on query '{q}': {e}")

    def test_load_session_transcript(self):
        """Test retrieving full dialogue transcript for an identified session."""
        transcript = load_session_transcript(
            session_id="sess_algorithms",
            include_system=False,
            archive=self.archive,
        )
        self.assertIsNotNone(transcript)
        self.assertEqual(transcript.session_id, "sess_algorithms")
        self.assertEqual(transcript.title, "Algorithm Design & Fibonacci")
        self.assertEqual(len(transcript.turns), 2)

        # Check turn order and roles
        self.assertEqual(transcript.turns[0].role, "user")
        self.assertIn("def calculate_fibonacci(n):", transcript.turns[0].content)
        self.assertEqual(transcript.turns[1].role, "assistant")
        self.assertIn("lru_cache", transcript.turns[1].content)

        # Check formatted string output
        formatted = transcript.formatted_transcript
        self.assertIn("=== Session Transcript: Algorithm Design & Fibonacci", formatted)
        self.assertIn("[USER | 2026-09-24T10:15:00Z]:", formatted)
        self.assertIn("[ASSISTANT | 2026-09-24T10:15:05Z]:", formatted)
        self.assertIn(str(transcript), formatted)

    def test_load_session_transcript_nonexistent(self):
        """Test transcript fetching for nonexistent session returns None."""
        transcript = load_session_transcript(
            session_id="sess_nonexistent_999",
            archive=self.archive,
        )
        self.assertIsNone(transcript)


class TestGeminiLexicalSearchIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration test using live Gemini API key for inference and FTS5 search verification."""

    async def test_gemini_inference_and_lexical_retrieval(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            self.skipTest("GEMINI_API_KEY not configured in environment")

        from ai_assistant.config import AppConfig
        from ai_assistant.core.assistant import AIAssistant
        from ai_assistant.providers.gemini_provider import GeminiProvider

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tmp_db = tmp.name

        try:
            cfg = AppConfig(
                provider="gemini",
                gemini_api_key=api_key,
                gemini_default_model="gemini-2.5-flash",
                default_model="gemini-2.5-flash",
                db_path=tmp_db,
            )
            provider = GeminiProvider(api_key=api_key)
            assistant = AIAssistant(config=cfg, provider=provider, archive=SQLiteArchive(db_path=tmp_db))

            # Run a chat turn with Gemini
            prompt = "In 1 sentence, define Python's calculate_matrix_norm function."
            response_chunks = []
            async for chunk in assistant.chat_stream(prompt):
                if chunk.delta:
                    response_chunks.append(chunk.delta)

            full_response = "".join(response_chunks)
            self.assertTrue(len(full_response) > 0)

            # Test Lexical FTS5 Search on the exact term from the prompt
            search_results = assistant.search_keyword("calculate_matrix_norm")
            self.assertGreaterEqual(len(search_results), 1)
            self.assertEqual(search_results[0].session_id, assistant.current_session.id)
            self.assertIn("calculate_matrix_norm", search_results[0].content)

            # Test transcript loading for this live session
            transcript = assistant.load_session_transcript(assistant.current_session.id)
            self.assertIsNotNone(transcript)
            self.assertGreaterEqual(len(transcript.turns), 2)  # user + assistant
            self.assertIn("calculate_matrix_norm", transcript.formatted_transcript)

            assistant.close()
        finally:
            if os.path.exists(tmp_db):
                os.remove(tmp_db)


if __name__ == "__main__":
    unittest.main()
