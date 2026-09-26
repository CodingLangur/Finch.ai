"""Unit tests for Phase 8: Storage Optimization, SQLite Housekeeping, and Compression."""
import os
import tempfile
import time
from datetime import datetime, timezone, timedelta
import unittest
from unittest.mock import patch

from ai_assistant.storage.compression import ColumnCompressor, ZSTD_AVAILABLE
from ai_assistant.storage.sqlite_archive import SQLiteArchive
from ai_assistant.config import AppConfig
from ai_assistant.core.assistant import AIAssistant
from tests.test_agent_mode import MockEmbedder
from tests.test_session_summarizer import MockLLMProvider


class TestColumnCompressor(unittest.TestCase):
    """Tests for ColumnCompressor using zstd and zlib fallback."""

    def setUp(self):
        self.compressor = ColumnCompressor(min_size_bytes=100)

    def test_compress_decompress_roundtrip(self):
        """Verify round-trip text compression and decompression."""
        large_text = "Finch AI Assistant log entry: status=OK, code=200, trace=None.\n" * 50
        compressed, was_compressed = self.compressor.compress_text(large_text)

        self.assertTrue(was_compressed)
        self.assertNotEqual(compressed, large_text)
        self.assertTrue(self.compressor.is_compressed(compressed))
        self.assertTrue(compressed.startswith("__ZSTD__:") or compressed.startswith("__ZLIB__:"))

        decompressed = self.compressor.decompress_text(compressed)
        self.assertEqual(decompressed, large_text)

    def test_compress_below_threshold(self):
        """Verify that short texts below threshold are not compressed."""
        short_text = "Short text under 100 bytes"
        compressed, was_compressed = self.compressor.compress_text(short_text)

        self.assertFalse(was_compressed)
        self.assertEqual(compressed, short_text)
        self.assertFalse(self.compressor.is_compressed(compressed))

    def test_decompress_uncompressed_text(self):
        """Verify that decompress_text safely passes through raw uncompressed strings."""
        raw_text = "Just a regular uncompressed string without any compression header."
        result = self.compressor.decompress_text(raw_text)
        self.assertEqual(result, raw_text)

    def test_bytes_compression_roundtrip(self):
        """Verify binary bytes compression and decompression."""
        data = b"Arbitrary binary payload data for archiving: " * 50
        compressed, was_compressed = self.compressor.compress_bytes(data)
        self.assertTrue(was_compressed)
        self.assertLess(len(compressed), len(data))

        decompressed = self.compressor.decompress_bytes(compressed)
        self.assertEqual(decompressed, data)

    def test_benchmark_utility(self):
        """Verify the benchmark method runs and produces expected metrics."""
        sample_text = '{"event": "system_metric", "cpu": 12.5, "mem_mb": 4096}\n' * 50
        stats = self.compressor.benchmark([sample_text])

        self.assertIn("total_raw_bytes", stats)
        self.assertIn("total_compressed_bytes", stats)
        self.assertIn("compression_ratio", stats)
        self.assertIn("savings_pct", stats)
        self.assertGreater(stats["savings_pct"], 50.0)

    def test_zlib_fallback_behavior(self):
        """Verify that compressor falls back to zlib when zstandard is disabled."""
        with patch("ai_assistant.storage.compression.ZSTD_AVAILABLE", False):
            fallback_compressor = ColumnCompressor(min_size_bytes=50)
            text = "Testing fallback zlib compression routine for database message bodies.\n" * 20
            compressed, was_comp = fallback_compressor.compress_text(text)

            self.assertTrue(was_comp)
            self.assertTrue(compressed.startswith("__ZLIB__:"))
            decompressed = fallback_compressor.decompress_text(compressed)
            self.assertEqual(decompressed, text)


class TestSQLiteHousekeeping(unittest.TestCase):
    """Tests for SQLite housekeeping routines: checkpoint, optimize, vacuum, integrity check."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp_dir.name, "test_maintenance.db")
        self.archive = SQLiteArchive(db_path=self.db_path)

    def tearDown(self):
        self.archive.close()
        self.tmp_dir.cleanup()

    def test_integrity_check(self):
        """Verify integrity_check returns 'ok'."""
        status = self.archive.integrity_check()
        self.assertEqual(status, "ok")

    def test_checkpoint_and_optimize(self):
        """Verify PRAGMA wal_checkpoint and PRAGMA optimize execute without error."""
        cp_result = self.archive.checkpoint()
        self.assertIn("busy", cp_result)
        self.assertIn("log", cp_result)
        self.assertIn("checkpointed", cp_result)

        # PRAGMA optimize executes cleanly (returns None)
        self.archive.optimize()

    def test_storage_stats(self):
        """Verify get_storage_stats returns accurate disk metrics."""
        self.archive.create_session("sess_stats", title="Stats Test")
        self.archive.add_message("sess_stats", "user", "Hello maintenance stats")

        stats = self.archive.get_storage_stats()
        self.assertIn("page_size", stats)
        self.assertIn("page_count", stats)
        self.assertIn("freelist_count", stats)
        self.assertIn("file_size_bytes", stats)
        self.assertIn("total_size_bytes", stats)
        self.assertIn("session_count", stats)
        self.assertIn("message_count", stats)
        self.assertEqual(stats["session_count"], 1)
        self.assertEqual(stats["message_count"], 1)
        self.assertGreater(stats["file_size_bytes"], 0)

    def test_vacuum_reclaims_freelist_pages(self):
        """Verify that inserting many messages, deleting them, and vacuuming reclaims space."""
        session = self.archive.create_session("sess_vacuum", title="Vacuum Test")
        # Insert 100 large messages to allocate multiple pages
        for i in range(100):
            self.archive.add_message("sess_vacuum", "user", f"Bulk message content payload number {i} " * 50)

        # Delete all messages from session
        conn = self.archive._get_connection()
        conn.execute("DELETE FROM messages WHERE session_id = 'sess_vacuum'")
        conn.commit()

        vac_result = self.archive.vacuum()
        self.assertIn("bytes_before", vac_result)
        self.assertIn("bytes_after", vac_result)
        self.assertIn("bytes_reclaimed", vac_result)

        stats_after = self.archive.get_storage_stats()
        self.assertEqual(stats_after["freelist_count"], 0)

    def test_run_maintenance(self):
        """Verify end-to-end run_maintenance routine."""
        self.archive.create_session("sess_maint", title="Maintenance Routine")
        self.archive.add_message("sess_maint", "user", "Routine maintenance test")

        report = self.archive.run_maintenance(vacuum=True)
        self.assertIn("integrity", report)
        self.assertIn("checkpoint", report)
        self.assertIn("storage_stats", report)
        self.assertIn("vacuum", report)
        self.assertEqual(report["integrity"], "ok")


class TestSQLiteMessageCompression(unittest.TestCase):
    """Tests for transparent column-level message body compression in SQLiteArchive."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp_dir.name, "test_compressed.db")
        # Enable compression with low threshold (100 bytes)
        self.archive = SQLiteArchive(
            db_path=self.db_path,
            compress_large_messages=True,
            compression_threshold_bytes=100,
        )

    def tearDown(self):
        self.archive.close()
        self.tmp_dir.cleanup()

    def test_large_message_is_compressed_on_disk(self):
        """Verify large messages are stored compressed in SQLite but read back uncompressed."""
        session = self.archive.create_session("sess_comp", title="Compression Session")
        large_body = "System diagnostic trace log: [ERROR] code 500 at worker node 42.\n" * 30

        msg_id = self.archive.add_message("sess_comp", "user", large_body)

        # Verify raw storage in SQLite table has compression prefix
        conn = self.archive._get_connection()
        cur = conn.cursor()
        cur.execute("SELECT content FROM messages WHERE id = ?", (msg_id,))
        raw_stored_content = cur.fetchone()[0]

        self.assertNotEqual(raw_stored_content, large_body)
        self.assertTrue(
            raw_stored_content.startswith("__ZSTD__:") or raw_stored_content.startswith("__ZLIB__:")
        )

        # Verify transparent retrieval through get_messages
        messages = self.archive.get_messages("sess_comp")
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].content, large_body)

    def test_short_message_remains_uncompressed_on_disk(self):
        """Verify short messages under threshold remain plain text."""
        session = self.archive.create_session("sess_short", title="Short Session")
        short_body = "Just a short 20-byte prompt."

        msg_id = self.archive.add_message("sess_short", "user", short_body)

        conn = self.archive._get_connection()
        cur = conn.cursor()
        cur.execute("SELECT content FROM messages WHERE id = ?", (msg_id,))
        raw_stored_content = cur.fetchone()[0]

        self.assertEqual(raw_stored_content, short_body)

        messages = self.archive.get_messages("sess_short")
        self.assertEqual(messages[0].content, short_body)

    def test_fts5_search_works_with_compressed_messages(self):
        """Verify that FTS5 full-text search works transparently with compressed messages."""
        session = self.archive.create_session("sess_fts", title="FTS Search Session")
        special_body = "UniqueKeywordAlpha999 diagnostic analysis for quantum computing subsystem.\n" * 15

        self.archive.add_message("sess_fts", "user", special_body)

        results = self.archive.search_messages("UniqueKeywordAlpha999")
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0]["session_id"], "sess_fts")
        self.assertIn("UniqueKeywordAlpha999", results[0]["snippet"])


class TestSQLiteSessionArchival(unittest.TestCase):
    """Tests for database-level session archival."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp_dir.name, "active.db")
        self.archive_db_path = os.path.join(self.tmp_dir.name, "archive_vault.db")
        self.archive = SQLiteArchive(db_path=self.db_path)

    def tearDown(self):
        self.archive.close()
        self.tmp_dir.cleanup()

    def test_archive_sessions_by_age(self):
        """Verify moving sessions older than N days into separate archive DB."""
        # Session 1: Old session (simulate created 40 days ago)
        old_dt = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()
        self.archive.create_session("old_session_1", title="Old Session 1")
        self.archive.add_message("old_session_1", "user", "Old message content")

        # Backdate created_at in ISO format
        conn = self.archive._get_connection()
        conn.execute("UPDATE sessions SET created_at = ? WHERE id = 'old_session_1'", (old_dt,))
        conn.commit()

        # Session 2: Recent session (created now)
        self.archive.create_session("recent_session_2", title="Recent Session 2")
        self.archive.add_message("recent_session_2", "user", "Recent message content")

        # Archive sessions older than 30 days
        result = self.archive.archive_sessions(
            older_than_days=30,
            archive_db_path=self.archive_db_path,
        )

        self.assertEqual(result["archived_sessions"], 1)
        self.assertEqual(result["archived_messages"], 1)
        self.assertTrue(os.path.exists(self.archive_db_path))

        # Check that old_session_1 is gone from active DB
        self.assertIsNone(self.archive.get_session("old_session_1"))
        self.assertEqual(len(self.archive.get_messages("old_session_1")), 0)

        # Check that recent_session_2 remains in active DB
        self.assertIsNotNone(self.archive.get_session("recent_session_2"))
        self.assertEqual(len(self.archive.get_messages("recent_session_2")), 1)

        # Verify old_session_1 is present in the archive DB
        archive_db = SQLiteArchive(db_path=self.archive_db_path)
        try:
            archived_session = archive_db.get_session("old_session_1")
            self.assertIsNotNone(archived_session)
            self.assertEqual(archived_session.title, "Old Session 1")
            archived_msgs = archive_db.get_messages("old_session_1")
            self.assertEqual(len(archived_msgs), 1)
            self.assertEqual(archived_msgs[0].content, "Old message content")
        finally:
            archive_db.close()


class TestAIAssistantMaintenanceIntegration(unittest.TestCase):
    """Tests for AIAssistant maintenance wrapper methods."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp_dir.name, "assistant_maint.db")
        self.config = AppConfig(
            provider="ollama",
            db_path=self.db_path,
            compress_message_bodies=True,
            message_compression_threshold=100,
        )
        self.archive = SQLiteArchive(
            db_path=self.db_path,
            compress_large_messages=True,
            compression_threshold_bytes=100,
        )
        self.assistant = AIAssistant(
            config=self.config,
            provider=MockLLMProvider(["Hello test"]),
            model="mock-model",
            archive=self.archive,
            embedder=MockEmbedder(8),
        )

    def tearDown(self):
        self.assistant.close()
        self.tmp_dir.cleanup()

    def test_assistant_maintenance_wrappers(self):
        """Verify AIAssistant delegates maintenance routines properly."""
        # Storage stats
        stats = self.assistant.get_storage_stats()
        self.assertIn("file_size_bytes", stats)
        self.assertIn("page_count", stats)

        # Vacuum
        vac = self.assistant.vacuum()
        self.assertIn("bytes_reclaimed", vac)

        # Full maintenance
        report = self.assistant.run_maintenance(vacuum=True)
        self.assertEqual(report["integrity"], "ok")
        self.assertIn("storage_stats", report)

        # Archive sessions
        archive_path = os.path.join(self.tmp_dir.name, "archive_test.db")
        arch_res = self.assistant.archive_sessions(older_than_days=1, archive_db_path=archive_path)
        self.assertIn("archived_sessions", arch_res)


if __name__ == "__main__":
    unittest.main()
