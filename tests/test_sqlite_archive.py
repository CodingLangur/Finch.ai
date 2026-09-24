"""Unit tests for SQLite conversation archive and session persistence."""
import os
import tempfile
import unittest

from ai_assistant.storage.sqlite_archive import SQLiteArchive, SessionRecord, MessageRecord


class TestSQLiteArchive(unittest.TestCase):
    def setUp(self):
        # Use an in-memory SQLite database for isolated unit testing
        self.archive = SQLiteArchive(db_path=":memory:")

    def tearDown(self):
        self.archive.close()

    def test_schema_creation(self):
        """Verify that tables and indexes are created properly."""
        conn = self.archive._get_connection()
        cursor = conn.cursor()

        # Check sessions table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'")
        self.assertIsNotNone(cursor.fetchone())

        # Check messages table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages'")
        self.assertIsNotNone(cursor.fetchone())

        # Check indexes
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_messages_session_id'")
        self.assertIsNotNone(cursor.fetchone())

    def test_create_and_get_session(self):
        """Test creating and retrieving a session record."""
        session = self.archive.create_session(
            session_id="test_sess_1",
            title="Python Discussion",
            model="test-model",
            mode="chatbot",
            metadata={"user_id": "test_user"},
        )
        self.assertEqual(session.id, "test_sess_1")
        self.assertEqual(session.title, "Python Discussion")
        self.assertEqual(session.model, "test-model")
        self.assertEqual(session.metadata.get("user_id"), "test_user")

        # Fetch session
        fetched = self.archive.get_session("test_sess_1")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.id, "test_sess_1")
        self.assertEqual(fetched.title, "Python Discussion")
        self.assertEqual(fetched.message_count, 0)

    def test_add_and_get_messages(self):
        """Test logging conversation turns and fetching them."""
        session = self.archive.create_session(session_id="sess_turns")

        # Add system message
        sys_id = self.archive.add_message(
            session_id=session.id,
            role="system",
            content="You are a helpful assistant.",
        )
        self.assertIsInstance(sys_id, int)

        # Add user message
        user_id = self.archive.add_message(
            session_id=session.id,
            role="user",
            content="What is SQLite?",
        )

        # Add assistant message with thinking and metadata
        asst_id = self.archive.add_message(
            session_id=session.id,
            role="assistant",
            content="SQLite is a C-language library that implements a SQL database engine.",
            thinking="Let me explain SQLite concisely.",
            tokens=18,
            metadata={"speed_tps": 42.5},
        )

        # Check messages with system
        all_msgs = self.archive.get_messages(session.id, include_system=True)
        self.assertEqual(len(all_msgs), 3)
        self.assertEqual(all_msgs[0].role, "system")
        self.assertEqual(all_msgs[1].role, "user")
        self.assertEqual(all_msgs[1].content, "What is SQLite?")
        self.assertEqual(all_msgs[2].role, "assistant")
        self.assertEqual(all_msgs[2].thinking, "Let me explain SQLite concisely.")
        self.assertEqual(all_msgs[2].tokens, 18)
        self.assertEqual(all_msgs[2].metadata.get("speed_tps"), 42.5)

        # Check messages without system
        chat_msgs = self.archive.get_messages(session.id, include_system=False)
        self.assertEqual(len(chat_msgs), 2)
        self.assertEqual(chat_msgs[0].role, "user")
        self.assertEqual(chat_msgs[1].role, "assistant")

        # Check message count on session
        sess_refreshed = self.archive.get_session(session.id)
        self.assertEqual(sess_refreshed.message_count, 3)

    def test_update_session_summary(self):
        """Test updating title and summary on a session."""
        session = self.archive.create_session(session_id="sess_summary_test")
        self.assertIsNone(session.summary)

        updated = self.archive.update_session_summary(
            session_id="sess_summary_test",
            title="New SQLite Title",
            summary="This is sentence one. This is sentence two.",
            metadata={"summarized": True},
        )
        self.assertTrue(updated)

        fetched = self.archive.get_session("sess_summary_test")
        self.assertEqual(fetched.title, "New SQLite Title")
        self.assertEqual(fetched.summary, "This is sentence one. This is sentence two.")
        self.assertTrue(fetched.metadata.get("summarized"))

    def test_delete_session_cascade(self):
        """Test that deleting a session deletes all related messages."""
        session = self.archive.create_session(session_id="sess_delete")
        self.archive.add_message(session.id, role="user", content="Hello")
        self.archive.add_message(session.id, role="assistant", content="Hi")

        self.assertEqual(len(self.archive.get_messages(session.id)), 2)

        deleted = self.archive.delete_session(session.id)
        self.assertTrue(deleted)
        self.assertIsNone(self.archive.get_session(session.id))
        self.assertEqual(len(self.archive.get_messages(session.id)), 0)

    def test_list_sessions(self):
        """Test listing multiple sessions in recency order."""
        self.archive.create_session(session_id="s1", title="Session 1")
        self.archive.create_session(session_id="s2", title="Session 2")
        self.archive.create_session(session_id="s3", title="Session 3")

        sessions = self.archive.list_sessions(limit=10)
        self.assertEqual(len(sessions), 3)
        ids = [s.id for s in sessions]
        self.assertIn("s1", ids)
        self.assertIn("s2", ids)
        self.assertIn("s3", ids)

    def test_file_based_persistence(self):
        """Test persisting data to a real file on disk, re-opening, and reading back."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Write with first instance
            archive1 = SQLiteArchive(db_path=tmp_path)
            s = archive1.create_session(session_id="persistent_1", title="Persistent Session")
            archive1.add_message(s.id, role="user", content="Persistent test message")
            archive1.close()

            # Read with second instance
            archive2 = SQLiteArchive(db_path=tmp_path)
            loaded_sess = archive2.get_session("persistent_1")
            self.assertIsNotNone(loaded_sess)
            self.assertEqual(loaded_sess.title, "Persistent Session")

            loaded_msgs = archive2.get_messages("persistent_1")
            self.assertEqual(len(loaded_msgs), 1)
            self.assertEqual(loaded_msgs[0].content, "Persistent test message")
            archive2.close()
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
