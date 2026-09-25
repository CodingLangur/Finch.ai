"""Unit tests for SessionSummarizer and AIAssistant session persistence integration."""
import unittest
from typing import Any, AsyncGenerator, Dict, List, Optional

from ai_assistant.config import AppConfig
from ai_assistant.core.assistant import AIAssistant
from ai_assistant.providers.base import BaseLLMProvider, ModelInfo, StreamChunk, StreamStats
from ai_assistant.storage.session_summarizer import SessionSummarizer
from ai_assistant.storage.sqlite_archive import MessageRecord, SQLiteArchive


class MockLLMProvider(BaseLLMProvider):
    """Mock LLM Provider for deterministic testing."""

    def __init__(self, canned_response: str = "TITLE: Local AI Storage\nSUMMARY: The user explored SQLite persistence. The assistant demonstrated session tracking."):
        self.canned_response = canned_response
        self.call_count = 0
        self.last_messages: Optional[List[Dict[str, str]]] = None

    @property
    def name(self) -> str:
        return "MockProvider"

    async def health_check(self) -> bool:
        return True

    async def list_models(self) -> List[ModelInfo]:
        return [ModelInfo(name="mock-model")]

    async def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        options: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        self.call_count += 1
        self.last_messages = messages
        # Yield simulated chunk
        yield StreamChunk(
            delta=self.canned_response,
            is_done=True,
            stats=StreamStats(
                ttft_ms=15.0,
                total_duration_ms=100.0,
                eval_count=25,
                eval_duration_ms=80.0,
            ),
        )


class FailingLLMProvider(BaseLLMProvider):
    """Mock Provider that simulates network or Ollama failure."""

    @property
    def name(self) -> str:
        return "FailingProvider"

    async def health_check(self) -> bool:
        return False

    async def list_models(self) -> List[ModelInfo]:
        return []

    async def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        options: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        raise ConnectionError("Ollama service unreachable")
        yield  # make it a generator


class TestSessionSummarizer(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.archive = SQLiteArchive(db_path=":memory:")
        self.provider = MockLLMProvider()
        self.summarizer = SessionSummarizer(archive=self.archive, provider=self.provider)

    async def asyncTearDown(self):
        self.archive.close()

    def test_parse_summary_output_standard(self):
        """Test parsing standard TITLE and SUMMARY labels."""
        raw = (
            "TITLE: Python Data Structures\n"
            "SUMMARY: The user asked about Python lists and dictionaries. The assistant provided performance trade-offs."
        )
        title, summary = self.summarizer.parse_summary_output(raw)
        self.assertEqual(title, "Python Data Structures")
        self.assertEqual(
            summary,
            "The user asked about Python lists and dictionaries. The assistant provided performance trade-offs.",
        )

    def test_parse_summary_output_with_quotes_and_markdown(self):
        """Test parsing when model surrounds output with markdown or quotes."""
        raw = (
            "**TITLE:** \"Asyncio Event Loops\"\n"
            "**SUMMARY:** The conversation focused on async task orchestration in Python. Key patterns for non-blocking I/O were explained."
        )
        title, summary = self.summarizer.parse_summary_output(raw)
        self.assertEqual(title, "Asyncio Event Loops")
        self.assertIn("orchestration", summary)

    def test_format_conversation_skips_system(self):
        """Test that format_conversation_for_summary excludes system prompt."""
        msgs = [
            MessageRecord(id=1, session_id="s", role="system", content="System instruction", thinking=None, tokens=0, timestamp=""),
            MessageRecord(id=2, session_id="s", role="user", content="How do I use SQLite?", thinking=None, tokens=0, timestamp=""),
            MessageRecord(id=3, session_id="s", role="assistant", content="Import sqlite3 in Python.", thinking=None, tokens=0, timestamp=""),
        ]
        formatted = self.summarizer.format_conversation_for_summary(msgs)
        self.assertNotIn("System instruction", formatted)
        self.assertIn("User: How do I use SQLite?", formatted)
        self.assertIn("Assistant: Import sqlite3 in Python.", formatted)

    async def test_summarize_session_persists_to_sqlite(self):
        """Test full summarization flow updating SQLite session record."""
        session = self.archive.create_session(session_id="test_summarize_flow")
        self.archive.add_message(session.id, role="user", content="Tell me about context compression.")
        self.archive.add_message(session.id, role="assistant", content="It saves tokens and cost.")

        title, summary = await self.summarizer.summarize_session(session.id, model="mock-model")

        self.assertEqual(title, "Local AI Storage")
        self.assertEqual(summary, "The user explored SQLite persistence. The assistant demonstrated session tracking.")
        self.assertEqual(self.provider.call_count, 1)

        # Verify database record updated
        updated_sess = self.archive.get_session(session.id)
        self.assertEqual(updated_sess.title, "Local AI Storage")
        self.assertEqual(updated_sess.summary, summary)

    async def test_summarize_empty_session(self):
        """Test empty session handling without calling the LLM."""
        session = self.archive.create_session(session_id="empty_sess")
        title, summary = await self.summarizer.summarize_session(session.id, model="mock-model")

        self.assertEqual(self.provider.call_count, 0)
        self.assertEqual(title, "Empty Session")
        self.assertIn("No conversational turns", summary)

    async def test_summarize_provider_failure_graceful(self):
        """Test that LLM failure during summarization falls back gracefully without exception."""
        failing_provider = FailingLLMProvider()
        summarizer = SessionSummarizer(archive=self.archive, provider=failing_provider)

        session = self.archive.create_session(session_id="fail_sess")
        self.archive.add_message(session.id, role="user", content="Hello there!")

        title, summary = await summarizer.summarize_session(session.id, model="failing-model")
        self.assertIn("Session (1 turns)", title)
        self.assertIn("unavailable", summary)

        updated_sess = self.archive.get_session(session.id)
        self.assertEqual(updated_sess.title, title)


class TestAssistantSessionPersistenceIntegration(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.config = AppConfig(
            default_model="mock-model",
            window_size=6,
            db_path=":memory:",
        )
        self.provider = MockLLMProvider(canned_response="This is a test response from assistant.")
        self.archive = SQLiteArchive(db_path=":memory:")
        self.assistant = AIAssistant(
            config=self.config,
            provider=self.provider,
            archive=self.archive,
        )

    async def asyncTearDown(self):
        self.assistant.close()

    async def test_chat_stream_logs_to_sqlite(self):
        """Test that AIAssistant.chat_stream automatically persists user and assistant turns into SQLite."""
        session_id = self.assistant.current_session.id
        self.assertIsNotNone(session_id)

        # Initial message in archive is system prompt
        init_msgs = self.archive.get_messages(session_id)
        self.assertGreaterEqual(len(init_msgs), 1)
        self.assertEqual(init_msgs[0].role, "system")

        # Perform chat stream
        chunks = []
        async for chunk in self.assistant.chat_stream("Hello from automated test"):
            chunks.append(chunk)

        self.assertGreater(len(chunks), 0)

        # Check messages in SQLite archive
        msgs = self.archive.get_messages(session_id, include_system=False)
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0].role, "user")
        self.assertEqual(msgs[0].content, "Hello from automated test")
        self.assertEqual(msgs[1].role, "assistant")
        self.assertEqual(msgs[1].content, "This is a test response from assistant.")

    async def test_new_session_lifecycle(self):
        """Test creating a new session resets in-memory buffer and updates active session."""
        old_id = self.assistant.current_session.id
        async for _ in self.assistant.chat_stream("First turn"):
            pass

        self.assertEqual(len(self.assistant.memory._messages), 2)

        # Start new session
        new_sess = self.assistant.new_session(title="Second Conversation")
        self.assertNotEqual(new_sess.id, old_id)
        self.assertEqual(self.assistant.current_session.id, new_sess.id)
        # In-memory conversational buffer is reset
        self.assertEqual(len(self.assistant.memory._messages), 0)

        # Past session is preserved in archive
        old_msgs = self.archive.get_messages(old_id, include_system=False)
        self.assertEqual(len(old_msgs), 2)


if __name__ == "__main__":
    unittest.main()
