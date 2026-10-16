"""Unit tests for Phase 1: Dynamic Temporal Injection and Headroom Compatibility."""
import unittest
from datetime import datetime, timezone, timedelta
import tempfile
import os

from finch.core.temporal import (
    get_temporal_context,
    inject_temporal_context,
    extract_temporal_context,
    strip_temporal_context,
    TEMPORAL_HEADER,
)
from finch.core.assistant import AIAssistant
from finch.config import AppConfig
from finch.compression.pipeline import ContextCompressionPipeline, HEADROOM_AVAILABLE
from finch.providers.base import BaseLLMProvider, StreamChunk, StreamStats


class MockLLMProvider(BaseLLMProvider):
    """Mock LLM Provider capturing received messages payload."""

    def __init__(self):
        self._name = "mock"
        self.default_model = "mock-model"
        self.received_messages = []

    @property
    def name(self) -> str:
        return self._name

    async def stream_chat(self, messages, model=None, tools=None, options=None):
        self.received_messages = list(messages)
        yield StreamChunk(
            delta="Mock response acknowledging temporal awareness.",
            is_done=True,
            stats=StreamStats(prompt_eval_count=20, eval_count=10),
        )

    async def list_models(self):
        return []

    async def health_check(self):
        return True


class TestTemporalInjection(unittest.TestCase):
    """Test dynamic temporal context generator and prompt assembly."""

    def test_get_temporal_context_structure(self):
        """Verify get_temporal_context output contains weekday, time, timezone, and header."""
        tc = get_temporal_context()
        self.assertIn(TEMPORAL_HEADER, tc)
        self.assertIn("- Current Time:", tc)
        self.assertIn("- Weekday:", tc)
        self.assertIn("- Timezone:", tc)
        self.assertIn("- ISO Timestamp:", tc)

    def test_get_temporal_context_deterministic(self):
        """Verify custom datetime produces accurate weekday and formatted timezone."""
        # Fixed point in time: Friday, October 16, 2026, 17:41:43 UTC+05:30
        tz_offset = timezone(timedelta(hours=5, minutes=30))
        dt = datetime(2026, 10, 16, 17, 41, 43, tzinfo=tz_offset)

        tc = get_temporal_context(dt)
        self.assertIn("Friday", tc)
        self.assertIn("2026-10-16 05:41:43 PM", tc)
        self.assertIn("UTC+05:30", tc)
        self.assertIn("2026-10-16T17:41:43+05:30", tc)

    def test_get_temporal_context_wrap_tags(self):
        """Verify wrap_tags wraps content inside <temporal_context> tags."""
        dt = datetime(2026, 10, 16, 12, 0, 0, tzinfo=timezone.utc)
        tc = get_temporal_context(dt, wrap_tags=True)
        self.assertTrue(tc.startswith("<temporal_context>\n"))
        self.assertTrue(tc.endswith("\n</temporal_context>"))
        self.assertIn(TEMPORAL_HEADER, tc)

    def test_extract_and_strip_temporal_context(self):
        """Verify extraction and clean stripping from system prompts."""
        base_prompt = "# Assistant Persona\nYou are Finch AI pair programmer.\n\n---\n### Rules\n1. Be helpful."
        dt = datetime(2026, 10, 16, 17, 41, 43, tzinfo=timezone.utc)
        temporal_block = get_temporal_context(dt)

        augmented = f"{base_prompt}\n\n{temporal_block}"
        extracted = extract_temporal_context(augmented)
        self.assertIsNotNone(extracted)
        self.assertIn(TEMPORAL_HEADER, extracted)
        self.assertIn("Friday", extracted)

        stripped = strip_temporal_context(augmented)
        self.assertEqual(stripped, base_prompt)

    def test_inject_temporal_context_replaces_stale_context(self):
        """Verify inject_temporal_context replaces stale context without duplicating blocks."""
        base_prompt = "You are Finch."
        dt1 = datetime(2026, 10, 16, 10, 0, 0, tzinfo=timezone.utc)
        dt2 = datetime(2026, 10, 16, 15, 0, 0, tzinfo=timezone.utc)

        tc1 = get_temporal_context(dt1)
        tc2 = get_temporal_context(dt2)

        first_injection = inject_temporal_context(base_prompt, tc1)
        self.assertIn("10:00:00 AM", first_injection)

        # Re-inject fresh timestamp
        second_injection = inject_temporal_context(first_injection, tc2)
        self.assertIn("03:00:00 PM", second_injection)
        self.assertNotIn("10:00:00 AM", second_injection)
        # Ensure only 1 [Temporal Context] header exists
        self.assertEqual(second_injection.count(TEMPORAL_HEADER), 1)

    def test_orchestrator_build_messages_payload(self):
        """Verify Finch orchestrator dynamically injects runtime temporal context into provider payload."""
        with tempfile.NamedTemporaryFile(suffix=".db") as tmp_db:
            cfg = AppConfig(db_path=tmp_db.name)
            assistant = AIAssistant(config=cfg)

            payload = assistant.build_messages_payload()
            self.assertGreaterEqual(len(payload), 1)
            self.assertEqual(payload[0]["role"], "system")
            self.assertIn(TEMPORAL_HEADER, payload[0]["content"])
            self.assertIn("- Current Time:", payload[0]["content"])
            self.assertIn("- Timezone:", payload[0]["content"])

            # Verify in-memory sliding window base system prompt is NOT polluted
            mem_prompt = assistant.memory.get_system_prompt()
            self.assertNotIn(TEMPORAL_HEADER, mem_prompt)

    def test_orchestrator_build_effective_system_prompt_flag(self):
        """Verify build_effective_system_prompt supports include_temporal parameter."""
        with tempfile.NamedTemporaryFile(suffix=".db") as tmp_db:
            cfg = AppConfig(db_path=tmp_db.name)
            assistant = AIAssistant(config=cfg)

            base = assistant.build_effective_system_prompt(include_temporal=False)
            self.assertNotIn(TEMPORAL_HEADER, base)

            with_temporal = assistant.build_effective_system_prompt(include_temporal=True)
            self.assertIn(TEMPORAL_HEADER, with_temporal)
            self.assertIn(base, with_temporal)

    def test_runtime_evaluation_per_turn(self):
        """Verify injection generates dynamic timestamps at runtime per turn."""
        with tempfile.NamedTemporaryFile(suffix=".db") as tmp_db:
            cfg = AppConfig(db_path=tmp_db.name)
            assistant = AIAssistant(config=cfg)

            payload1 = assistant.build_messages_payload()
            time1 = extract_temporal_context(payload1[0]["content"])

            # Even within sub-seconds, ISO timestamp or time should be populated dynamically
            payload2 = assistant.build_messages_payload()
            time2 = extract_temporal_context(payload2[0]["content"])

            self.assertIsNotNone(time1)
            self.assertIsNotNone(time2)


class TestHeadroomCompatibility(unittest.IsolatedAsyncioTestCase):
    """Test Headroom Context Compression Pipeline preservation of [Temporal Context]."""

    def setUp(self):
        self.tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp_db.close()

    def tearDown(self):
        if os.path.exists(self.tmp_db.name):
            os.unlink(self.tmp_db.name)

    def test_temporal_context_preserved_when_compress_system_messages_false(self):
        """Verify temporal block in system message is preserved when compress_system_messages=False."""
        pipeline = ContextCompressionPipeline(
            enabled=True,
            compress_user_messages=True,
            compress_system_messages=False,
            protect_recent=0,
            min_tokens_to_compress=50,
        )
        tc = get_temporal_context()
        messages = [
            {"role": "system", "content": f"# Assistant Persona\n\n{tc}"},
            {"role": "user", "content": '{"event": "log_entry", "code": 200, "metrics": [1, 2, 3]} ' * 30},
        ]

        compressed_msgs, stats = pipeline.compress_messages(messages)
        self.assertIn(TEMPORAL_HEADER, compressed_msgs[0]["content"])
        self.assertIn("- Current Time:", compressed_msgs[0]["content"])

    def test_temporal_context_preserved_when_compress_system_messages_true(self):
        """Verify temporal block is flagged immutable and restored even if system messages are compressed."""
        pipeline = ContextCompressionPipeline(
            enabled=True,
            compress_user_messages=True,
            compress_system_messages=True,
            protect_recent=0,
            min_tokens_to_compress=20,
        )
        tc = get_temporal_context()
        long_system = (
            "System Guidelines: Maintain professional demeanor, clear markdown formatting, and concise answers.\n"
            * 40
            + f"\n\n{tc}"
        )
        messages = [
            {"role": "system", "content": long_system},
            {"role": "user", "content": "What is today's schedule?"},
        ]

        compressed_msgs, stats = pipeline.compress_messages(messages)
        self.assertIn(TEMPORAL_HEADER, compressed_msgs[0]["content"])
        self.assertIn("- Current Time:", compressed_msgs[0]["content"])
        self.assertIn("- Weekday:", compressed_msgs[0]["content"])

    def test_temporal_context_preserved_in_user_or_tool_payload(self):
        """Verify temporal block is preserved if included in conversation payload turns."""
        pipeline = ContextCompressionPipeline(
            enabled=True,
            compress_user_messages=True,
            compress_system_messages=False,
            protect_recent=0,
            min_tokens_to_compress=20,
        )
        tc = get_temporal_context()
        user_turn_with_temporal = (
            '{"records": [{"id": 1, "value": "A"}, {"id": 2, "value": "B"}]} ' * 20
            + f"\n\n{tc}"
        )
        messages = [
            {"role": "system", "content": "Assistant prompt."},
            {"role": "user", "content": user_turn_with_temporal},
        ]

        compressed_msgs, stats = pipeline.compress_messages(messages)
        self.assertIn(TEMPORAL_HEADER, compressed_msgs[1]["content"])

    async def test_end_to_end_chat_stream_delivers_temporal_payload(self):
        """Verify AIAssistant.chat_stream dispatches dynamic temporal payload to LLM provider."""
        mock_provider = MockLLMProvider()
        cfg = AppConfig(db_path=self.tmp_db.name)
        assistant = AIAssistant(config=cfg, provider=mock_provider)

        chunks = []
        async for chunk in assistant.chat_stream("What time is it?"):
            chunks.append(chunk)

        self.assertGreater(len(chunks), 0)
        received = mock_provider.received_messages
        self.assertGreater(len(received), 0)
        self.assertEqual(received[0]["role"], "system")
        self.assertIn(TEMPORAL_HEADER, received[0]["content"])
        self.assertIn("- Weekday:", received[0]["content"])
        self.assertIn("- Timezone:", received[0]["content"])


if __name__ == "__main__":
    unittest.main()
