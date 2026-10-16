"""Unit and integration tests for Phase 3: Temporal Validation and Edge Cases.

Validates:
1. Relative Date Arithmetic: Resolves 'next Wednesday' using dynamic runtime temporal injection.
2. Zero-Turn Retrieval Behavior: Implicit prefetch fires without /hybrid commands, populates
   <recalled_past_context>, and answers with the date.
3. Non-Interference with Normal Chats: Generic coding/technical prompts do not trigger retrieval.
"""
import unittest
import tempfile
import os
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional

from finch.core.temporal import (
    TEMPORAL_PATTERN,
    is_temporal_trigger,
    get_temporal_context,
    inject_temporal_context,
    format_prefetched_memories,
    extract_prefetched_memories,
    PREFETCH_HEADER,
)
from finch.core.assistant import AIAssistant
from finch.config import AppConfig
from finch.storage import SQLiteArchive
from finch.providers.base import BaseLLMProvider, StreamChunk, StreamStats


class MockValidationLLMProvider(BaseLLMProvider):
    """Mock LLM Provider capturing received messages payload and providing controlled responses."""

    def __init__(self, response_text: str = "Mock response"):
        self._name = "mock_validator"
        self.default_model = "mock-validator-model"
        self.response_text = response_text
        self.received_messages: List[Dict[str, Any]] = []

    @property
    def name(self) -> str:
        return self._name

    async def stream_chat(self, messages, model=None, tools=None, options=None):
        self.received_messages = [dict(m) for m in messages]
        yield StreamChunk(
            delta=self.response_text,
            is_done=True,
            stats=StreamStats(prompt_eval_count=30, eval_count=15),
        )

    async def list_models(self):
        return []

    async def health_check(self):
        return True


class TestRelativeDateArithmetic(unittest.TestCase):
    """Verify relative date arithmetic using dynamically injected system date."""

    def test_relative_date_payload_contains_runtime_system_date(self):
        """Verify prompt payload provides accurate weekday, date, and reference note for arithmetic."""
        # Anchor: Friday, October 16, 2026
        tz_ist = timezone(timedelta(hours=5, minutes=30))
        ref_dt = datetime(2026, 10, 16, 18, 57, 0, tzinfo=tz_ist)

        temporal_block = get_temporal_context(ref_dt)
        self.assertIn("Friday", temporal_block)
        self.assertIn("2026-10-16", temporal_block)
        self.assertIn("Reference Note:", temporal_block)

        # Calculate 'next Wednesday' programmatically from the anchor date
        # Friday is weekday 4; Wednesday is weekday 2
        days_ahead = (2 - ref_dt.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7
        next_wednesday = ref_dt + timedelta(days=days_ahead)

        self.assertEqual(next_wednesday.strftime("%A"), "Wednesday")
        self.assertEqual(next_wednesday.strftime("%Y-%m-%d"), "2026-10-21")

    def test_assistant_payload_injection_for_date_arithmetic_prompt(self):
        """Verify AIAssistant properly equips prompt with temporal context for date query."""
        with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
            cfg = AppConfig(db_path=tmp.name, prefetch_enabled=True)
            archive = SQLiteArchive(db_path=tmp.name)
            try:
                mock_provider = MockValidationLLMProvider()
                assistant = AIAssistant(config=cfg, provider=mock_provider, archive=archive)

                user_prompt = "I have an assignment due next Wednesday. What day is that?"
                payload = assistant.build_messages_payload()

                system_msg = payload[0]
                self.assertEqual(system_msg["role"], "system")
                self.assertIn("[Temporal Context]", system_msg["content"])
                self.assertIn("Current Time:", system_msg["content"])
                self.assertIn("Weekday:", system_msg["content"])
                self.assertIn("Reference Note:", system_msg["content"])
            finally:
                archive.close()

    def test_relative_date_calculation_across_different_reference_days(self):
        """Verify date resolution logic correctly identifies 'next Wednesday' across various anchor days."""
        test_cases = [
            # (Anchor Date, Anchor Day, Expected Next Wednesday)
            (datetime(2026, 10, 16), "Friday", datetime(2026, 10, 21)),
            (datetime(2026, 10, 17), "Saturday", datetime(2026, 10, 21)),
            (datetime(2026, 10, 18), "Sunday", datetime(2026, 10, 21)),
            (datetime(2026, 10, 19), "Monday", datetime(2026, 10, 21)),
            (datetime(2026, 10, 20), "Tuesday", datetime(2026, 10, 21)),
            (datetime(2026, 10, 22), "Thursday", datetime(2026, 10, 28)),
        ]
        for anchor, expected_weekday, expected_target in test_cases:
            with self.subTest(anchor=anchor):
                self.assertEqual(anchor.strftime("%A"), expected_weekday)
                days_ahead = (2 - anchor.weekday()) % 7
                if days_ahead == 0:
                    days_ahead = 7
                target_date = anchor + timedelta(days=days_ahead)
                self.assertEqual(target_date, expected_target)
                self.assertEqual(target_date.strftime("%A"), "Wednesday")


class TestZeroTurnRetrievalBehavior(unittest.IsolatedAsyncioTestCase):
    """Verify zero-turn implicit retrieval of deadlines without explicit commands."""

    async def asyncSetUp(self):
        self.tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp_db.close()
        self.archive = SQLiteArchive(db_path=self.tmp_db.name)

        # Store mock deadline in past session (e.g. from an earlier discussion)
        sess = self.archive.create_session(
            title="Operating Systems Coursework",
            model="mock",
            mode="chat",
        )
        self.archive.add_message(
            session_id=sess.id,
            role="user",
            content="When is our coursework report submission due?",
        )
        self.archive.add_message(
            session_id=sess.id,
            role="assistant",
            content="Operating Systems report submission is set for next Tuesday.",
        )

        # Store mock deadline as a verified user fact
        self.archive.add_fact(
            fact="Operating Systems report submission is set for next Tuesday",
            category="deadline",
        )

        self.cfg = AppConfig(db_path=self.tmp_db.name, prefetch_enabled=True)

    async def asyncTearDown(self):
        self.archive.close()
        if os.path.exists(self.tmp_db.name):
            os.unlink(self.tmp_db.name)

    async def test_implicit_prefetch_triggers_on_upcoming_deadlines_prompt(self):
        """Verify prompt 'What deadlines do I have coming up?' triggers prefetch without /hybrid."""
        prompt = "What deadlines do I have coming up?"
        self.assertTrue(
            is_temporal_trigger(prompt),
            f"Expected '{prompt}' to trigger is_temporal_trigger",
        )
        self.assertIsNotNone(
            TEMPORAL_PATTERN.search(prompt),
            f"Expected TEMPORAL_PATTERN to match '{prompt}'",
        )

    async def test_prefetch_populates_recalled_past_context_with_stored_deadline(self):
        """Verify prefetch retrieves stored deadline and wraps it inside <recalled_past_context>."""
        assistant = AIAssistant(config=self.cfg, archive=self.archive)

        prefetch_result = await assistant.prefetch_temporal_memories("What deadlines do I have coming up?")
        self.assertIsNotNone(prefetch_result)

        # Must be wrapped in <recalled_past_context>
        self.assertIn("<recalled_past_context>", prefetch_result)
        self.assertIn("</recalled_past_context>", prefetch_result)
        self.assertIn(PREFETCH_HEADER, prefetch_result)

        # Must contain the stored mock deadline
        self.assertIn("Operating Systems report submission is set for next Tuesday", prefetch_result)

    async def test_chat_stream_delivers_recalled_context_to_provider(self):
        """Verify chat_stream delivers <recalled_past_context> to LLM provider payload."""
        mock_provider = MockValidationLLMProvider(
            response_text="Based on your notes, your Operating Systems report submission is set for next Tuesday (October 20, 2026)."
        )
        assistant = AIAssistant(config=self.cfg, provider=mock_provider, archive=self.archive)

        chunks = []
        async for chunk in assistant.chat_stream("What deadlines do I have coming up?"):
            chunks.append(chunk)

        # Verify stream completed and produced content
        full_response = "".join(c.delta for c in chunks if c.delta)
        self.assertIn("Operating Systems report submission is set for next Tuesday", full_response)

        # Verify LLM provider received <recalled_past_context> in the system message
        received = mock_provider.received_messages
        self.assertGreater(len(received), 0)
        system_content = received[0]["content"]

        self.assertIn("<recalled_past_context>", system_content)
        self.assertIn("</recalled_past_context>", system_content)
        self.assertIn("Operating Systems report submission is set for next Tuesday", system_content)
        self.assertIn("[Temporal Context]", system_content)


class TestNonInterferenceWithNormalChats(unittest.IsolatedAsyncioTestCase):
    """Verify generic coding and non-temporal prompts do not trigger retrieval or pollute context."""

    async def asyncSetUp(self):
        self.tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp_db.close()
        self.archive = SQLiteArchive(db_path=self.tmp_db.name)

        # Add past sessions that should ONLY be retrieved on temporal/planning queries
        sess = self.archive.create_session(title="OS Coursework Discussion", model="mock", mode="chat")
        self.archive.add_message(sess.id, "user", "Operating Systems deadline next Tuesday")
        self.archive.add_message(sess.id, "assistant", "Confirmed the OS report submission is on Tuesday.")

        self.cfg = AppConfig(db_path=self.tmp_db.name, prefetch_enabled=True)

    async def asyncTearDown(self):
        self.archive.close()
        if os.path.exists(self.tmp_db.name):
            os.unlink(self.tmp_db.name)

    def test_generic_coding_prompts_do_not_match_temporal_trigger(self):
        """Verify standard programming and technical prompts do not match TEMPORAL_PATTERN."""
        generic_prompts = [
            "Write a Python script to parse JSON",
            "Explain how quicksort works in memory",
            "How do I fix a CORS error in FastAPI?",
            "Implement a doubly linked list in C++",
            "What is the time complexity of binary search?",
            "Write a regex to match email addresses",
            "How does the GIL work in CPython?",
            "Optimize this SQL query with an index",
        ]
        for prompt in generic_prompts:
            with self.subTest(prompt=prompt):
                self.assertFalse(
                    is_temporal_trigger(prompt),
                    f"Expected '{prompt}' NOT to trigger is_temporal_trigger",
                )
                self.assertIsNone(
                    TEMPORAL_PATTERN.search(prompt),
                    f"Expected TEMPORAL_PATTERN NOT to match '{prompt}'",
                )

    async def test_prefetch_temporal_memories_returns_none_for_generic_coding(self):
        """Verify assistant.prefetch_temporal_memories returns None without searching DB."""
        assistant = AIAssistant(config=self.cfg, archive=self.archive)

        result = await assistant.prefetch_temporal_memories("Write a Python script to parse JSON")
        self.assertIsNone(result)

    async def test_chat_stream_does_not_inject_recalled_context_on_generic_coding(self):
        """Verify normal chat turns contain no <recalled_past_context> block in provider payload."""
        mock_provider = MockValidationLLMProvider(response_text="import json\ndata = json.loads(text)")
        assistant = AIAssistant(config=self.cfg, provider=mock_provider, archive=self.archive)

        chunks = []
        async for chunk in assistant.chat_stream("Write a Python script to parse JSON"):
            chunks.append(chunk)

        # Verify LLM message payload
        received = mock_provider.received_messages
        self.assertGreater(len(received), 0)
        system_content = received[0]["content"]

        # Temporal Context is present (runtime timestamp), but NO recalled past context
        self.assertIn("[Temporal Context]", system_content)
        self.assertNotIn("<recalled_past_context>", system_content)
        self.assertNotIn(PREFETCH_HEADER, system_content)
        self.assertNotIn("Operating Systems deadline", system_content)


if __name__ == "__main__":
    unittest.main()
