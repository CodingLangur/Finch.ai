"""Unit tests for Phase 2: Zero-Turn Implicit Memory Prefetch."""
import unittest
import tempfile
import os
from unittest.mock import AsyncMock, patch

from finch.core.temporal import (
    TEMPORAL_PATTERN,
    TEMPORAL_TRIGGER_PATTERN,
    is_temporal_trigger,
    format_prefetched_memories,
    extract_prefetched_memories,
    strip_prefetched_memories,
    PREFETCH_HEADER,
    get_temporal_context,
)
from finch.core.assistant import AIAssistant
from finch.config import AppConfig
from finch.retrieval.search import HybridSearchResult
from finch.storage import SQLiteArchive
from finch.compression.pipeline import ContextCompressionPipeline
from finch.providers.base import BaseLLMProvider, StreamChunk, StreamStats


class MockLLMProvider(BaseLLMProvider):
    """Mock provider capturing received message payloads for testing."""

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
            delta="Here is your requested schedule.",
            is_done=True,
            stats=StreamStats(prompt_eval_count=25, eval_count=12),
        )

    async def list_models(self):
        return []

    async def health_check(self):
        return True


class TestTemporalTriggerRules(unittest.TestCase):
    """Verify temporal retrieval trigger rules and keyword/regex pattern."""

    def test_temporal_trigger_pattern_matches_planning_and_scheduling(self):
        """Verify matcher triggers on planning, scheduling, and calendar queries."""
        positive_queries = [
            "What is my schedule for today?",
            "Can we plan our sprint tomorrow?",
            "Do I have any meetings next week?",
            "Show me my calendar appointments for Friday",
            "Please reschedule my 10am standup",
            "Remind me when the project deadline is",
            "What did we discuss yesterday regarding the roadmap?",
            "What are our goals for next month?",
            "When is our next milestone due?",
            "Check my availability for this afternoon",
            "What is on my todo list for tomorrow morning?",
        ]
        for q in positive_queries:
            with self.subTest(query=q):
                self.assertTrue(
                    is_temporal_trigger(q),
                    f"Expected '{q}' to match TEMPORAL_PATTERN",
                )
                self.assertIsNotNone(
                    TEMPORAL_PATTERN.search(q),
                    f"Expected TEMPORAL_PATTERN.search to match '{q}'",
                )

    def test_temporal_trigger_pattern_ignores_unrelated_prompts(self):
        """Verify matcher does not fire on general programming or generic questions."""
        negative_queries = [
            "How do I reverse a binary tree in Python?",
            "Explain what an HTTP 502 status code means",
            "Write a shell command to find all .py files",
            "What is the difference between TCP and UDP?",
            "Format this JSON string properly",
            "Help me debug this NullPointerException in Java",
        ]
        for q in negative_queries:
            with self.subTest(query=q):
                self.assertFalse(
                    is_temporal_trigger(q),
                    f"Expected '{q}' NOT to match TEMPORAL_PATTERN",
                )
                self.assertIsNone(
                    TEMPORAL_PATTERN.search(q),
                    f"Expected TEMPORAL_PATTERN.search NOT to match '{q}'",
                )


class TestFormatContextInjectionBlock(unittest.TestCase):
    """Verify clean context formatting and token guardrail limits."""

    def test_format_prefetched_memories_structure(self):
        """Verify format structure contains header, session titles, snippets, and facts."""
        sessions = [
            HybridSearchResult(
                session_id="sess_123",
                title="Q4 Sprint Planning",
                summary="Discussed API endpoints and deployment milestones.",
                rrf_score=0.03,
                lexical_rank=1,
                semantic_rank=1,
                cosine_similarity=0.88,
                matched_snippets=["We agreed to deploy the auth microservice on [MATCH]Friday[/MATCH]."],
                model="gpt-4o",
                mode="chat",
                updated_at="2026-10-15T10:00:00Z",
            )
        ]
        facts = [
            {"id": 7, "category": "preference", "fact": "Prefers morning standup at 9:30 AM"}
        ]

        block = format_prefetched_memories(sessions, facts, max_tokens=350)
        self.assertIsNotNone(block)
        self.assertIn(PREFETCH_HEADER, block)
        self.assertIn("Q4 Sprint Planning", block)
        self.assertIn("auth microservice on Friday", block)
        # Verify [MATCH] tags are stripped
        self.assertNotIn("[MATCH]", block)
        self.assertNotIn("[/MATCH]", block)
        self.assertIn("Fact #7", block)
        self.assertIn("Prefers morning standup at 9:30 AM", block)

    def test_format_prefetched_memories_returns_none_when_empty(self):
        """Verify None is returned when no sessions or facts are found."""
        self.assertIsNone(format_prefetched_memories([], []))
        self.assertIsNone(format_prefetched_memories([], None))

    def test_guardrail_token_limits_enforced(self):
        """Verify prefetch block is cleanly truncated when exceeding token budget."""
        # Create a series of sessions that together exceed 100 tokens (400 chars)
        sessions = [
            HybridSearchResult(
                session_id=f"sess_{i}",
                title=f"Detailed Planning Session #{i}",
                summary="Extensive roadmap planning with many tasks and milestones." * 5,
                rrf_score=0.01 * i,
                lexical_rank=i,
                semantic_rank=i,
                cosine_similarity=0.75,
                matched_snippets=["Detailed snippet description with many requirements for the team." * 4],
                model="gpt-4o",
                mode="chat",
                updated_at="2026-10-10T12:00:00Z",
            )
            for i in range(10)
        ]

        max_tokens = 80  # ~320 characters
        block = format_prefetched_memories(sessions, None, max_tokens=max_tokens)
        self.assertIsNotNone(block)
        self.assertIn("truncated to 80 tokens context limit", block)
        # Character count must be within budget
        self.assertLessEqual(len(block), (max_tokens * 4) + 100)

    def test_extract_and_strip_prefetched_memories(self):
        """Verify extraction and clean stripping of prefetch block."""
        base_text = "# Persona\nYou are Finch AI."
        sessions = [{"title": "Team Sync", "matched_snippets": ["Standup notes"]}]
        prefetch = format_prefetched_memories(sessions, None, max_tokens=200)

        combined = f"{base_text}\n\n{prefetch}"
        extracted = extract_prefetched_memories(combined)
        self.assertIsNotNone(extracted)
        self.assertIn(PREFETCH_HEADER, extracted)

        stripped = strip_prefetched_memories(combined)
        self.assertEqual(stripped, base_text)


class TestImplicitMemoryPrefetchIntegration(unittest.IsolatedAsyncioTestCase):
    """Verify orchestrator implicit memory prefetch and zero-turn delivery."""

    def setUp(self):
        self.tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp_db.close()
        self.archive = SQLiteArchive(db_path=self.tmp_db.name)

        # Seed archive with past sessions and messages for retrieval
        s1 = self.archive.create_session(title="GATE Exam Study Schedule", model="gpt-4o", mode="chat")
        self.archive.add_message(s1.id, "user", "Create a study schedule for my GATE exam tomorrow.")
        self.archive.add_message(s1.id, "assistant", "Here is your GATE study schedule: Algorithms from 9am to 12pm, OS from 2pm to 5pm.")

        s2 = self.archive.create_session(title="Python Project Roadmap", model="gpt-4o", mode="agent")
        self.archive.add_message(s2.id, "user", "Plan the database migration timeline for next week.")
        self.archive.add_message(s2.id, "assistant", "Database migration will be executed on Tuesday at 11pm.")

        # Seed user facts
        self.archive.add_fact(fact="Prefers studying algorithms in the morning", category="preference")
        self.archive.add_fact(fact="Daily sprint standup scheduled at 10am", category="schedule")

    def tearDown(self):
        self.archive.close()
        if os.path.exists(self.tmp_db.name):
            os.unlink(self.tmp_db.name)

    async def test_prefetch_temporal_memories_returns_relevant_context(self):
        """Verify prefetch_temporal_memories retrieves relevant past memories on scheduling queries."""
        cfg = AppConfig(db_path=self.tmp_db.name, prefetch_enabled=True)
        assistant = AIAssistant(config=cfg, archive=self.archive)

        prefetch_block = await assistant.prefetch_temporal_memories("What is my study schedule for tomorrow?")
        self.assertIsNotNone(prefetch_block)
        self.assertIn(PREFETCH_HEADER, prefetch_block)
        self.assertIn("GATE Exam Study Schedule", prefetch_block)

    async def test_prefetch_temporal_memories_skips_non_temporal_queries(self):
        """Verify prefetch_temporal_memories returns None for non-temporal queries."""
        cfg = AppConfig(db_path=self.tmp_db.name, prefetch_enabled=True)
        assistant = AIAssistant(config=cfg, archive=self.archive)

        result = await assistant.prefetch_temporal_memories("How do I sort a list in Python?")
        self.assertIsNone(result)

    async def test_chat_stream_zero_turn_delivers_prefetched_memory(self):
        """Verify chat_stream performs zero-turn memory prefetch and delivers it to LLM provider."""
        mock_provider = MockLLMProvider()
        cfg = AppConfig(db_path=self.tmp_db.name, prefetch_enabled=True)
        assistant = AIAssistant(config=cfg, provider=mock_provider, archive=self.archive)

        chunks = []
        async for chunk in assistant.chat_stream("What was my GATE study schedule for tomorrow?"):
            chunks.append(chunk)

        self.assertGreater(len(chunks), 0)
        received = mock_provider.received_messages
        self.assertGreater(len(received), 0)

        system_msg = received[0]
        self.assertEqual(system_msg["role"], "system")
        # System prompt must contain both [Temporal Context] (Phase 1) and [Implicit Memory Prefetch] (Phase 2)
        self.assertIn("[Temporal Context]", system_msg["content"])
        self.assertIn(PREFETCH_HEADER, system_msg["content"])
        self.assertIn("GATE Exam Study Schedule", system_msg["content"])

    async def test_prefetched_context_preserved_in_headroom_compression(self):
        """Verify Headroom ContextCompressionPipeline preserves prefetch block intact."""
        pipeline = ContextCompressionPipeline(
            enabled=True,
            compress_user_messages=True,
            compress_system_messages=True,  # Even when system compression is enabled
            protect_recent=0,
            min_tokens_to_compress=20,
        )

        tc = get_temporal_context()
        sessions = [{"title": "Sprint Review", "matched_snippets": ["Discussed timeline for milestone." * 5]}]
        prefetch = format_prefetched_memories(sessions, None, max_tokens=200)

        long_system = (
            "System prompt instructions for AI assistant companion.\n" * 30
            + f"\n\n{tc}\n\n{prefetch}"
        )
        messages = [
            {"role": "system", "content": long_system},
            {"role": "user", "content": "When is our sprint deadline?"},
        ]

        compressed, stats = pipeline.compress_messages(messages)
        self.assertIn("[Temporal Context]", compressed[0]["content"])
        self.assertIn(PREFETCH_HEADER, compressed[0]["content"])
        self.assertIn("Sprint Review", compressed[0]["content"])


if __name__ == "__main__":
    unittest.main()
