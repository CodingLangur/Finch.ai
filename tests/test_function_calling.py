"""Tests for Phase 6: Function Calling in Chat Mode.
Tests tool schemas, tool dispatcher, single-hop execution loop, and adherence testing.
"""
import asyncio
import json
import os
import unittest
from typing import Any, AsyncGenerator, Dict, List, Optional

from ai_assistant.config import AppConfig
from ai_assistant.core.assistant import AIAssistant
from ai_assistant.core.tools import (
    CHAT_TOOLS,
    LOAD_SESSION_TRANSCRIPT_TOOL,
    SEARCH_PAST_CONVERSATIONS_TOOL,
    ToolDispatcher,
)
from ai_assistant.embeddings.embedder import MockEmbedder
from ai_assistant.providers.base import BaseLLMProvider, ModelInfo, StreamChunk, StreamStats
from ai_assistant.providers.ollama_provider import OllamaProvider
from ai_assistant.storage.sqlite_archive import SQLiteArchive


class TestToolSchemas(unittest.TestCase):
    """Verify tool schemas exposed for function calling."""

    def test_search_past_conversations_schema(self):
        tool = SEARCH_PAST_CONVERSATIONS_TOOL
        self.assertEqual(tool["type"], "function")
        fn = tool["function"]
        self.assertEqual(fn["name"], "search_past_conversations")
        self.assertIn("Search archived conversation history", fn["description"])
        params = fn["parameters"]
        self.assertEqual(params["type"], "object")
        self.assertIn("query", params["properties"])
        self.assertIn("limit", params["properties"])
        self.assertIn("query", params["required"])

    def test_load_session_transcript_schema(self):
        tool = LOAD_SESSION_TRANSCRIPT_TOOL
        self.assertEqual(tool["type"], "function")
        fn = tool["function"]
        self.assertEqual(fn["name"], "load_session_transcript")
        self.assertIn("dialogue transcript", fn["description"])
        params = fn["parameters"]
        self.assertEqual(params["type"], "object")
        self.assertIn("session_id", params["properties"])
        self.assertIn("session_id", params["required"])

    def test_chat_tools_list(self):
        self.assertEqual(len(CHAT_TOOLS), 2)
        names = [t["function"]["name"] for t in CHAT_TOOLS]
        self.assertIn("search_past_conversations", names)
        self.assertIn("load_session_transcript", names)


class TestToolDispatcher(unittest.IsolatedAsyncioTestCase):
    """Test ToolDispatcher execution of search and transcript tools."""

    async def asyncSetUp(self):
        self.archive = SQLiteArchive(db_path=":memory:")
        self.embedder = MockEmbedder(dimension=8)
        self.config = AppConfig(
            db_path=":memory:",
            provider="ollama",
            default_model="mock-model",
            system_prompt="Test Assistant",
            embedding_dim=8,
        )
        self.assistant = AIAssistant(config=self.config, archive=self.archive, embedder=self.embedder)
        self.dispatcher = ToolDispatcher(self.assistant)

    async def asyncTearDown(self):
        self.assistant.close()

    async def test_search_past_conversations_empty_query(self):
        result = await self.dispatcher.execute("search_past_conversations", {"query": ""})
        self.assertIn("Error: Search query cannot be empty", result)

    async def test_search_past_conversations_no_matches(self):
        result = await self.dispatcher.execute("search_past_conversations", {"query": "nonexistent term"})
        self.assertIn("No past conversations found matching 'nonexistent term'", result)

    async def test_search_past_conversations_with_matches(self):
        sess = self.archive.create_session(title="Python Fibonacci Discussion")
        self.archive.add_message(sess.id, "user", "How do I write a recursive fibonacci function?")
        self.archive.add_message(sess.id, "assistant", "def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)")
        self.archive.update_session_summary(
            sess.id,
            title="Python Fibonacci Discussion",
            summary="Discussion of recursive fibonacci implementation in Python.",
        )

        result = await self.dispatcher.execute("search_past_conversations", {"query": "fibonacci"})
        self.assertIn("Found 1 relevant past conversation sessions", result)
        self.assertIn(sess.id, result)
        self.assertIn("Python Fibonacci Discussion", result)
        self.assertIn("load_session_transcript", result)

    async def test_load_session_transcript_valid(self):
        sess = self.archive.create_session(title="Git Workflow")
        self.archive.add_message(sess.id, "user", "How do I create a branch?")
        self.archive.add_message(sess.id, "assistant", "Use git checkout -b branch-name.")

        result = await self.dispatcher.execute("load_session_transcript", {"session_id": sess.id})
        self.assertIn("Session Transcript", result)
        self.assertIn("Git Workflow", result)
        self.assertIn("git checkout -b", result)

    async def test_load_session_transcript_not_found(self):
        result = await self.dispatcher.execute("load_session_transcript", {"session_id": "nonexistent_sess_123"})
        self.assertIn("Session 'nonexistent_sess_123' not found in database", result)

    async def test_load_session_transcript_empty_id(self):
        result = await self.dispatcher.execute("load_session_transcript", {"session_id": ""})
        self.assertIn("Error: session_id is required", result)

    async def test_dispatcher_with_json_string_arguments(self):
        sess = self.archive.create_session(title="JSON Test")
        self.archive.add_message(sess.id, "user", "Testing JSON argument passing")

        json_args = json.dumps({"session_id": sess.id})
        result = await self.dispatcher.execute("load_session_transcript", json_args)
        self.assertIn("JSON Test", result)

    async def test_dispatcher_unknown_tool(self):
        result = await self.dispatcher.execute("unknown_tool_name", {})
        self.assertIn("Error: Tool 'unknown_tool_name' is not recognized", result)


class MockToolCallingLLMProvider(BaseLLMProvider):
    """Simulates an LLM that makes a tool call in Pass 1, and answers in Pass 2."""

    def __init__(self, tool_to_call: str, tool_args: Dict[str, Any], final_answer: str):
        self.tool_to_call = tool_to_call
        self.tool_args = tool_args
        self.final_answer = final_answer
        self.call_count = 0
        self.tools_received_per_call: List[Optional[List[Dict[str, Any]]]] = []
        self.messages_received_per_call: List[List[Dict[str, Any]]] = []

    @property
    def name(self) -> str:
        return "MockToolCallingProvider"

    async def health_check(self) -> bool:
        return True

    async def list_models(self) -> List[ModelInfo]:
        return [ModelInfo(name="mock-tool-model")]

    async def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        options: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        self.call_count += 1
        self.tools_received_per_call.append(tools)
        self.messages_received_per_call.append(messages)

        if self.call_count == 1:
            # Pass 1: Model decides to call a tool
            yield StreamChunk(
                delta="",
                thinking_delta="Thinking about searching archives...",
                is_done=False,
            )
            yield StreamChunk(
                delta="",
                tool_calls=[{
                    "id": "call_test_001",
                    "function": {
                        "name": self.tool_to_call,
                        "arguments": self.tool_args,
                    },
                }],
                is_done=False,
            )
            yield StreamChunk(
                delta="",
                is_done=True,
                stats=StreamStats(
                    ttft_ms=10.0,
                    total_duration_ms=50.0,
                    eval_count=15,
                    eval_duration_ms=40.0,
                ),
            )
        else:
            # Pass 2: Final answer pass
            # Strict single-hop assertion: tools must be None
            assert tools is None, f"Expected tools=None in Pass 2, got: {tools}"

            yield StreamChunk(
                delta="",
                thinking_delta="Analyzing retrieved archive context...",
                is_done=False,
            )
            words = self.final_answer.split(" ")
            for i, w in enumerate(words):
                yield StreamChunk(delta=(w + " " if i < len(words) - 1 else w))

            yield StreamChunk(
                delta="",
                is_done=True,
                stats=StreamStats(
                    ttft_ms=5.0,
                    total_duration_ms=100.0,
                    eval_count=len(words),
                    eval_duration_ms=90.0,
                ),
            )


class MockDirectLLMProvider(BaseLLMProvider):
    """Simulates an LLM answering directly without tool calls."""

    def __init__(self, answer: str = "Direct answer to user inquiry."):
        self.answer = answer
        self.call_count = 0
        self.tools_received_per_call: List[Optional[List[Dict[str, Any]]]] = []

    @property
    def name(self) -> str:
        return "MockDirectProvider"

    async def health_check(self) -> bool:
        return True

    async def list_models(self) -> List[ModelInfo]:
        return [ModelInfo(name="mock-direct-model")]

    async def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        options: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        self.call_count += 1
        self.tools_received_per_call.append(tools)

        yield StreamChunk(delta=self.answer, is_done=False)
        yield StreamChunk(
            delta="",
            is_done=True,
            stats=StreamStats(
                ttft_ms=12.0,
                total_duration_ms=60.0,
                eval_count=10,
                eval_duration_ms=50.0,
            ),
        )


class TestSingleHopExecutionLoop(unittest.IsolatedAsyncioTestCase):
    """Verify single-hop retrieval cycle: User Query -> LLM Tool Call -> DB Fetch -> Final Answer."""

    async def asyncSetUp(self):
        self.archive = SQLiteArchive(db_path=":memory:")
        self.embedder = MockEmbedder(dimension=8)
        self.config = AppConfig(
            db_path=":memory:",
            provider="ollama",
            default_model="mock-model",
            system_prompt="Test Assistant",
            embedding_dim=8,
        )

        # Pre-seed archive with past session
        self.past_sess = self.archive.create_session(title="SQLite FTS5 Optimization")
        self.archive.add_message(self.past_sess.id, "user", "How fast is FTS5 in SQLite?")
        self.archive.add_message(self.past_sess.id, "assistant", "FTS5 provides sub-millisecond lexical search.")
        self.archive.update_session_summary(self.past_sess.id, "Discussion about SQLite FTS5 search performance.")

    async def asyncTearDown(self):
        self.archive.close()

    async def test_single_hop_tool_execution(self):
        provider = MockToolCallingLLMProvider(
            tool_to_call="search_past_conversations",
            tool_args={"query": "SQLite FTS5"},
            final_answer="According to our past discussion, FTS5 offers sub-millisecond search.",
        )
        assistant = AIAssistant(config=self.config, provider=provider, archive=self.archive, embedder=self.embedder)

        chunks: List[StreamChunk] = []
        notices: List[str] = []
        answer_parts: List[str] = []

        async for chunk in assistant.chat_stream("What did we say about SQLite FTS5 speed?"):
            chunks.append(chunk)
            if chunk.tool_call_notice:
                notices.append(chunk.tool_call_notice)
            if chunk.delta:
                answer_parts.append(chunk.delta)

        # 1. Notice chunk was emitted for the UI
        self.assertEqual(len(notices), 1)
        self.assertIn("search_past_conversations", notices[0])
        self.assertIn("SQLite FTS5", notices[0])

        # 2. Strict Single-Hop verification: exactly 2 LLM passes occurred
        self.assertEqual(provider.call_count, 2)
        # Pass 1 received CHAT_TOOLS
        self.assertIsNotNone(provider.tools_received_per_call[0])
        # Pass 2 received tools=None (strict single-hop enforcement!)
        self.assertIsNone(provider.tools_received_per_call[1])

        # 3. Final answer received matches Pass 2 output
        full_answer = "".join(answer_parts).strip()
        self.assertEqual(full_answer, "According to our past discussion, FTS5 offers sub-millisecond search.")

        # 4. Turn was persisted to SQLite archive with tool execution metadata
        messages = self.archive.get_messages(session_id=assistant.current_session.id)
        # 1 system + 1 user + 1 assistant = 3 messages
        self.assertEqual(len(messages), 3)
        asst_msg = messages[-1]
        self.assertEqual(asst_msg.role, "assistant")
        self.assertEqual(asst_msg.content, full_answer)
        self.assertIn("tool_calls", asst_msg.metadata)
        self.assertIn("search_past_conversations", asst_msg.metadata["tools_executed"])

        assistant.close()

    async def test_direct_answer_flow_without_tools(self):
        provider = MockDirectLLMProvider(answer="2 + 2 equals 4.")
        assistant = AIAssistant(config=self.config, provider=provider, archive=self.archive, embedder=self.embedder)

        chunks: List[StreamChunk] = []
        notices: List[str] = []
        answer_parts: List[str] = []

        async for chunk in assistant.chat_stream("What is 2 + 2?"):
            chunks.append(chunk)
            if chunk.tool_call_notice:
                notices.append(chunk.tool_call_notice)
            if chunk.delta:
                answer_parts.append(chunk.delta)

        # No tool notices emitted
        self.assertEqual(len(notices), 0)
        # Only 1 LLM pass occurred
        self.assertEqual(provider.call_count, 1)
        self.assertEqual("".join(answer_parts).strip(), "2 + 2 equals 4.")

        # Persisted turn has no tool execution metadata
        messages = self.archive.get_messages(session_id=assistant.current_session.id)
        asst_msg = messages[-1]
        self.assertNotIn("tools_executed", asst_msg.metadata)

        assistant.close()


class TestLiveOllamaAdherence(unittest.IsolatedAsyncioTestCase):
    """Test model adherence against live Ollama: reaches into history only when relevant."""

    async def asyncSetUp(self):
        self.provider = OllamaProvider()
        self.live_server = await self.provider.health_check()

    async def test_adherence_historical_vs_general_queries(self):
        if not self.live_server:
            self.skipTest("Live Ollama daemon not reachable at localhost:11434")

        # 1. Historical query: MUST trigger search_past_conversations
        messages_hist = [{
            "role": "user",
            "content": "What was that python script about fibonacci we discussed earlier?",
        }]
        tool_calls_hist: List[Dict[str, Any]] = []

        async for chunk in self.provider.stream_chat(
            messages=messages_hist,
            model="Gemma4-26000-ctx:latest",
            tools=CHAT_TOOLS,
        ):
            if chunk.tool_calls:
                tool_calls_hist.extend(chunk.tool_calls)

        self.assertGreater(len(tool_calls_hist), 0, "Model failed to invoke tool for historical query")
        called_names = [tc["function"]["name"] for tc in tool_calls_hist]
        self.assertIn("search_past_conversations", called_names)

        # 2. General knowledge query: MUST NOT trigger tools
        messages_gen = [{
            "role": "user",
            "content": "What is the capital of France? Please keep your response to one word.",
        }]
        tool_calls_gen: List[Dict[str, Any]] = []
        direct_tokens: List[str] = []

        async for chunk in self.provider.stream_chat(
            messages=messages_gen,
            model="Gemma4-26000-ctx:latest",
            tools=CHAT_TOOLS,
        ):
            if chunk.tool_calls:
                tool_calls_gen.extend(chunk.tool_calls)
            if chunk.delta:
                direct_tokens.append(chunk.delta)

        self.assertEqual(len(tool_calls_gen), 0, f"Model unnecessarily called tools for general query: {tool_calls_gen}")
        direct_answer = "".join(direct_tokens).strip()
        self.assertTrue(len(direct_answer) > 0, "Model failed to produce direct answer for general query")


if __name__ == "__main__":
    unittest.main()
