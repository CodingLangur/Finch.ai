"""Comprehensive tests for Phase 7: Chat vs. Agent Mode Toggle, Action Tools, and Multi-Turn Loop."""
import asyncio
import json
import os
import shutil
import tempfile
import unittest
from typing import Any, AsyncGenerator, Dict, List, Optional

from ai_assistant.config import AppConfig
from ai_assistant.core.assistant import AIAssistant, AssistantMode
from ai_assistant.core.tools import (
    AGENT_ACTION_TOOLS,
    AGENT_TOOLS,
    CHAT_TOOLS,
    LIST_DIRECTORY_TOOL,
    LOAD_SESSION_TRANSCRIPT_TOOL,
    PYTHON_INTERPRETER_TOOL,
    READ_FILE_TOOL,
    RUN_TERMINAL_COMMAND_TOOL,
    SEARCH_PAST_CONVERSATIONS_TOOL,
    WRITE_FILE_TOOL,
    ToolDispatcher,
)
from ai_assistant.embeddings.embedder import MockEmbedder
from ai_assistant.providers.base import BaseLLMProvider, ModelInfo, StreamChunk, StreamStats
from ai_assistant.providers.gemini_provider import GeminiProvider
from ai_assistant.storage.sqlite_archive import SQLiteArchive


class TestAssistantModeStateMachine(unittest.TestCase):
    """Test AssistantMode enum, string conversion, toggling, and persistence."""

    def setUp(self):
        self.archive = SQLiteArchive(db_path=":memory:")
        self.config = AppConfig(db_path=":memory:", default_mode="chat")
        self.assistant = AIAssistant(config=self.config, archive=self.archive)

    def tearDown(self):
        self.assistant.close()

    def test_default_mode(self):
        self.assertEqual(self.assistant.mode, AssistantMode.CHAT)
        self.assertEqual(self.assistant.mode.value, "chat")

    def test_set_mode_agent(self):
        mode = self.assistant.set_mode("agent")
        self.assertEqual(mode, AssistantMode.AGENT)
        self.assertEqual(self.assistant.mode, AssistantMode.AGENT)
        # Verify archive session update
        sess = self.assistant.get_current_session()
        self.assertEqual(sess.mode, "agent")

    def test_set_mode_chat(self):
        self.assistant.set_mode("agent")
        mode = self.assistant.set_mode("chat")
        self.assertEqual(mode, AssistantMode.CHAT)
        self.assertEqual(self.assistant.mode, AssistantMode.CHAT)
        sess = self.assistant.get_current_session()
        self.assertEqual(sess.mode, "chat")

    def test_chatbot_backward_compatibility(self):
        mode = self.assistant.set_mode("chatbot")
        self.assertEqual(mode, AssistantMode.CHAT)
        self.assertEqual(mode, AssistantMode.CHATBOT)

    def test_toggle_mode(self):
        self.assertEqual(self.assistant.mode, AssistantMode.CHAT)
        toggled = self.assistant.toggle_mode()
        self.assertEqual(toggled, AssistantMode.AGENT)
        self.assertEqual(self.assistant.mode, AssistantMode.AGENT)

        toggled_back = self.assistant.toggle_mode()
        self.assertEqual(toggled_back, AssistantMode.CHAT)
        self.assertEqual(self.assistant.mode, AssistantMode.CHAT)

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            self.assistant.set_mode("super_ai_mode")


class TestActionToolsSchemas(unittest.TestCase):
    """Verify tool schemas for Phase 7 agent capabilities."""

    def test_run_terminal_command_schema(self):
        tool = RUN_TERMINAL_COMMAND_TOOL
        self.assertEqual(tool["type"], "function")
        fn = tool["function"]
        self.assertEqual(fn["name"], "run_terminal_command")
        self.assertIn("command", fn["parameters"]["required"])
        self.assertIn("cwd", fn["parameters"]["properties"])
        self.assertIn("timeout", fn["parameters"]["properties"])

    def test_python_interpreter_schema(self):
        tool = PYTHON_INTERPRETER_TOOL
        self.assertEqual(tool["type"], "function")
        fn = tool["function"]
        self.assertEqual(fn["name"], "python_interpreter")
        self.assertIn("code", fn["parameters"]["required"])
        self.assertIn("timeout", fn["parameters"]["properties"])

    def test_read_file_schema(self):
        tool = READ_FILE_TOOL
        self.assertEqual(tool["type"], "function")
        fn = tool["function"]
        self.assertEqual(fn["name"], "read_file")
        self.assertIn("file_path", fn["parameters"]["required"])
        self.assertIn("max_lines", fn["parameters"]["properties"])

    def test_write_file_schema(self):
        tool = WRITE_FILE_TOOL
        self.assertEqual(tool["type"], "function")
        fn = tool["function"]
        self.assertEqual(fn["name"], "write_file")
        self.assertIn("file_path", fn["parameters"]["required"])
        self.assertIn("content", fn["parameters"]["required"])

    def test_list_directory_schema(self):
        tool = LIST_DIRECTORY_TOOL
        self.assertEqual(tool["type"], "function")
        fn = tool["function"]
        self.assertEqual(fn["name"], "list_directory")
        self.assertIn("directory_path", fn["parameters"]["properties"])

    def test_tool_groupings(self):
        self.assertEqual(len(CHAT_TOOLS), 2)
        chat_names = [t["function"]["name"] for t in CHAT_TOOLS]
        self.assertIn("search_past_conversations", chat_names)
        self.assertIn("load_session_transcript", chat_names)

        self.assertEqual(len(AGENT_ACTION_TOOLS), 5)
        action_names = [t["function"]["name"] for t in AGENT_ACTION_TOOLS]
        self.assertIn("run_terminal_command", action_names)
        self.assertIn("python_interpreter", action_names)
        self.assertIn("read_file", action_names)
        self.assertIn("write_file", action_names)
        self.assertIn("list_directory", action_names)

        self.assertEqual(len(AGENT_TOOLS), 7)


class TestToolDispatcherExecution(unittest.IsolatedAsyncioTestCase):
    """Test ToolDispatcher execution of action tools."""

    async def asyncSetUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.archive = SQLiteArchive(db_path=":memory:")
        self.config = AppConfig(db_path=":memory:")
        self.assistant = AIAssistant(config=self.config, archive=self.archive)
        self.dispatcher = ToolDispatcher(self.assistant)

    async def asyncTearDown(self):
        self.assistant.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    async def test_run_terminal_command_success(self):
        result = await self.dispatcher.execute(
            "run_terminal_command",
            {"command": "echo 'Hello Phase 7'"},
        )
        self.assertIn("Exit Code: 0", result)
        self.assertIn("Hello Phase 7", result)

    async def test_run_terminal_command_empty(self):
        result = await self.dispatcher.execute("run_terminal_command", {"command": ""})
        self.assertIn("Error: command is required", result)

    async def test_python_interpreter_success(self):
        result = await self.dispatcher.execute(
            "python_interpreter",
            {"code": "import math\nprint(f'SQRT: {math.isqrt(144)}')"},
        )
        self.assertIn("Exit Code: 0", result)
        self.assertIn("SQRT: 12", result)

    async def test_python_interpreter_syntax_error(self):
        result = await self.dispatcher.execute(
            "python_interpreter",
            {"code": "def bad_syntax("},
        )
        self.assertNotEqual(result.find("Exit Code:"), -1)
        self.assertIn("SyntaxError", result)

    async def test_file_system_operations(self):
        test_file = os.path.join(self.tmp_dir, "sub", "test.txt")

        # 1. Write file
        write_res = await self.dispatcher.execute(
            "write_file",
            {"file_path": test_file, "content": "Line 1: Hello\nLine 2: World"},
        )
        self.assertIn("Successfully wrote", write_res)
        self.assertTrue(os.path.exists(test_file))

        # 2. Read file
        read_res = await self.dispatcher.execute(
            "read_file",
            {"file_path": test_file},
        )
        self.assertIn("Line 1: Hello", read_res)
        self.assertIn("Line 2: World", read_res)

        # 3. Read non-existent file
        missing_res = await self.dispatcher.execute(
            "read_file",
            {"file_path": os.path.join(self.tmp_dir, "nonexistent.txt")},
        )
        self.assertIn("does not exist", missing_res)

        # 4. List directory
        list_res = await self.dispatcher.execute(
            "list_directory",
            {"directory_path": self.tmp_dir},
        )
        self.assertIn("Directory listing", list_res)
        self.assertIn("[DIR] sub", list_res)


class MockMultiStepAgentProvider(BaseLLMProvider):
    """Simulates an LLM agent that executes a sequence of tool calls across multiple turns."""

    def __init__(self, steps: List[Dict[str, Any]], final_answer: str):
        """
        steps is a list of turns where the model calls tools:
        [
            {"tool": "write_file", "args": {"file_path": "...", "content": "..."}},
            {"tool": "read_file", "args": {"file_path": "..."}}
        ]
        """
        self.steps = steps
        self.final_answer = final_answer
        self.call_count = 0
        self.tools_received_per_call: List[Optional[List[Dict[str, Any]]]] = []
        self.messages_received_per_call: List[List[Dict[str, Any]]] = []

    @property
    def name(self) -> str:
        return "MockMultiStepAgentProvider"

    async def health_check(self) -> bool:
        return True

    async def list_models(self) -> List[ModelInfo]:
        return [ModelInfo(name="mock-agent-model")]

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

        step_idx = self.call_count - 1
        if step_idx < len(self.steps):
            step = self.steps[step_idx]
            tool_name = step["tool"]
            tool_args = step["args"]

            yield StreamChunk(
                delta="",
                thinking_delta=f"Thinking about step {self.call_count}...",
                is_done=False,
            )
            yield StreamChunk(
                delta="",
                tool_calls=[{
                    "id": f"call_step_{self.call_count}",
                    "function": {
                        "name": tool_name,
                        "arguments": tool_args,
                    },
                }],
                is_done=False,
            )
            yield StreamChunk(
                delta="",
                is_done=True,
                stats=StreamStats(
                    ttft_ms=5.0,
                    total_duration_ms=20.0,
                    eval_count=10,
                ),
            )
        else:
            # Final turn: model outputs the solution
            yield StreamChunk(
                delta=self.final_answer,
                is_done=False,
            )
            yield StreamChunk(
                delta="",
                is_done=True,
                stats=StreamStats(
                    ttft_ms=5.0,
                    total_duration_ms=30.0,
                    eval_count=15,
                ),
            )


class TestChatVsAgentMultiTurnExecution(unittest.IsolatedAsyncioTestCase):
    """Test Chat mode single-hop restriction vs Agent mode sequential multi-turn loop."""

    async def asyncSetUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.archive = SQLiteArchive(db_path=":memory:")
        self.config = AppConfig(db_path=":memory:", agent_max_turns=5)

    async def asyncTearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    async def test_chat_mode_restricts_to_chat_tools_and_single_hop(self):
        """Verify that in chat mode:
        1. Only CHAT_TOOLS (search + transcript) are supplied to LLM.
        2. Strictly executes at most 1 retrieval cycle (Pass 1 tool call -> Pass 2 final answer).
        """
        chat_steps = [{
            "tool": "search_past_conversations",
            "args": {"query": "test conversation"},
        }]
        provider = MockMultiStepAgentProvider(
            steps=chat_steps,
            final_answer="Here is what was found in past conversations.",
        )
        assistant = AIAssistant(
            config=self.config,
            provider=provider,
            archive=self.archive,
            embedder=MockEmbedder(8),
        )
        assistant.set_mode(AssistantMode.CHAT)

        chunks = []
        async for chunk in assistant.chat_stream("Search for our prior talk"):
            chunks.append(chunk)

        # In chat mode: exactly 2 provider calls (Pass 1 with CHAT_TOOLS, Pass 2 with tools=None)
        self.assertEqual(provider.call_count, 2)
        # Pass 1 received CHAT_TOOLS (2 tools)
        self.assertEqual(len(provider.tools_received_per_call[0]), 2)
        # Pass 2 received tools=None (strict single-hop enforcement)
        self.assertIsNone(provider.tools_received_per_call[1])

        # Verify final assistant message in memory and archive
        asst_msg = assistant.memory.get_messages()[-1]
        self.assertEqual(asst_msg.content, "Here is what was found in past conversations.")
        saved_msgs = self.archive.get_messages(assistant.current_session.id)
        self.assertEqual(saved_msgs[-1].metadata.get("mode"), "chat")
        self.assertEqual(saved_msgs[-1].metadata.get("tools_executed"), ["search_past_conversations"])

        assistant.close()

    async def test_agent_mode_executes_sequential_multi_turn_loop(self):
        """Verify that in agent mode:
        1. Model receives AGENT_TOOLS (7 tools).
        2. Model executes sequential tool calls in a while loop across multiple turns.
        3. Real-time intermediate notices are emitted for each step.
        4. Final answer is produced and persisted with full multi-turn metadata.
        """
        test_file = os.path.join(self.tmp_dir, "calc.py")
        agent_steps = [
            {
                "tool": "write_file",
                "args": {"file_path": test_file, "content": "res = 17 * 23\nprint(f'VAL={res}')"},
            },
            {
                "tool": "python_interpreter",
                "args": {"code": "import sys\nwith open('" + test_file + "') as f: exec(f.read())"},
            },
        ]
        provider = MockMultiStepAgentProvider(
            steps=agent_steps,
            final_answer="The computation of 17 * 23 gives 391.",
        )
        assistant = AIAssistant(
            config=self.config,
            provider=provider,
            archive=self.archive,
            embedder=MockEmbedder(8),
        )
        assistant.set_mode(AssistantMode.AGENT)

        chunks: List[StreamChunk] = []
        async for chunk in assistant.chat_stream("Calculate 17 * 23 using a Python script"):
            chunks.append(chunk)

        # 3 turns: Step 1 (write_file) -> Step 2 (python_interpreter) -> Step 3 (final answer)
        self.assertEqual(provider.call_count, 3)

        # Verify AGENT_TOOLS passed during agent loop
        self.assertEqual(len(provider.tools_received_per_call[0]), 7)
        self.assertEqual(len(provider.tools_received_per_call[1]), 7)

        # Verify intermediate tool call notices emitted to user stream
        notices = [c.tool_call_notice for c in chunks if c.tool_call_notice]
        self.assertEqual(len(notices), 2)
        self.assertIn("Agent Step 1: Executing write_file", notices[0])
        self.assertIn("Agent Step 2: Executing python_interpreter", notices[1])

        # Verify file was actually created on disk by tool execution
        self.assertTrue(os.path.exists(test_file))

        # Verify final answer in memory and archive
        asst_msg = assistant.memory.get_messages()[-1]
        self.assertEqual(asst_msg.content, "The computation of 17 * 23 gives 391.")

        saved_msgs = self.archive.get_messages(assistant.current_session.id)
        saved_meta = saved_msgs[-1].metadata
        self.assertEqual(saved_meta.get("mode"), "agent")
        self.assertEqual(saved_meta.get("agent_turns"), 3)
        self.assertEqual(
            saved_meta.get("tools_executed"),
            ["write_file", "python_interpreter"],
        )
        self.assertEqual(len(saved_meta.get("tool_calls")), 2)

        assistant.close()

    async def test_agent_mode_max_turns_limit(self):
        """Verify that agent mode safely stops when reaching max_turns."""
        # Set max_turns=2, but model tries to make 5 tool calls
        cfg = AppConfig(db_path=":memory:", agent_max_turns=2)
        infinite_steps = [
            {"tool": "run_terminal_command", "args": {"command": "echo 'step 1'"}},
            {"tool": "run_terminal_command", "args": {"command": "echo 'step 2'"}},
            {"tool": "run_terminal_command", "args": {"command": "echo 'step 3'"}},
        ]
        provider = MockMultiStepAgentProvider(
            steps=infinite_steps,
            final_answer="Completed maximum allowed steps.",
        )
        assistant = AIAssistant(
            config=cfg,
            provider=provider,
            archive=self.archive,
            embedder=MockEmbedder(8),
        )
        assistant.set_mode(AssistantMode.AGENT)

        chunks = []
        async for chunk in assistant.chat_stream("Run loop"):
            chunks.append(chunk)

        # Should execute turn 1, turn 2, then synthesize with tools=None -> total 3 calls
        self.assertEqual(provider.call_count, 3)
        self.assertIsNone(provider.tools_received_per_call[2])

        assistant.close()


class TestGeminiProviderFunctionCalling(unittest.IsolatedAsyncioTestCase):
    """Test GeminiProvider tool conversion, message formatting, and function call streaming."""

    def setUp(self):
        self.provider = GeminiProvider(api_key="test-key")

    def test_convert_tools_format(self):
        gemini_tools = self.provider._convert_tools(CHAT_TOOLS)
        self.assertIsNotNone(gemini_tools)
        self.assertEqual(len(gemini_tools), 1)
        decs = gemini_tools[0]["functionDeclarations"]
        self.assertEqual(len(decs), 2)
        self.assertEqual(decs[0]["name"], "search_past_conversations")
        self.assertEqual(decs[1]["name"], "load_session_transcript")

    def test_convert_messages_with_tool_calls_and_responses(self):
        messages = [
            {"role": "user", "content": "What was discussed earlier?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_1",
                    "function": {"name": "search_past_conversations", "arguments": {"query": "python"}},
                }],
            },
            {
                "role": "tool",
                "name": "search_past_conversations",
                "content": "Found 1 session: Python discussion",
            },
        ]
        sys_inst, contents = self.provider._convert_messages(messages)
        self.assertIsNone(sys_inst)
        self.assertEqual(len(contents), 3)

        # User turn
        self.assertEqual(contents[0]["role"], "user")

        # Model turn with functionCall
        self.assertEqual(contents[1]["role"], "model")
        fc_part = contents[1]["parts"][0]
        self.assertIn("functionCall", fc_part)
        self.assertEqual(fc_part["functionCall"]["name"], "search_past_conversations")

        # Tool response turn with functionResponse
        self.assertEqual(contents[2]["role"], "function")
        fr_part = contents[2]["parts"][0]
        self.assertIn("functionResponse", fr_part)
        self.assertEqual(fr_part["functionResponse"]["name"], "search_past_conversations")
        self.assertEqual(fr_part["functionResponse"]["response"]["result"], "Found 1 session: Python discussion")


class TestLiveGeminiAgentModeIntegration(unittest.IsolatedAsyncioTestCase):
    """Live inference integration test using Gemini API key for agent mode multi-turn loop."""

    async def asyncSetUp(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            self.skipTest("GEMINI_API_KEY not configured in environment")
        self.provider = GeminiProvider(api_key=self.api_key)
        is_healthy = await self.provider.health_check()
        if not is_healthy:
            self.skipTest("Cannot connect to Gemini API with provided key")

        self.tmp_db = tempfile.mktemp(suffix=".db")
        self.archive = SQLiteArchive(db_path=self.tmp_db)
        self.config = AppConfig(
            provider="gemini",
            default_model="gemini-2.5-flash",
            db_path=self.tmp_db,
            agent_max_turns=5,
        )
        self.assistant = AIAssistant(
            config=self.config,
            provider=self.provider,
            model="gemini-2.5-flash",
            archive=self.archive,
            embedder=MockEmbedder(8),
        )
        self.assistant.set_mode(AssistantMode.AGENT)

    async def asyncTearDown(self):
        self.assistant.close()
        if os.path.exists(self.tmp_db):
            try:
                os.remove(self.tmp_db)
            except OSError:
                pass

    async def test_live_gemini_agent_executes_tool_and_answers(self):
        """Test live Gemini agent mode: executes python_interpreter and returns final answer."""
        prompt = (
            "What is 347 multiplied by 89? "
            "You MUST use the python_interpreter tool to compute the exact product. "
            "Then provide the final answer clearly."
        )

        response_tokens: List[str] = []
        tool_notices: List[str] = []

        try:
            async for chunk in self.assistant.chat_stream(prompt):
                if chunk.tool_call_notice:
                    tool_notices.append(chunk.tool_call_notice)
                if chunk.delta:
                    response_tokens.append(chunk.delta)
        except RuntimeError as e:
            if "429" in str(e) or "quota" in str(e).lower() or "RESOURCE_EXHAUSTED" in str(e):
                self.skipTest(f"Gemini API rate limit / quota exceeded: {e}")
            raise

        full_answer = "".join(response_tokens).strip()

        # 347 * 89 = 30883
        self.assertIn("30883", full_answer, f"Expected 30883 in response, got: {full_answer}")
        self.assertGreaterEqual(len(tool_notices), 1, "Gemini failed to invoke python_interpreter tool")
        self.assertIn("python_interpreter", tool_notices[0])

        # Verify session metadata in SQLite
        sess_messages = self.archive.get_messages(self.assistant.current_session.id)
        asst_record = sess_messages[-1]
        self.assertEqual(asst_record.metadata.get("mode"), "agent")
        self.assertIn("python_interpreter", asst_record.metadata.get("tools_executed", []))


if __name__ == "__main__":
    unittest.main()
