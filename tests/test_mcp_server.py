"""Unit tests for Finch FastMCP Service (finch/mcp_server.py)."""
import asyncio
import io
import json
import os
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout

from finch.mcp_server import mcp


class TestFinchFastMCPServer(unittest.IsolatedAsyncioTestCase):
    def test_clean_stdout_on_import(self):
        """Verify that importing or accessing mcp_server produces 0 bytes on stdout."""
        buf_out = io.StringIO()
        buf_err = io.StringIO()

        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            # Access mcp tools
            name = mcp.name
            self.assertEqual(name, "Finch")

        self.assertEqual(buf_out.getvalue(), "", "Stdout was polluted during module access!")

    async def test_tools_registered(self):
        """Verify core Finch capabilities are exposed as MCP tools."""
        tools = await mcp.list_tools()
        tool_names = [t.name for t in tools]

        expected_tools = [
            "finch_remember_fact",
            "finch_get_user_facts",
            "finch_search_history",
            "finch_get_transcript",
            "remember_fact",
            "list_facts",
            "forget_fact",
            "search_past_conversations",
            "load_transcript",
            "compress_context",
            "get_storage_stats",
            "run_finch_query",
            "finch_run_subagent",
            "run_subagent",
        ]

        for expected in expected_tools:
            self.assertIn(expected, tool_names, f"Missing expected MCP tool: {expected}")

    async def test_finch_remember_and_get_user_facts(self):
        """Test finch_remember_fact and finch_get_user_facts tool execution."""
        from finch.mcp_server import finch_remember_fact, finch_get_user_facts

        # Remember a fact
        res1 = finch_remember_fact("Prefers dark mode in IDE", category="preference")
        self.assertIsInstance(res1, dict)
        self.assertIn("id", res1)
        self.assertEqual(res1["status"], "stored")
        self.assertEqual(res1["category"], "preference")

        res2 = finch_remember_fact("Uses SQLite for local storage", category="tech_stack")
        self.assertIsInstance(res2, dict)
        self.assertEqual(res2["status"], "stored")

        # Fetch all facts (category="")
        all_facts = finch_get_user_facts(category="")
        self.assertIsInstance(all_facts, list)
        self.assertGreaterEqual(len(all_facts), 2)
        fact_texts = [f["fact"] for f in all_facts]
        self.assertIn("Prefers dark mode in IDE", fact_texts)
        self.assertIn("Uses SQLite for local storage", fact_texts)

        # Filter by category
        pref_facts = finch_get_user_facts(category="preference")
        self.assertTrue(all(f["category"] == "preference" for f in pref_facts))
        self.assertIn("Prefers dark mode in IDE", [f["fact"] for f in pref_facts])

    async def test_finch_search_history_and_get_transcript_zstd(self):
        """Test finch_search_history and finch_get_transcript with zstd-decompressed records."""
        from finch.mcp_server import get_archive, finch_search_history, finch_get_transcript
        import uuid

        archive = get_archive()
        test_session_id = f"test_sess_{uuid.uuid4().hex[:8]}"
        archive.create_session(
            session_id=test_session_id,
            title="FastMCP Retrieval Test Session",
            model="gemini-2.5-flash",
            mode="chat",
        )

        # Add messages, including one large enough or explicitly zstd compressed
        large_payload = "Hyperdrive warp coil calibration diagnostics sequence 99482. " * 30
        compressed_payload, was_compressed = archive.compressor.compress_text(large_payload)
        self.assertTrue(was_compressed)
        self.assertTrue(archive.compressor.is_compressed(compressed_payload))

        try:
            archive.add_message(
                session_id=test_session_id,
                role="user",
                content=f"Can you verify the quantum flux capacitor calibration {test_session_id}?",
            )
            archive.add_message(
                session_id=test_session_id,
                role="assistant",
                content=compressed_payload,  # Stored compressed
            )

            # 1. Test finch_search_history
            search_results = await finch_search_history(f"quantum flux capacitor {test_session_id}", limit=5)
            self.assertIsInstance(search_results, list)
            found_sessions = [r["session_id"] for r in search_results]
            self.assertIn(test_session_id, found_sessions)

            # 2. Test finch_get_transcript
            transcript_data = finch_get_transcript(test_session_id)
            self.assertIsInstance(transcript_data, dict)
            self.assertEqual(transcript_data["session_id"], test_session_id)
            self.assertIn("turns", transcript_data)
            turns = transcript_data["turns"]
            self.assertEqual(len(turns), 2)

            # Verify structured turn details
            self.assertEqual(turns[0]["role"], "user")
            self.assertIn("quantum flux capacitor", turns[0]["content"])

            self.assertEqual(turns[1]["role"], "assistant")
            # Ensure zstd record was decompressed back to original string!
            self.assertFalse(turns[1]["content"].startswith("__ZSTD__:"))
            self.assertEqual(turns[1]["content"], large_payload)
        finally:
            archive.delete_session(test_session_id)

    async def test_finch_run_subagent_filesystem_and_python(self):
        """Test finch_run_subagent executing filesystem and Python REPL tasks headlessly in agent mode."""
        from finch.mcp_server import finch_run_subagent, get_archive
        from finch.providers.base import BaseLLMProvider, ModelInfo, StreamChunk, StreamStats
        from finch.core.assistant import AIAssistant
        from unittest.mock import patch
        import tempfile
        import shutil

        tmp_dir = tempfile.mkdtemp()
        test_file = os.path.join(tmp_dir, "generated_script.py")

        try:
            class MockSubagentProvider(BaseLLMProvider):
                def __init__(self):
                    self.call_count = 0

                @property
                def name(self) -> str:
                    return "MockSubagentProvider"

                async def health_check(self) -> bool:
                    return True

                async def list_models(self) -> List[ModelInfo]:
                    return [ModelInfo(name="mock-subagent-model")]

                async def stream_chat(
                    self,
                    messages: List[Dict[str, Any]],
                    model: str,
                    options: Optional[Dict[str, Any]] = None,
                    tools: Optional[List[Dict[str, Any]]] = None,
                ) -> AsyncGenerator[StreamChunk, None]:
                    self.call_count += 1
                    if self.call_count == 1:
                        yield StreamChunk(
                            thinking_delta="Step 1: Write Python script file to disk...",
                            tool_calls=[{
                                "id": "call_1",
                                "function": {
                                    "name": "write_file",
                                    "arguments": {
                                        "file_path": test_file,
                                        "content": "val = 40 + 2\nprint(f'COMPUTED:{val}')\n",
                                    },
                                },
                            }],
                        )
                    elif self.call_count == 2:
                        yield StreamChunk(
                            thinking_delta="Step 2: Execute script via Python interpreter...",
                            tool_calls=[{
                                "id": "call_2",
                                "function": {
                                    "name": "python_interpreter",
                                    "arguments": {
                                        "code": f"with open(r'{test_file}') as f: exec(f.read())",
                                    },
                                },
                            }],
                        )
                    else:
                        yield StreamChunk(
                            delta="Python script generated and executed successfully with result COMPUTED:42.",
                            is_done=True,
                            stats=StreamStats(ttft_ms=12.0, total_duration_ms=45.0, eval_count=30),
                        )

            mock_prov = MockSubagentProvider()
            original_init = AIAssistant.__init__

            def custom_init(assistant_self, *args, **kwargs):
                kwargs["provider"] = mock_prov
                original_init(assistant_self, *args, **kwargs)

            with patch.object(AIAssistant, "__init__", custom_init):
                result = await finch_run_subagent(
                    instruction="Create a script that computes 42 and execute it with Python REPL.",
                    max_turns=6,
                )

            self.assertIsInstance(result, dict)
            self.assertEqual(result["status"], "completed")
            self.assertIn("COMPUTED:42", result["response"])
            self.assertIn("write_file", result["tools_executed"])
            self.assertIn("python_interpreter", result["tools_executed"])
            self.assertGreaterEqual(result["agent_turns"], 2)
            self.assertEqual(result["max_turns"], 6)

            # Check that file was created on disk
            self.assertTrue(os.path.exists(test_file))
            with open(test_file, "r", encoding="utf-8") as f:
                self.assertIn("COMPUTED", f.read())

            # Check that conversations.db was updated with agent execution metadata
            archive = get_archive()
            session_id = result["session_id"]
            db_session = archive.get_session(session_id)
            self.assertIsNotNone(db_session)
            self.assertEqual(db_session.mode, "agent")
            self.assertIn("Subagent:", db_session.title)

            db_messages = archive.get_messages(session_id)
            self.assertGreaterEqual(len(db_messages), 2)
            assistant_record = [m for m in db_messages if m.role == "assistant"][-1]
            self.assertIsNotNone(assistant_record.metadata)
            self.assertEqual(assistant_record.metadata["mode"], "agent")
            self.assertEqual(assistant_record.metadata["max_turns"], 6)
            self.assertIn("write_file", assistant_record.metadata["tools_executed"])
            self.assertIn("python_interpreter", assistant_record.metadata["tools_executed"])

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    async def test_finch_run_subagent_shell_execution(self):
        """Test finch_run_subagent executing a delegated shell execution command."""
        from finch.mcp_server import run_subagent, get_archive
        from finch.providers.base import BaseLLMProvider, ModelInfo, StreamChunk, StreamStats
        from finch.core.assistant import AIAssistant
        from unittest.mock import patch

        class MockShellAgentProvider(BaseLLMProvider):
            def __init__(self):
                self.call_count = 0

            @property
            def name(self) -> str:
                return "MockShellAgentProvider"

            async def health_check(self) -> bool:
                return True

            async def list_models(self) -> List[ModelInfo]:
                return [ModelInfo(name="mock-shell-model")]

            async def stream_chat(
                self,
                messages: List[Dict[str, Any]],
                model: str,
                options: Optional[Dict[str, Any]] = None,
                tools: Optional[List[Dict[str, Any]]] = None,
            ) -> AsyncGenerator[StreamChunk, None]:
                self.call_count += 1
                if self.call_count == 1:
                    yield StreamChunk(
                        thinking_delta="Running shell echo command...",
                        tool_calls=[{
                            "id": "call_sh_1",
                            "function": {
                                "name": "run_terminal_command",
                                "arguments": {
                                    "command": "echo 'Subagent Shell Ping OK'",
                                },
                            },
                        }],
                    )
                else:
                    yield StreamChunk(
                        delta="Shell command succeeded: Subagent Shell Ping OK",
                        is_done=True,
                        stats=StreamStats(ttft_ms=8.0, total_duration_ms=30.0, eval_count=18),
                    )

        mock_prov = MockShellAgentProvider()
        original_init = AIAssistant.__init__

        def custom_init(assistant_self, *args, **kwargs):
            kwargs["provider"] = mock_prov
            original_init(assistant_self, *args, **kwargs)

        with patch.object(AIAssistant, "__init__", custom_init):
            # Test the run_subagent alias as well
            result = await run_subagent(
                instruction="Run echo 'Subagent Shell Ping OK' in terminal",
                max_turns=4,
            )

        self.assertEqual(result["status"], "completed")
        self.assertIn("Subagent Shell Ping OK", result["response"])
        self.assertIn("run_terminal_command", result["tools_executed"])
        self.assertEqual(result["max_turns"], 4)

        archive = get_archive()
        session_id = result["session_id"]
        db_messages = archive.get_messages(session_id)
        assistant_record = [m for m in db_messages if m.role == "assistant"][-1]
        self.assertIn("run_terminal_command", assistant_record.metadata["tools_executed"])

    async def test_resources_registered(self):
        """Verify Finch resources are exposed over MCP."""
        resources = await mcp.list_resources()
        resource_uris = [str(r.uri) for r in resources]

        expected_uris = [
            "finch://facts",
            "finch://persona",
            "finch://storage",
        ]

        for expected in expected_uris:
            self.assertIn(expected, resource_uris, f"Missing expected MCP resource: {expected}")

    async def test_stdio_jsonrpc_initialize(self):
        """Verify the STDIO server boots silently and responds to standard MCP JSON-RPC initialize."""
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "finch.mcp_server",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        init_req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0.0"},
            },
        }

        try:
            proc.stdin.write(json.dumps(init_req).encode("utf-8") + b"\n")
            await proc.stdin.drain()

            line = await asyncio.wait_for(proc.stdout.readline(), timeout=5.0)
            line_str = line.decode("utf-8").strip()

            data = json.loads(line_str)
            self.assertEqual(data.get("id"), 1)
            self.assertIn("result", data)
            self.assertIn("protocolVersion", data["result"])
        finally:
            proc.kill()
            await proc.wait()

    async def test_stdio_jsonrpc_tools_call(self):
        """Verify calling finch_* endpoints over STDIO JSON-RPC transport."""
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "finch.mcp_server",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            # 1. Initialize
            init_req = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "test-client", "version": "1.0.0"},
                },
            }
            proc.stdin.write(json.dumps(init_req).encode("utf-8") + b"\n")
            await proc.stdin.drain()
            line = await asyncio.wait_for(proc.stdout.readline(), timeout=5.0)
            init_resp = json.loads(line.decode("utf-8").strip())
            self.assertEqual(init_resp.get("id"), 1)

            # Initialized notification
            init_notif = {"jsonrpc": "2.0", "method": "notifications/initialized"}
            proc.stdin.write(json.dumps(init_notif).encode("utf-8") + b"\n")
            await proc.stdin.drain()

            # 2. Call finch_remember_fact
            call_req = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "finch_remember_fact",
                    "arguments": {"fact": "JSON-RPC test fact", "category": "rpc_test"},
                },
            }
            proc.stdin.write(json.dumps(call_req).encode("utf-8") + b"\n")
            await proc.stdin.drain()
            line = await asyncio.wait_for(proc.stdout.readline(), timeout=5.0)
            call_resp = json.loads(line.decode("utf-8").strip())
            self.assertEqual(call_resp.get("id"), 2)
            self.assertIn("result", call_resp)
            self.assertFalse(call_resp["result"].get("isError", False))

            # 3. Call finch_get_user_facts
            get_req = {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "finch_get_user_facts",
                    "arguments": {"category": "rpc_test"},
                },
            }
            proc.stdin.write(json.dumps(get_req).encode("utf-8") + b"\n")
            await proc.stdin.drain()
            line = await asyncio.wait_for(proc.stdout.readline(), timeout=5.0)
            get_resp = json.loads(line.decode("utf-8").strip())
            self.assertEqual(get_resp.get("id"), 3)
            self.assertIn("result", get_resp)
            self.assertFalse(get_resp["result"].get("isError", False))

            # 4. List tools over JSON-RPC and verify finch_run_subagent is exposed
            list_tools_req = {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/list",
                "params": {},
            }
            proc.stdin.write(json.dumps(list_tools_req).encode("utf-8") + b"\n")
            await proc.stdin.drain()
            line = await asyncio.wait_for(proc.stdout.readline(), timeout=5.0)
            list_resp = json.loads(line.decode("utf-8").strip())
            self.assertEqual(list_resp.get("id"), 4)
            tool_names = [t["name"] for t in list_resp["result"]["tools"]]
            self.assertIn("finch_run_subagent", tool_names)
            self.assertIn("run_subagent", tool_names)

        finally:
            proc.kill()
            await proc.wait()


if __name__ == "__main__":
    unittest.main()

