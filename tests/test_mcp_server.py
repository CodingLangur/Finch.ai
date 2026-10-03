"""Unit tests for Finch FastMCP Service (finch/mcp_server.py)."""
import asyncio
import io
import json
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

        archive.add_message(
            session_id=test_session_id,
            role="user",
            content="Can you verify the quantum flux capacitor calibration?",
        )
        archive.add_message(
            session_id=test_session_id,
            role="assistant",
            content=compressed_payload,  # Stored compressed
        )

        # 1. Test finch_search_history
        search_results = await finch_search_history("quantum flux capacitor", limit=5)
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

        finally:
            proc.kill()
            await proc.wait()


if __name__ == "__main__":
    unittest.main()

