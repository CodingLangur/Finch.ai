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


if __name__ == "__main__":
    unittest.main()
