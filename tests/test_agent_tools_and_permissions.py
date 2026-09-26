"""Unit tests for Agent Mode Tools: Terminal Access, Python, Web, and Granular Permission Toggles."""
import io
import os
import shutil
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from rich.console import Console

from ai_assistant.config import AppConfig
from ai_assistant.core.assistant import AIAssistant, AssistantMode
from ai_assistant.core.tools import (
    ALL_AGENT_TOOLS,
    CHAT_TOOLS,
    FETCH_WEB_PAGE_TOOL,
    LIST_DIRECTORY_TOOL,
    PYTHON_INTERPRETER_TOOL,
    READ_FILE_TOOL,
    RUN_TERMINAL_COMMAND_TOOL,
    TOOL_CATEGORIES,
    TOOL_NAME_TO_CATEGORY,
    WEB_SEARCH_TOOL,
    WEB_TOOLS,
    WRITE_FILE_TOOL,
    ToolDispatcher,
)
from ai_assistant.storage.sqlite_archive import SQLiteArchive
from ai_assistant.ui.cli import InteractiveCLI
from tests.test_agent_mode import MockEmbedder
from tests.test_session_summarizer import MockLLMProvider


class TestTerminalToolExecution(unittest.IsolatedAsyncioTestCase):
    """Test execution of terminal commands via run_terminal_command."""

    async def asyncSetUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.archive = SQLiteArchive(db_path=":memory:")
        self.config = AppConfig(
            db_path=":memory:",
            enable_terminal_tool=True,
            terminal_timeout=10.0,
        )
        self.assistant = AIAssistant(
            config=self.config,
            archive=self.archive,
            provider=MockLLMProvider(["ok"]),
            embedder=MockEmbedder(8),
        )
        self.assistant.set_mode(AssistantMode.AGENT)
        self.dispatcher = ToolDispatcher(self.assistant)

    async def asyncTearDown(self):
        self.assistant.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    async def test_terminal_command_stdout(self):
        """Verify successful command execution capturing stdout and exit code 0."""
        result = await self.dispatcher.execute(
            "run_terminal_command",
            {"command": "echo 'Testing Terminal Access 123'"},
        )
        self.assertIn("Exit Code: 0", result)
        self.assertIn("STDOUT:", result)
        self.assertIn("Testing Terminal Access 123", result)

    async def test_terminal_command_stderr(self):
        """Verify execution capturing stderr and non-zero exit code."""
        result = await self.dispatcher.execute(
            "run_terminal_command",
            {"command": "bash -c 'echo \"Fatal error encountered\" >&2; exit 42'"},
        )
        self.assertIn("Exit Code: 42", result)
        self.assertIn("STDERR:", result)
        self.assertIn("Fatal error encountered", result)

    async def test_terminal_command_cwd(self):
        """Verify executing command in a specific working directory."""
        sub_dir = os.path.join(self.tmp_dir, "custom_working_dir")
        os.makedirs(sub_dir, exist_ok=True)

        result = await self.dispatcher.execute(
            "run_terminal_command",
            {"command": "pwd", "cwd": sub_dir},
        )
        self.assertIn("Exit Code: 0", result)
        self.assertIn(os.path.realpath(sub_dir), os.path.realpath(result))

    async def test_terminal_command_invalid_cwd(self):
        """Verify error handling when cwd does not exist or is a file."""
        non_existent = os.path.join(self.tmp_dir, "ghost_folder")
        result = await self.dispatcher.execute(
            "run_terminal_command",
            {"command": "pwd", "cwd": non_existent},
        )
        self.assertIn("does not exist", result)

    async def test_terminal_command_timeout(self):
        """Verify that long-running commands terminate cleanly on timeout."""
        result = await self.dispatcher.execute(
            "run_terminal_command",
            {"command": "sleep 10", "timeout": 0.1},
        )
        self.assertIn("timed out after 0.1 seconds", result)

    async def test_terminal_command_empty(self):
        """Verify error when command is empty string."""
        result = await self.dispatcher.execute(
            "run_terminal_command",
            {"command": "   "},
        )
        self.assertIn("Error: command is required", result)

    async def test_terminal_command_truncation(self):
        """Verify that excessively large output is truncated safely."""
        # Generate 20,000 characters
        result = await self.dispatcher.execute(
            "run_terminal_command",
            {"command": "python -c 'print(\"A\" * 20000)'"},
        )
        self.assertIn("...[Output truncated]", result)
        self.assertLess(len(result), 15000)


class TestToolPermissionToggles(unittest.IsolatedAsyncioTestCase):
    """Test granular permission toggles for terminal, python, web, and file tools."""

    async def asyncSetUp(self):
        self.archive = SQLiteArchive(db_path=":memory:")
        self.config = AppConfig(
            db_path=":memory:",
            enable_terminal_tool=True,
            enable_python_tool=True,
            enable_web_tool=False,
            enable_file_tools=True,
        )
        self.assistant = AIAssistant(
            config=self.config,
            archive=self.archive,
            provider=MockLLMProvider(["ok"]),
            embedder=MockEmbedder(8),
        )
        self.assistant.set_mode(AssistantMode.AGENT)
        self.dispatcher = ToolDispatcher(self.assistant)

    async def asyncTearDown(self):
        self.assistant.close()

    def test_default_permissions_state(self):
        """Verify initial default permissions from config."""
        perms = self.assistant.get_tool_permissions()
        self.assertTrue(perms["terminal"])
        self.assertTrue(perms["python"])
        self.assertFalse(perms["web"])
        self.assertTrue(perms["files"])

        # In Agent mode with default permissions:
        active_tools = self.assistant.get_active_tools()
        active_names = [t["function"]["name"] for t in active_tools]
        self.assertIn("run_terminal_command", active_names)
        self.assertIn("python_interpreter", active_names)
        self.assertIn("read_file", active_names)
        self.assertNotIn("web_search", active_names)
        self.assertNotIn("fetch_web_page", active_names)

    async def test_disable_terminal_tool_blocks_execution(self):
        """Verify disabling terminal access removes tool from schema and blocks execution."""
        # 1. Disable terminal access
        self.assistant.set_tool_permission("terminal", False)
        self.assertFalse(self.assistant.is_tool_category_enabled("terminal"))

        # Verify excluded from LLM tools schema
        active_tools = self.assistant.get_active_tools()
        active_names = [t["function"]["name"] for t in active_tools]
        self.assertNotIn("run_terminal_command", active_names)

        # Verify dispatcher blocks execution with polite explanation
        result = await self.dispatcher.execute(
            "run_terminal_command",
            {"command": "echo 'Should not run'"},
        )
        self.assertIn("Error: Access to tool 'run_terminal_command' is disabled", result)
        self.assertIn("terminal", result)

    async def test_re_enable_terminal_tool_allows_execution(self):
        """Verify re-enabling terminal access restores tool to schema and allows execution."""
        self.assistant.set_tool_permission("terminal", False)
        self.assistant.set_tool_permission("terminal", True)
        self.assertTrue(self.assistant.is_tool_category_enabled("terminal"))

        active_tools = self.assistant.get_active_tools()
        active_names = [t["function"]["name"] for t in active_tools]
        self.assertIn("run_terminal_command", active_names)

        result = await self.dispatcher.execute(
            "run_terminal_command",
            {"command": "echo 'Allowed again'"},
        )
        self.assertIn("Exit Code: 0", result)
        self.assertIn("Allowed again", result)

    async def test_toggle_python_tool(self):
        """Verify disabling and enabling python tool."""
        self.assistant.set_tool_permission("python", False)
        active_tools = self.assistant.get_active_tools()
        active_names = [t["function"]["name"] for t in active_tools]
        self.assertNotIn("python_interpreter", active_names)

        result = await self.dispatcher.execute(
            "python_interpreter",
            {"code": "print(1 + 1)"},
        )
        self.assertIn("disabled", result)

        # Re-enable
        self.assistant.toggle_tool_permission("python")
        self.assertTrue(self.assistant.is_tool_category_enabled("python"))
        result_ok = await self.dispatcher.execute(
            "python_interpreter",
            {"code": "print('Python is active')"},
        )
        self.assertIn("Exit Code: 0", result_ok)
        self.assertIn("Python is active", result_ok)

    async def test_toggle_web_tool(self):
        """Verify disabling and enabling web tool (web_search & fetch_web_page)."""
        # Default is False
        self.assertFalse(self.assistant.is_tool_category_enabled("web"))
        result = await self.dispatcher.execute(
            "web_search",
            {"query": "Python news"},
        )
        self.assertIn("disabled", result)

        # Enable web tool
        self.assistant.set_tool_permission("web", True)
        self.assertTrue(self.assistant.is_tool_category_enabled("web"))

        active_tools = self.assistant.get_active_tools()
        active_names = [t["function"]["name"] for t in active_tools]
        self.assertIn("web_search", active_names)
        self.assertIn("fetch_web_page", active_names)

    async def test_toggle_file_tools(self):
        """Verify disabling and enabling filesystem tools (read_file, write_file, list_directory)."""
        self.assistant.set_tool_permission("files", False)
        active_tools = self.assistant.get_active_tools()
        active_names = [t["function"]["name"] for t in active_tools]
        self.assertNotIn("read_file", active_names)
        self.assertNotIn("write_file", active_names)
        self.assertNotIn("list_directory", active_names)

        res_read = await self.dispatcher.execute("read_file", {"file_path": "any.txt"})
        self.assertIn("disabled", res_read)

        res_write = await self.dispatcher.execute("write_file", {"file_path": "any.txt", "content": "x"})
        self.assertIn("disabled", res_write)

        res_list = await self.dispatcher.execute("list_directory", {"directory_path": "."})
        self.assertIn("disabled", res_list)

    def test_chat_mode_strictly_retains_memory_tools_only(self):
        """Verify that in Chat mode, only memory retrieval tools are provided regardless of toggles."""
        self.assistant.set_mode(AssistantMode.CHAT)
        self.assistant.set_tool_permission("terminal", True)
        self.assistant.set_tool_permission("python", True)
        self.assistant.set_tool_permission("web", True)
        self.assistant.set_tool_permission("files", True)

        active_tools = self.assistant.get_active_tools()
        active_names = [t["function"]["name"] for t in active_tools]

        self.assertEqual(len(active_tools), 2)
        self.assertIn("search_past_conversations", active_names)
        self.assertIn("load_session_transcript", active_names)
        self.assertNotIn("run_terminal_command", active_names)
        self.assertNotIn("python_interpreter", active_names)


class TestCLIToolsCommand(unittest.TestCase):
    """Test CLI /tools and /toggle interactive commands."""

    def setUp(self):
        self.archive = SQLiteArchive(db_path=":memory:")
        self.config = AppConfig(db_path=":memory:")
        self.assistant = AIAssistant(
            config=self.config,
            archive=self.archive,
            provider=MockLLMProvider(["ok"]),
            embedder=MockEmbedder(8),
        )
        self.assistant.set_mode(AssistantMode.AGENT)
        self.cli = InteractiveCLI(self.assistant)
        self.string_io = io.StringIO()
        self.cli.console = Console(file=self.string_io, color_system=None)

    def tearDown(self):
        self.assistant.close()

    def test_tools_table_display(self):
        """Verify /tools with no arguments displays the status table."""
        self.cli.handle_tools_command("")
        output = self.string_io.getvalue()
        self.assertIn("Terminal", output)
        self.assertIn("Python", output)
        self.assertIn("Web", output)
        self.assertIn("Files", output)

    def test_tools_toggle_terminal_off_and_on(self):
        """Verify /tools terminal off disables terminal access, and on enables it."""
        self.cli.handle_tools_command("terminal off")
        self.assertFalse(self.assistant.is_tool_category_enabled("terminal"))
        self.assertIn("DISABLED", self.string_io.getvalue())

        self.cli.handle_tools_command("terminal on")
        self.assertTrue(self.assistant.is_tool_category_enabled("terminal"))
        self.assertIn("ENABLED", self.string_io.getvalue())

    def test_tools_toggle_web_on(self):
        """Verify /tools web on enables web tools."""
        self.cli.handle_tools_command("web on")
        self.assertTrue(self.assistant.is_tool_category_enabled("web"))

    def test_tools_invalid_category(self):
        """Verify /tools with invalid category displays error message."""
        self.cli.handle_tools_command("invalid_cat on")
        self.assertIn("Unknown tool category", self.string_io.getvalue())


if __name__ == "__main__":
    unittest.main()
