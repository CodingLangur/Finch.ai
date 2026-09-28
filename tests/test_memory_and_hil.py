"""Comprehensive tests for:
1. Long-Term Fact Store / Core User Memory (Separate from Personality)
2. Human-in-the-Loop Confirmation for Agent Tools (3-state: OFF, ASK, AUTO with warning)
3. Markdown Transcript & Rich HTML Export
4. Backup & Restore integration with facts
"""
import asyncio
import io
import json
import os
import shutil
import tempfile
import unittest
from typing import Any, Dict, List, Optional
from rich.console import Console

from ai_assistant.config import AppConfig
from ai_assistant.core.assistant import AIAssistant, AssistantMode, ToolPermissionMode
from ai_assistant.core.tools import (
    ToolDispatcher,
    RUN_TERMINAL_COMMAND_TOOL,
    PYTHON_INTERPRETER_TOOL,
    READ_FILE_TOOL,
    WRITE_FILE_TOOL,
    SAVE_USER_FACT_TOOL,
    LIST_USER_FACTS_TOOL,
)
from ai_assistant.embeddings.embedder import MockEmbedder
from ai_assistant.providers.base import BaseLLMProvider, ModelInfo, StreamChunk, StreamStats
from ai_assistant.storage.sqlite_archive import SQLiteArchive
from ai_assistant.storage.backup import export_backup_bundle, import_backup_bundle
from ai_assistant.storage.transcript_exporter import export_transcript_markdown, export_transcript_html
from ai_assistant.ui.cli import InteractiveCLI


class MockProvider(BaseLLMProvider):
    def __init__(self, response_text: str = "Test response"):
        self.response_text = response_text

    @property
    def name(self) -> str:
        return "mock"

    async def list_models(self) -> List[ModelInfo]:
        return [ModelInfo(name="mock-model", modified_at="", size=0)]

    async def stream_chat(self, messages, model=None, stream_stats=False, tools=None):
        yield StreamChunk(delta=self.response_text)
        if stream_stats:
            yield StreamChunk(stats=StreamStats(prompt_eval_count=10, eval_count=20, total_duration_ns=1000000))

    async def health_check(self) -> bool:
        return True


class TestUserFactStore(unittest.TestCase):
    """Test core user memory CRUD operations and isolation from personality.md."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "test_facts.db")
        self.persona_path = os.path.join(self.tmp_dir, "test_persona.md")
        with open(self.persona_path, "w", encoding="utf-8") as f:
            f.write("# Identity\nYou are Finch, a helpful coding assistant.\n")

        self.archive = SQLiteArchive(db_path=self.db_path)
        self.config = AppConfig(
            db_path=self.db_path,
            personality_path=self.persona_path,
        )
        self.assistant = AIAssistant(
            config=self.config,
            archive=self.archive,
            provider=MockProvider(),
            embedder=MockEmbedder(8),
        )

    def tearDown(self):
        self.assistant.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_add_and_list_facts(self):
        fact1_id = self.assistant.add_user_fact("User prefers Python over JavaScript", category="preference")
        fact2_id = self.assistant.add_user_fact("Project root is /workspace/finch", category="project")

        self.assertIsInstance(fact1_id, int)
        self.assertIsInstance(fact2_id, int)

        facts = self.assistant.list_user_facts()
        self.assertEqual(len(facts), 2)
        fact_texts = [f["fact"] for f in facts]
        self.assertIn("User prefers Python over JavaScript", fact_texts)
        self.assertIn("Project root is /workspace/finch", fact_texts)

    def test_filter_facts_by_category(self):
        self.assistant.add_user_fact("Fact A", category="preference")
        self.assistant.add_user_fact("Fact B", category="project")
        self.assistant.add_user_fact("Fact C", category="preference")

        prefs = self.assistant.list_user_facts(category="preference")
        self.assertEqual(len(prefs), 2)
        for p in prefs:
            self.assertEqual(p["category"], "preference")

    def test_delete_and_wipe_facts(self):
        f1_id = self.assistant.add_user_fact("Fact 1")
        f2_id = self.assistant.add_user_fact("Fact 2")

        # Delete single fact
        deleted = self.assistant.delete_user_fact(f1_id)
        self.assertTrue(deleted)
        self.assertEqual(len(self.assistant.list_user_facts()), 1)

        # Wipe remaining facts
        wiped_count = self.assistant.wipe_user_facts()
        self.assertEqual(wiped_count, 1)
        self.assertEqual(len(self.assistant.list_user_facts()), 0)

    def test_facts_injected_into_effective_system_prompt_without_modifying_persona(self):
        # Initial personality file content
        with open(self.persona_path, "r", encoding="utf-8") as f:
            original_persona_content = f.read()

        self.assistant.add_user_fact("User's timezone is UTC+5:30", category="environment")
        self.assistant.add_user_fact("Favorite editor is Neovim", category="preference")

        prompt = self.assistant.build_effective_system_prompt()
        self.assertIn("### Verified User Facts & Long-Term Memory", prompt)
        self.assertIn("User's timezone is UTC+5:30", prompt)
        self.assertIn("Favorite editor is Neovim", prompt)

        # Ensure personality.md was NOT touched
        with open(self.persona_path, "r", encoding="utf-8") as f:
            current_persona_content = f.read()
        self.assertEqual(original_persona_content, current_persona_content)


class TestHumanInTheLoopToolPermissions(unittest.IsolatedAsyncioTestCase):
    """Test 3-state tool permissions: OFF, ASK (Confirm), AUTO (Complete AI Control)."""

    async def asyncSetUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "test_hil.db")
        self.archive = SQLiteArchive(db_path=self.db_path)
        self.config = AppConfig(db_path=self.db_path)
        self.assistant = AIAssistant(
            config=self.config,
            archive=self.archive,
            provider=MockProvider(),
            embedder=MockEmbedder(8),
        )
        self.assistant.set_mode(AssistantMode.AGENT)
        self.dispatcher = ToolDispatcher(self.assistant)

    async def asyncTearDown(self):
        self.assistant.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_three_state_toggling_cycle(self):
        # Default terminal is ask
        self.assistant.set_tool_permission("terminal", "off")
        self.assertEqual(self.assistant.get_tool_permission_mode("terminal"), "off")

        # Cycle 1: off -> ask
        mode1 = self.assistant.toggle_tool_permission("terminal")
        self.assertEqual(mode1, "ask")

        # Cycle 2: ask -> auto
        mode2 = self.assistant.toggle_tool_permission("terminal")
        self.assertEqual(mode2, "auto")

        # Cycle 3: auto -> off
        mode3 = self.assistant.toggle_tool_permission("terminal")
        self.assertEqual(mode3, "off")

    async def test_tool_execution_when_off(self):
        self.assistant.set_tool_permission("terminal", "off")
        res = await self.dispatcher.execute("run_terminal_command", {"command": "echo 'hi'"})
        self.assertIn("disabled", res)

    async def test_tool_execution_in_ask_mode_confirmed(self):
        self.assistant.set_tool_permission("terminal", "ask")

        callback_called = False
        def confirm_callback(tool_name, arguments):
            nonlocal callback_called
            callback_called = True
            self.assertEqual(tool_name, "run_terminal_command")
            self.assertIn("echo 'approved'", arguments.get("command", ""))
            return True

        self.dispatcher.confirmation_callback = confirm_callback

        res = await self.dispatcher.execute("run_terminal_command", {"command": "echo 'approved'"})
        self.assertTrue(callback_called)
        self.assertIn("Exit Code: 0", res)
        self.assertIn("approved", res)

    async def test_tool_execution_in_ask_mode_rejected(self):
        self.assistant.set_tool_permission("terminal", "ask")

        def reject_callback(tool_name, arguments):
            return False

        self.dispatcher.confirmation_callback = reject_callback

        res = await self.dispatcher.execute("run_terminal_command", {"command": "echo 'rejected'"})
        self.assertIn("Execution cancelled: User denied permission", res)

    async def test_tool_execution_in_auto_mode_bypasses_callback(self):
        self.assistant.set_tool_permission("terminal", "auto")

        callback_called = False
        def confirm_callback(tool_name, arguments):
            nonlocal callback_called
            callback_called = True
            return False  # If called, would reject

        self.dispatcher.confirmation_callback = confirm_callback

        res = await self.dispatcher.execute("run_terminal_command", {"command": "echo 'auto-run'"})
        self.assertFalse(callback_called, "Confirmation callback should not be invoked in AUTO mode")
        self.assertIn("Exit Code: 0", res)
        self.assertIn("auto-run", res)


class TestCLIToolWarningAndMemoryCommands(unittest.TestCase):
    """Test CLI /tools auto warning, /remember, /forget, and /facts commands."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.archive = SQLiteArchive(db_path=":memory:")
        self.config = AppConfig(db_path=":memory:")
        self.assistant = AIAssistant(
            config=self.config,
            archive=self.archive,
            provider=MockProvider(),
            embedder=MockEmbedder(8),
        )
        self.assistant.set_mode(AssistantMode.AGENT)
        self.cli = InteractiveCLI(self.assistant)
        self.string_io = io.StringIO()
        self.cli.console = Console(file=self.string_io, color_system=None)

    def tearDown(self):
        self.assistant.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_cli_tools_auto_displays_warning(self):
        self.cli.handle_tools_command("terminal auto")
        output = self.string_io.getvalue()
        self.assertIn("ENABLED (AUTO", output)
        self.assertIn("WARNING: Complete AI Control enabled for 'TERMINAL'", output)
        self.assertIn("human confirmation", output)

    def test_cli_remember_and_facts_command(self):
        self.cli.handle_remember_command("User prefers tabs over spaces")
        output = self.string_io.getvalue()
        self.assertIn("Stored in long-term memory", output)

        self.string_io.truncate(0)
        self.string_io.seek(0)
        self.cli.handle_facts_command("")
        facts_output = self.string_io.getvalue()
        self.assertIn("User prefers tabs", facts_output)
        self.assertIn("spaces", facts_output)

    def test_cli_forget_command(self):
        fact_id = self.assistant.add_user_fact("Temporary note")

        self.cli.handle_forget_command(str(fact_id))
        output = self.string_io.getvalue()
        self.assertIn(f"Fact #{fact_id} removed", output)
        self.assertEqual(len(self.assistant.list_user_facts()), 0)


class TestTranscriptExports(unittest.TestCase):
    """Test Markdown and HTML transcript export functionality."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.archive = SQLiteArchive(db_path=":memory:")
        self.config = AppConfig(db_path=":memory:")
        self.assistant = AIAssistant(
            config=self.config,
            archive=self.archive,
            provider=MockProvider(),
            embedder=MockEmbedder(8),
        )
        # Populate a session with messages
        self.session = self.assistant.current_session
        self.archive.add_message(self.session.id, "user", "Can you list the current directory?")
        self.archive.add_message(
            self.session.id,
            "assistant",
            "<think>Checking directory contents...</think>Here are the files: README.md, main.py",
            metadata={"tools_executed": ["list_directory"]},
        )

    def tearDown(self):
        self.assistant.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_export_markdown_transcript(self):
        out_path = os.path.join(self.tmp_dir, "transcript.md")
        result_path = self.assistant.export_transcript_markdown(session_id=self.session.id, output_path=out_path)
        self.assertTrue(os.path.exists(result_path))

        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("# Conversation Transcript", content)
        self.assertIn(f"**Session ID**: `{self.session.id}`", content)
        self.assertIn("### 👤 User", content)
        self.assertIn("Can you list the current directory?", content)
        self.assertIn("### 🤖 Assistant", content)
        self.assertIn("Here are the files: README.md, main.py", content)

    def test_export_html_transcript(self):
        out_path = os.path.join(self.tmp_dir, "transcript.html")
        result_path = self.assistant.export_transcript_html(session_id=self.session.id, output_path=out_path)
        self.assertTrue(os.path.exists(result_path))

        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("<!DOCTYPE html>", content)
        self.assertIn("<title>Transcript: ", content)
        self.assertIn("Can you list the current directory?", content)
        self.assertIn("Thinking Process", content)
        self.assertIn("Checking directory contents...", content)
        self.assertIn("Here are the files: README.md, main.py", content)


class TestBackupAndRestoreWithFacts(unittest.TestCase):
    """Test full backup bundle export and restore including user facts."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "conversations.db")
        self.persona_path = os.path.join(self.tmp_dir, "personality.md")
        with open(self.persona_path, "w", encoding="utf-8") as f:
            f.write("# Identity\nOriginal Finch persona.\n")

        self.archive = SQLiteArchive(db_path=self.db_path)
        self.archive.add_fact("Persistent user fact #1", category="preference")
        self.archive.add_fact("Persistent user fact #2", category="system")

        sess = self.archive.create_session(title="Backup Test Session")
        self.archive.add_message(sess.id, "user", "Message to backup")

    def tearDown(self):
        self.archive.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_export_and_import_bundle_with_facts(self):
        bundle_path = os.path.join(self.tmp_dir, "bundle.zip")
        created_path = export_backup_bundle(
            archive=self.archive,
            personality_path=self.persona_path,
            output_path=bundle_path,
        )
        self.assertTrue(os.path.exists(created_path))

        # Wipe facts from active archive
        self.archive.wipe_all_facts()
        self.assertEqual(len(self.archive.list_facts()), 0)

        # Import bundle
        import_meta = import_backup_bundle(
            archive=self.archive,
            personality_path=self.persona_path,
            backup_path=bundle_path,
        )
        self.assertEqual(import_meta["facts_restored"], 2)

        restored_facts = self.archive.list_facts()
        self.assertEqual(len(restored_facts), 2)
        fact_texts = [f["fact"] for f in restored_facts]
        self.assertIn("Persistent user fact #1", fact_texts)
        self.assertIn("Persistent user fact #2", fact_texts)


if __name__ == "__main__":
    unittest.main()
