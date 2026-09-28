import io
import os
import shutil
import tempfile
import unittest
import zipfile
from unittest.mock import AsyncMock, MagicMock

from rich.console import Console

from ai_assistant.config import AppConfig
from ai_assistant.core.assistant import AIAssistant
from ai_assistant.persona.manager import DEFAULT_PERSONA_TEMPLATE, PersonaManager
from ai_assistant.providers.base import BaseLLMProvider, ModelInfo, StreamChunk
from ai_assistant.storage.backup import export_backup_bundle, import_backup_bundle
from ai_assistant.storage.sqlite_archive import SQLiteArchive
from ai_assistant.ui.cli import InteractiveCLI


class MockLLMProvider(BaseLLMProvider):
    def __init__(self, responses=None):
        self.responses = responses or []
        self._resp_index = 0

    @property
    def name(self) -> str:
        return "mock"

    async def list_models(self):
        return [ModelInfo(name="mock-model", size_gb=1.0)]

    async def health_check(self) -> bool:
        return True

    async def stream_chat(self, messages, model, tools=None, options=None):
        resp = (
            self.responses[self._resp_index]
            if self._resp_index < len(self.responses)
            else "Mock default response."
        )
        self._resp_index += 1
        yield StreamChunk(delta=resp, is_done=True)


class TestPersonaWipeAndAdapt(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.personality_file = os.path.join(self.test_dir, "personality.md")
        self.manager = PersonaManager(file_path=self.personality_file)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_wipe_persona(self):
        """Verify wiping persona resets it to default template and backs up current file."""
        # Modify persona
        custom_content = "# Custom Persona\n- Speak like a pirate."
        self.manager.save_persona(custom_content)
        self.assertEqual(self.manager.load_persona(), custom_content)

        # Wipe persona
        reset_content = self.manager.wipe_persona()
        self.assertIn("Assistant Persona & Guidelines", reset_content)
        self.assertEqual(self.manager.load_persona(), DEFAULT_PERSONA_TEMPLATE.strip())

        # Backup file should contain the custom content
        backup_file = self.personality_file + ".bak"
        self.assertTrue(os.path.exists(backup_file))
        with open(backup_file, "r", encoding="utf-8") as f:
            bak_data = f.read().strip()
        self.assertEqual(bak_data, custom_content)

    def test_build_adaptation_prompt(self):
        """Verify adaptation prompt generation with session summaries."""
        current_persona = "# Current Persona"
        summaries = [
            {"title": "Python Refactoring", "summary": "Discussed asyncio best practices."},
            {"title": "Database Optimization", "summary": "User preferred concise raw SQL."},
        ]
        prompt = self.manager.build_adaptation_prompt(current_persona, summaries)
        self.assertIn("Python Refactoring", prompt)
        self.assertIn("Discussed asyncio best practices.", prompt)
        self.assertIn("Database Optimization", prompt)
        self.assertIn("<personality_update>", prompt)


class TestArchiveWipeAndBackup(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "conversations.db")
        self.personality_file = os.path.join(self.test_dir, "personality.md")
        self.archive = SQLiteArchive(db_path=self.db_path)
        self.persona_manager = PersonaManager(file_path=self.personality_file)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_wipe_session_messages(self):
        """Verify wiping messages for one session leaves other sessions intact."""
        s1 = self.archive.create_session(title="Session 1")
        s2 = self.archive.create_session(title="Session 2")

        self.archive.add_message(s1.id, "user", "Message 1 in S1")
        self.archive.add_message(s1.id, "assistant", "Response 1 in S1")
        self.archive.add_message(s2.id, "user", "Message in S2")

        self.assertEqual(len(self.archive.get_messages(s1.id)), 2)
        self.assertEqual(len(self.archive.get_messages(s2.id)), 1)

        # Wipe session 1 messages
        wiped = self.archive.wipe_session_messages(s1.id)
        self.assertEqual(wiped, 2)
        self.assertEqual(len(self.archive.get_messages(s1.id)), 0)
        # Session 2 must remain completely intact
        self.assertEqual(len(self.archive.get_messages(s2.id)), 1)

    def test_wipe_all_conversations(self):
        """Verify wiping all conversations clears sessions and messages."""
        s1 = self.archive.create_session(title="Session 1")
        s2 = self.archive.create_session(title="Session 2")
        self.archive.add_message(s1.id, "user", "Hello")
        self.archive.add_message(s2.id, "user", "World")

        stats = self.archive.wipe_all_conversations()
        self.assertEqual(stats["sessions_wiped"], 2)
        self.assertEqual(stats["messages_wiped"], 2)

        self.assertEqual(len(self.archive.list_sessions()), 0)

    def test_export_and_import_backup_bundle(self):
        """Verify exporting a backup bundle and restoring it in replace mode."""
        # 1. Setup initial state
        s1 = self.archive.create_session(title="Session A")
        self.archive.add_message(s1.id, "user", "First user question")
        self.archive.add_message(s1.id, "assistant", "First assistant answer")

        custom_persona = "# Unique Persona\n- Custom rule 12345."
        self.persona_manager.save_persona(custom_persona)

        # 2. Export backup
        backup_zip = export_backup_bundle(
            archive=self.archive,
            personality_path=self.personality_file,
            backup_dir=self.test_dir,
        )
        self.assertTrue(os.path.exists(backup_zip))
        self.assertTrue(zipfile.is_zipfile(backup_zip))

        # Inspect zip contents
        with zipfile.ZipFile(backup_zip, "r") as zf:
            namelist = zf.namelist()
            self.assertIn("manifest.json", namelist)
            self.assertIn("personality.md", namelist)
            self.assertIn("conversations.json", namelist)

        # 3. Wipe current state
        self.archive.wipe_all_conversations()
        self.persona_manager.wipe_persona()
        self.assertEqual(len(self.archive.list_sessions()), 0)
        self.assertNotIn("Unique Persona", self.persona_manager.load_persona())

        # 4. Import and restore from backup
        res = import_backup_bundle(
            archive=self.archive,
            personality_path=self.personality_file,
            backup_path=backup_zip,
            mode="replace",
        )
        self.assertTrue(res["success"])
        self.assertEqual(res["sessions_restored"], 1)
        self.assertEqual(res["messages_restored"], 2)
        self.assertTrue(res["persona_restored"])

        # Check restored data
        self.assertIn("Unique Persona", self.persona_manager.load_persona())
        sessions = self.archive.list_sessions()
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0].title, "Session A")
        messages = self.archive.get_messages(sessions[0].id)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0].content, "First user question")


class TestAIAssistantWipeExportImport(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "conversations.db")
        self.personality_file = os.path.join(self.test_dir, "personality.md")

        self.config = AppConfig(
            db_path=self.db_path,
            personality_path=self.personality_file,
            backup_dir=os.path.join(self.test_dir, "backups"),
            compression_enabled=False,
        )
        self.provider = MockLLMProvider()
        self.archive = SQLiteArchive(db_path=self.db_path)
        self.assistant = AIAssistant(
            config=self.config,
            provider=self.provider,
            archive=self.archive,
        )

    async def asyncTearDown(self):
        self.assistant.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    async def test_assistant_wipe_conversation_current_session(self):
        """Verify assistant wipes current session messages and clears in-memory buffer."""
        self.assistant.memory.add_user_message("Hello memory")
        self.assistant.archive.add_message(self.assistant.current_session.id, "user", "Hello DB")

        self.assertEqual(len(self.assistant.memory._messages), 1)
        res = self.assistant.wipe_conversation(all_sessions=False)
        self.assertFalse(res["all_sessions"])
        self.assertEqual(len(self.assistant.memory._messages), 0)

    async def test_assistant_wipe_conversation_all(self):
        """Verify assistant wipes all sessions and starts a fresh one."""
        self.assistant.archive.add_message(self.assistant.current_session.id, "user", "Message 1")
        res = self.assistant.wipe_conversation(all_sessions=True)
        self.assertTrue(res["all_sessions"])
        self.assertEqual(len(self.assistant.memory._messages), 0)
        self.assertIsNotNone(self.assistant.current_session)

    async def test_assistant_wipe_persona(self):
        """Verify assistant wipes persona and reloads system prompt in memory."""
        self.assistant.persona_manager.save_persona("# Temporary Persona")
        self.assistant.reload_persona()
        self.assertIn("Temporary Persona", self.assistant.memory.get_system_prompt())

        self.assistant.wipe_persona()
        self.assertNotIn("Temporary Persona", self.assistant.memory.get_system_prompt())
        self.assertIn("Assistant Persona & Guidelines", self.assistant.memory.get_system_prompt())

    async def test_assistant_export_and_import_backup(self):
        """Verify AIAssistant export and import backup end-to-end."""
        # Add messages and custom persona
        self.assistant.persona_manager.save_persona("# Developer Assistant\n- Fast answers.")
        self.assistant.reload_persona()
        self.assistant.archive.add_message(self.assistant.current_session.id, "user", "Export test query")

        # Export backup
        zip_path = self.assistant.export_backup()
        self.assertTrue(os.path.exists(zip_path))

        # Wipe both
        self.assistant.wipe_conversation(all_sessions=True)
        self.assistant.wipe_persona()

        # Import backup
        res = self.assistant.import_backup(zip_path, mode="replace")
        self.assertTrue(res["success"])
        self.assertIn("Developer Assistant", self.assistant.persona_manager.load_persona())
        self.assertIn("Developer Assistant", self.assistant.memory.get_system_prompt())

    async def test_assistant_adapt_persona(self):
        """Verify AIAssistant adapt_persona parses model update tags and updates personality.md."""
        adapted_doc = (
            "# Assistant Persona & Guidelines\n\n"
            "## Core Identity\n- Name: Adaptive Antigravity\n\n"
            "## Learned Preferences & Adaptive Traits\n- User frequently programs in Python and prefers minimal syntax explanations."
        )
        self.assistant.provider = MockLLMProvider(
            responses=[f"Certainly, adapting persona!\n<personality_update>\n{adapted_doc}\n</personality_update>"]
        )
        # Add past user interaction
        self.assistant.archive.add_message(self.assistant.current_session.id, "user", "I like Python without extra comments.")

        success, msg = await self.assistant.adapt_persona()
        self.assertTrue(success)
        self.assertIn("Adaptive Antigravity", self.assistant.persona_manager.load_persona())
        self.assertIn("Python and prefers minimal syntax explanations", self.assistant.persona_manager.load_persona())

    async def test_assistant_wipe_helpers(self):
        """Verify wipe_current_conversation and wipe_all_conversations helper methods."""
        # 1. Wipe current session (contains system message + user message)
        self.assistant.archive.add_message(self.assistant.current_session.id, "user", "Msg in curr")
        self.assistant.memory.add_user_message("Msg in buffer")
        res1 = self.assistant.wipe_current_conversation()
        self.assertFalse(res1["all_sessions"])
        self.assertGreaterEqual(res1["messages_wiped"], 1)
        self.assertEqual(len(self.assistant.memory._messages), 0)

        # 2. Wipe all conversations
        s2 = self.assistant.archive.create_session(title="Second")
        self.assistant.archive.add_message(s2.id, "user", "Msg in s2")
        res2 = self.assistant.wipe_all_conversations()
        self.assertTrue(res2["all_sessions"])
        self.assertGreaterEqual(res2["sessions_wiped"], 2)
        self.assertEqual(len(self.assistant.archive.list_sessions()), 1)  # Only the newly created fresh session


class TestCLIWipeAndClearCommands(unittest.IsolatedAsyncioTestCase):
    """Test CLI /wipe and /clear command processing."""

    async def asyncSetUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "conversations.db")
        self.personality_file = os.path.join(self.test_dir, "personality.md")

        self.config = AppConfig(
            db_path=self.db_path,
            personality_path=self.personality_file,
            compression_enabled=False,
        )
        self.provider = MockLLMProvider()
        self.archive = SQLiteArchive(db_path=self.db_path)
        self.assistant = AIAssistant(
            config=self.config,
            provider=self.provider,
            archive=self.archive,
        )
        self.cli = InteractiveCLI(self.assistant)
        self.string_io = io.StringIO()
        self.cli.console = Console(file=self.string_io, color_system=None)

    async def asyncTearDown(self):
        self.assistant.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_cli_wipe_current_shorthand(self):
        self.assistant.archive.add_message(self.assistant.current_session.id, "user", "Testing wipe current")
        self.cli.handle_wipe_command("current")
        output = self.string_io.getvalue()
        self.assertIn("Successfully wiped conversation messages for active session", output)
        self.assertEqual(len(self.assistant.archive.get_messages(self.assistant.current_session.id)), 0)

    def test_cli_wipe_all_shorthand(self):
        self.assistant.archive.add_message(self.assistant.current_session.id, "user", "Msg 1")
        s2 = self.assistant.archive.create_session(title="S2")
        self.assistant.archive.add_message(s2.id, "user", "Msg 2")

        self.cli.handle_wipe_command("all")
        output = self.string_io.getvalue()
        self.assertIn("Successfully wiped ALL conversation history", output)
        self.assertEqual(len(self.assistant.archive.list_sessions()), 1)

    def test_cli_wipe_conversation_synonyms(self):
        self.assistant.archive.add_message(self.assistant.current_session.id, "user", "Msg")
        self.cli.handle_wipe_command("conversations current")
        self.assertIn("Successfully wiped conversation messages", self.string_io.getvalue())

    async def test_cli_clear_command_variants(self):
        # 1. /clear default (clears in-memory buffer)
        self.assistant.memory.add_user_message("Buffer message")
        self.assertEqual(len(self.assistant.memory._messages), 1)
        await self.cli.handle_input("/clear")
        self.assertEqual(len(self.assistant.memory._messages), 0)
        self.assertIn("Conversation history cleared", self.string_io.getvalue())

        # 2. /clear all (wipes all conversations in SQLite)
        self.string_io.truncate(0)
        self.string_io.seek(0)
        self.assistant.archive.add_message(self.assistant.current_session.id, "user", "Msg to wipe")
        await self.cli.handle_input("/clear all")
        self.assertIn("Successfully wiped ALL conversation history from SQLite", self.string_io.getvalue())


if __name__ == "__main__":
    unittest.main()
