"""Unit tests for CLI provider commands and model listing."""
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from finch.config import AppConfig
from finch.core.assistant import AIAssistant
from finch.providers.base import ModelInfo
from finch.ui.cli import InteractiveCLI


class TestCLIProviderCommands(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.config = AppConfig(
            provider="ollama",
            gemini_api_key="dummy-gemini-key",
            gemini_default_model="gemini-2.5-flash",
        )
        self.assistant = AIAssistant(config=self.config)
        self.cli = InteractiveCLI(assistant=self.assistant)
        self.cli.console.print = MagicMock()

    async def test_handle_provider_no_args_displays_info(self):
        await self.cli.handle_provider_command("")
        self.cli.console.print.assert_called()

    async def test_handle_provider_switch_to_gemini(self):
        with patch.object(self.assistant, "verify_provider_health", AsyncMock(return_value=True)):
            await self.cli.handle_provider_command("gemini")
            self.assertEqual(self.assistant.provider.name, "Gemini")
            self.assertEqual(self.assistant.active_model, "gemini-2.5-flash")

    async def test_handle_providers_table(self):
        await self.cli.handle_providers_command()
        self.cli.console.print.assert_called()

    async def test_handle_models_specific_provider(self):
        mock_models = [ModelInfo(name="gemini-2.5-flash", family="Gemini")]
        with patch.object(self.assistant, "list_available_models", AsyncMock(return_value=mock_models)):
            await self.cli.handle_models_command("gemini")
            self.cli.console.print.assert_called()

    async def test_handle_models_all_providers(self):
        mock_data = {
            "Ollama": {"models": [ModelInfo(name="llama3.2:latest")], "error": None, "active": True},
            "Gemini": {"models": [ModelInfo(name="gemini-2.5-flash")], "error": None, "active": False},
        }
        with patch.object(self.assistant, "list_models_all_providers", AsyncMock(return_value=mock_data)):
            await self.cli.handle_models_command("all")
            self.cli.console.print.assert_called()

    async def test_handle_input_provider_slash_command(self):
        with patch.object(self.assistant, "verify_provider_health", AsyncMock(return_value=True)):
            handled = await self.cli.handle_input("/provider gemini")
            self.assertTrue(handled)
            self.assertEqual(self.assistant.provider.name, "Gemini")

    async def test_handle_input_use_gemini_model_autoswitches_provider(self):
        self.assertEqual(self.assistant.provider.name, "Ollama")
        handled = await self.cli.handle_input("/use gemini-2.5-flash")
        self.assertTrue(handled)
        self.assertEqual(self.assistant.provider.name, "Gemini")
        self.assertEqual(self.assistant.active_model, "gemini-2.5-flash")

    async def test_handle_provider_switch_to_openai(self):
        with patch.object(self.assistant, "verify_provider_health", AsyncMock(return_value=True)):
            await self.cli.handle_provider_command("openai sk-test-key")
            self.assertEqual(self.assistant.provider.name, "OpenAI-Compatible")
            self.assertEqual(self.assistant.active_model, "gpt-4o-mini")

    async def test_handle_input_use_openai_model_autoswitches_provider(self):
        self.assertEqual(self.assistant.provider.name, "Ollama")
        handled = await self.cli.handle_input("/use gpt-4o")
        self.assertTrue(handled)
        self.assertEqual(self.assistant.provider.name, "OpenAI-Compatible")
        self.assertEqual(self.assistant.active_model, "gpt-4o")

    async def test_handle_transcript_command_current_session(self):
        self.assistant.archive.add_message(
            session_id=self.assistant.current_session.id,
            role="user",
            content="Hello from test",
        )
        self.cli.handle_transcript_command("")
        self.cli.console.print.assert_called()

    async def test_handle_transcript_command_by_id(self):
        sess = self.assistant.archive.create_session(title="Historical Session")
        self.assistant.archive.add_message(
            session_id=sess.id,
            role="user",
            content="Historical question",
        )
        self.assistant.archive.add_message(
            session_id=sess.id,
            role="assistant",
            content="Historical answer",
        )
        self.cli.handle_transcript_command(sess.id)
        self.cli.console.print.assert_called()

    async def test_handle_transcript_command_by_number(self):
        sess = self.assistant.archive.create_session(title="Numbered Session")
        self.assistant.archive.add_message(
            session_id=sess.id,
            role="user",
            content="Question for numbered selection",
        )
        self.cli._last_search_sessions = [sess.id]

        # Test both "1" and "#1"
        self.cli.handle_transcript_command("1")
        self.cli.console.print.assert_called()

        self.cli.handle_transcript_command("#1")
        self.cli.console.print.assert_called()

    async def test_handle_transcript_command_number_out_of_range(self):
        self.cli._last_search_sessions = ["sess_sample_1", "sess_sample_2"]
        self.cli.handle_transcript_command("5")
        args, _ = self.cli.console.print.call_args
        self.assertIn("out of range", args[0])

    async def test_handle_transcript_command_number_no_prior_results(self):
        self.cli._last_search_sessions = []
        self.cli.handle_transcript_command("1")
        args, _ = self.cli.console.print.call_args
        self.assertIn("No recent search or session results found", args[0])

    async def test_handle_transcript_command_prefix_resolution(self):
        sess = self.assistant.archive.create_session(title="Prefix Resolution Session")
        self.assistant.archive.add_message(
            session_id=sess.id,
            role="user",
            content="Prefix match testing",
        )
        # Call with partial ID (without full hex)
        partial_id = sess.id[:10]
        self.cli.handle_transcript_command(partial_id)
        self.cli.console.print.assert_called()

        # Call with truncated ellipsis (e.g. sess_xxxx…)
        truncated_ellipsis = sess.id[:12] + "…"
        self.cli.handle_transcript_command(truncated_ellipsis)
        self.cli.console.print.assert_called()

    async def test_handle_sessions_command_populates_last_search_sessions(self):
        sess = self.assistant.archive.create_session(title="Listed Session")
        self.cli.handle_sessions_command("")
        self.assertIn(sess.id, self.cli._last_search_sessions)

    async def test_handle_transcript_command_not_found(self):
        self.cli.handle_transcript_command("nonexistent_session_id")
        args, _ = self.cli.console.print.call_args
        self.assertIn("not found", args[0])

    async def test_handle_transcript_command_edge_case_zero_and_negative(self):
        self.cli._last_search_sessions = ["sess_a", "sess_b"]
        # Zero
        self.cli.handle_transcript_command("0")
        args, _ = self.cli.console.print.call_args
        self.assertIn("out of range", args[0])

        # Negative number
        self.cli.handle_transcript_command("-1")
        args, _ = self.cli.console.print.call_args
        self.assertIn("out of range", args[0])

    async def test_handle_transcript_command_edge_case_formatting(self):
        sess = self.assistant.archive.create_session(title="Formatted Selection Session")
        self.assistant.archive.add_message(
            session_id=sess.id,
            role="user",
            content="Testing brackets and dot formatting",
        )
        self.cli._last_search_sessions = [sess.id]

        # Bracket format: "[1]"
        self.cli.handle_transcript_command("[1]")
        self.cli.console.print.assert_called()

        # Dot format: "1."
        self.cli.handle_transcript_command("1.")
        self.cli.console.print.assert_called()

    async def test_handle_transcript_command_ambiguous_prefix(self):
        sess1 = self.assistant.archive.create_session(session_id="sess_ambig_alpha1", title="Ambig 1")
        sess2 = self.assistant.archive.create_session(session_id="sess_ambig_alpha2", title="Ambig 2")
        self.assistant.archive.add_message(session_id=sess1.id, role="user", content="msg 1")
        self.assistant.archive.add_message(session_id=sess2.id, role="user", content="msg 2")

        # Query with ambiguous prefix
        self.cli.handle_transcript_command("sess_ambig_alpha")
        args, _ = self.cli.console.print.call_args
        self.assertIn("matches multiple sessions", args[0])
        self.assertIn("sess_ambig_alpha1", args[0])
        self.assertIn("sess_ambig_alpha2", args[0])

    async def test_handle_export_command_with_numeric_index(self):
        sess = self.assistant.archive.create_session(title="Export Numbered Session")
        self.assistant.archive.add_message(
            session_id=sess.id,
            role="user",
            content="Content to export",
        )
        self.cli._last_search_sessions = [sess.id]

        import tempfile
        import os
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            self.cli.handle_export_command(f"md 1 {tmp_path}")
            self.assertTrue(os.path.exists(tmp_path))
            with open(tmp_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("Content to export", content)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    async def test_handle_input_transcript_slash_command(self):
        handled = await self.cli.handle_input("/transcript")
        self.assertTrue(handled)
        self.cli.console.print.assert_called()

    async def test_handle_export_hermes_slash_command(self):
        import tempfile
        import shutil
        tmp = tempfile.mkdtemp()
        try:
            handled = await self.cli.handle_input(f"/export hermes {tmp}")
            self.assertTrue(handled)
            self.cli.console.print.assert_called()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
