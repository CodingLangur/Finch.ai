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


if __name__ == "__main__":
    unittest.main()
