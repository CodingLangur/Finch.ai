"""Unit tests for provider switching and multi-provider model listing."""
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from ai_assistant.config import AppConfig
from ai_assistant.core.assistant import AIAssistant
from ai_assistant.providers.base import ModelInfo
from ai_assistant.providers.gemini_provider import GeminiProvider
from ai_assistant.providers.ollama_provider import OllamaProvider


class TestProviderSwitching(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.config = AppConfig(
            provider="ollama",
            gemini_api_key="AQ.test_key_12345678",
            gemini_default_model="gemini-2.5-flash",
        )
        self.assistant = AIAssistant(config=self.config)

    def test_initial_provider_is_ollama(self):
        self.assertEqual(self.assistant.provider.name, "Ollama")

    def test_switch_to_gemini(self):
        success, msg = self.assistant.switch_provider("gemini")
        self.assertTrue(success)
        self.assertIn("Google Gemini", msg)
        self.assertEqual(self.assistant.provider.name, "Gemini")
        self.assertEqual(self.assistant.active_model, "gemini-2.5-flash")

    def test_switch_to_gemini_with_explicit_key_and_model(self):
        new_key = "AQ.new_api_key_888"
        success, msg = self.assistant.switch_provider("gemini", api_key=new_key, model="gemini-2.5-pro")
        self.assertTrue(success)
        self.assertEqual(self.assistant.provider.name, "Gemini")
        self.assertEqual(self.assistant.provider.api_key, new_key)
        self.assertEqual(self.assistant.active_model, "gemini-2.5-pro")

    def test_switch_to_gemini_missing_key_fails(self):
        self.assistant.config.gemini_api_key = ""
        success, msg = self.assistant.switch_provider("gemini")
        self.assertFalse(success)
        self.assertIn("API key is required", msg)

    def test_switch_back_to_ollama(self):
        self.assistant.switch_provider("gemini")
        self.assertEqual(self.assistant.provider.name, "Gemini")

        success, msg = self.assistant.switch_provider("ollama")
        self.assertTrue(success)
        self.assertEqual(self.assistant.provider.name, "Ollama")
        self.assertNotIn("gemini", self.assistant.active_model.lower())

    async def test_list_available_models_by_provider(self):
        mock_gemini_models = [
            ModelInfo(name="gemini-2.5-flash", family="Gemini"),
            ModelInfo(name="gemini-2.5-pro", family="Gemini"),
        ]
        with patch.object(GeminiProvider, "list_models", AsyncMock(return_value=mock_gemini_models)):
            models = await self.assistant.list_available_models(provider_name="gemini")
            self.assertEqual(len(models), 2)
            self.assertEqual(models[0].name, "gemini-2.5-flash")

    async def test_list_models_all_providers(self):
        mock_gemini_models = [ModelInfo(name="gemini-2.5-flash", family="Gemini")]
        mock_ollama_models = [ModelInfo(name="llama3.2:latest", family="llama")]

        with patch.object(GeminiProvider, "list_models", AsyncMock(return_value=mock_gemini_models)), \
             patch.object(OllamaProvider, "list_models", AsyncMock(return_value=mock_ollama_models)):
            all_models = await self.assistant.list_models_all_providers()
            self.assertIn("Gemini", all_models)
            self.assertIn("Ollama", all_models)
            self.assertEqual(len(all_models["Gemini"]["models"]), 1)
            self.assertEqual(len(all_models["Ollama"]["models"]), 1)


if __name__ == "__main__":
    unittest.main()
