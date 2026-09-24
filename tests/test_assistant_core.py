"""Integration and unit tests for AIAssistant core and Ollama provider."""
import unittest
from ai_assistant.config import AppConfig
from ai_assistant.core.assistant import AIAssistant, AssistantMode
from ai_assistant.providers.ollama_provider import OllamaProvider


class TestAssistantCore(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.config = AppConfig(
            provider="ollama",
            default_model="Gemma4-26000-ctx:latest",
            window_size=6,
            db_path=":memory:",
        )
        self.assistant = AIAssistant(config=self.config)

    async def asyncTearDown(self):
        self.assistant.close()

    def test_mode_switching(self):
        """Test toggling and setting modes."""
        self.assertEqual(self.assistant.mode, AssistantMode.CHATBOT)
        self.assistant.toggle_mode()
        self.assertEqual(self.assistant.mode, AssistantMode.AGENT)
        self.assistant.set_mode("chatbot")
        self.assertEqual(self.assistant.mode, AssistantMode.CHATBOT)

    async def test_ollama_health_and_models(self):
        """Test actual Ollama health check and listing models."""
        healthy = await self.assistant.verify_provider_health()
        self.assertTrue(healthy, "Ollama service should be reachable.")

        models = await self.assistant.list_available_models()
        self.assertGreater(len(models), 0, "Should have at least one Ollama model.")

        model_names = [m.name for m in models]
        self.assertIn(
            "Gemma4-26000-ctx:latest",
            model_names,
            f"Expected Gemma4-26000-ctx:latest in models, found: {model_names}",
        )


if __name__ == "__main__":
    unittest.main()
