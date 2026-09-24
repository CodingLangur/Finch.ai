"""Unit tests for GeminiProvider."""
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from ai_assistant.providers.gemini_provider import GeminiProvider


class TestGeminiProvider(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.provider = GeminiProvider(api_key="test-api-key")

    def test_provider_name(self):
        self.assertEqual(self.provider.name, "Gemini")

    def test_convert_messages(self):
        """Test conversion of standard chat turns to Gemini system instruction and contents."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        sys_inst, contents = self.provider._convert_messages(messages)

        # Verify system prompt extraction
        self.assertIsNotNone(sys_inst)
        self.assertEqual(sys_inst["parts"][0]["text"], "You are a helpful assistant.")

        # Verify contents role mapping
        self.assertEqual(len(contents), 3)
        self.assertEqual(contents[0]["role"], "user")
        self.assertEqual(contents[0]["parts"][0]["text"], "Hello")
        self.assertEqual(contents[1]["role"], "model")
        self.assertEqual(contents[1]["parts"][0]["text"], "Hi there!")
        self.assertEqual(contents[2]["role"], "user")
        self.assertEqual(contents[2]["parts"][0]["text"], "How are you?")

    def test_convert_messages_merge_adjacent_roles(self):
        """Test that adjacent messages of the same role are merged to satisfy Gemini API constraints."""
        messages = [
            {"role": "user", "content": "Part 1"},
            {"role": "user", "content": "Part 2"},
        ]
        _, contents = self.provider._convert_messages(messages)
        self.assertEqual(len(contents), 1)
        self.assertEqual(contents[0]["role"], "user")
        self.assertEqual(len(contents[0]["parts"]), 2)
        self.assertEqual(contents[0]["parts"][0]["text"], "Part 1")
        self.assertEqual(contents[0]["parts"][1]["text"], "Part 2")

    async def test_health_check_without_api_key(self):
        """Health check should return False if no API key is set."""
        p = GeminiProvider(api_key="")
        healthy = await p.health_check()
        self.assertFalse(healthy)

    async def test_list_models_parsing(self):
        """Test parsing of models list from Gemini REST response."""
        mock_response_data = {
            "models": [
                {
                    "name": "models/gemini-2.5-flash",
                    "description": "Fast and versatile multimodal model",
                    "supportedGenerationMethods": ["generateContent", "countTokens"],
                },
                {
                    "name": "models/embedding-001",
                    "description": "Embedding model",
                    "supportedGenerationMethods": ["embedContent"],
                },
                {
                    "name": "models/gemini-2.5-pro",
                    "description": "High reasoning model",
                    "supportedGenerationMethods": ["generateContent"],
                },
            ]
        }

        with patch("httpx.AsyncClient.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = mock_response_data
            mock_get.return_value = mock_resp

            models = await self.provider.list_models()
            self.assertEqual(len(models), 2)
            names = [m.name for m in models]
            self.assertIn("gemini-2.5-flash", names)
            self.assertIn("gemini-2.5-pro", names)
            self.assertNotIn("embedding-001", names)


if __name__ == "__main__":
    unittest.main()
