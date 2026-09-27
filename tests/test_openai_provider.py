"""Unit tests for OpenAICompatibleProvider."""
import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from ai_assistant.config import AppConfig
from ai_assistant.core.assistant import AIAssistant
from ai_assistant.providers.base import ModelInfo, StreamChunk
from ai_assistant.providers.openai_provider import OpenAICompatibleProvider


class TestOpenAICompatibleProvider(unittest.IsolatedAsyncioTestCase):
    """Test suite for OpenAI-compatible endpoint provider."""

    def setUp(self):
        self.provider = OpenAICompatibleProvider(
            api_key="test-sk-123",
            base_url="https://api.openai.com/v1",
            timeout=30.0,
            default_model="gpt-4o-mini",
        )

    def test_provider_name_and_headers(self):
        self.assertEqual(self.provider.name, "OpenAI-Compatible")
        headers = self.provider._get_headers()
        self.assertEqual(headers["Authorization"], "Bearer test-sk-123")
        self.assertEqual(headers["Content-Type"], "application/json")

    def test_headers_without_api_key(self):
        provider_no_key = OpenAICompatibleProvider(
            api_key="",
            base_url="http://localhost:8000/v1",
        )
        headers = provider_no_key._get_headers()
        self.assertNotIn("Authorization", headers)
        self.assertEqual(headers["Content-Type"], "application/json")

    def test_sanitize_messages(self):
        raw_messages = [
            {"role": "system", "content": "You are Finch.ai"},
            {"role": "user", "content": "Hello", "extra_metadata": 123},
            {"role": "model", "content": "Hi there"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c1", "function": {"name": "test"}}],
            },
            {"role": "tool", "content": "result", "tool_call_id": "c1"},
        ]
        sanitized = self.provider._sanitize_messages(raw_messages)
        self.assertEqual(len(sanitized), 5)
        self.assertEqual(sanitized[0]["role"], "system")
        self.assertEqual(sanitized[1]["role"], "user")
        self.assertNotIn("extra_metadata", sanitized[1])
        # Model should be converted to assistant
        self.assertEqual(sanitized[2]["role"], "assistant")
        self.assertEqual(sanitized[3]["tool_calls"][0]["id"], "c1")
        self.assertEqual(sanitized[4]["tool_call_id"], "c1")

    @patch("httpx.AsyncClient.get")
    async def test_health_check_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        healthy = await self.provider.health_check()
        self.assertTrue(healthy)

    @patch("httpx.AsyncClient.get")
    async def test_health_check_failure(self, mock_get):
        mock_get.side_effect = Exception("Connection refused")
        healthy = await self.provider.health_check()
        self.assertFalse(healthy)

    @patch("httpx.AsyncClient.get")
    async def test_list_models(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"id": "gpt-4o", "owned_by": "openai", "created": 1700000000},
                {"id": "gpt-4o-mini", "owned_by": "openai", "created": 1700000001},
            ]
        }
        mock_get.return_value = mock_response

        models = await self.provider.list_models()
        self.assertEqual(len(models), 2)
        self.assertEqual(models[0].name, "gpt-4o")
        self.assertEqual(models[1].name, "gpt-4o-mini")

    async def test_stream_chat_text_and_thinking(self):
        sse_lines = [
            'data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}',
            'data: {"choices":[{"delta":{"thinking":"Let me reason about this..."},"finish_reason":null}]}',
            'data: {"choices":[{"delta":{"content":"Hello! How can "},"finish_reason":null}]}',
            'data: {"choices":[{"delta":{"content":"I assist you?"},"finish_reason":null}]}',
            'data: {"usage":{"prompt_tokens":10,"completion_tokens":8,"total_tokens":18}}',
            'data: [DONE]',
        ]

        class MockStreamResponse:
            def __init__(self, lines):
                self.status_code = 200
                self.lines = lines

            async def aiter_lines(self):
                for line in self.lines:
                    yield line

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass

        mock_post_resp = MagicMock()
        mock_post_resp.status_code = 200

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_post_resp
            with patch("httpx.AsyncClient.stream", return_value=MockStreamResponse(sse_lines)):
                chunks = []
                async for chunk in self.provider.stream_chat(
                    messages=[{"role": "user", "content": "Hi"}],
                    model="gpt-4o-mini",
                ):
                    chunks.append(chunk)

                # Check received chunks
                self.assertGreaterEqual(len(chunks), 3)
                thinking_chunks = [c for c in chunks if c.thinking_delta]
                text_chunks = [c for c in chunks if c.delta]
                done_chunks = [c for c in chunks if c.is_done]

                self.assertTrue(any("reason about this" in c.thinking_delta for c in thinking_chunks))
                full_text = "".join(c.delta for c in text_chunks)
                self.assertEqual(full_text, "Hello! How can I assist you?")
                self.assertEqual(len(done_chunks), 1)
                self.assertIsNotNone(done_chunks[0].stats)
                self.assertEqual(done_chunks[0].stats.prompt_eval_count, 10)

    async def test_stream_chat_streaming_tool_calls(self):
        sse_lines = [
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_999","type":"function","function":{"name":"search_past_conversations","arguments":""}}]},"finish_reason":null}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"query\\": \\""}}]},"finish_reason":null}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"python\\"}"}}]},"finish_reason":"tool_calls"}]}',
            'data: [DONE]',
        ]

        class MockStreamResponse:
            def __init__(self, lines):
                self.status_code = 200
                self.lines = lines

            async def aiter_lines(self):
                for line in self.lines:
                    yield line

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass

        mock_post_resp = MagicMock()
        mock_post_resp.status_code = 200

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_post_resp
            with patch("httpx.AsyncClient.stream", return_value=MockStreamResponse(sse_lines)):
                chunks = []
                async for chunk in self.provider.stream_chat(
                    messages=[{"role": "user", "content": "What did we talk about?"}],
                    model="gpt-4o-mini",
                    tools=[{"type": "function", "function": {"name": "search_past_conversations"}}],
                ):
                    chunks.append(chunk)

                done_chunks = [c for c in chunks if c.is_done]
                self.assertEqual(len(done_chunks), 1)
                self.assertIsNotNone(done_chunks[0].tool_calls)
                tc = done_chunks[0].tool_calls[0]
                self.assertEqual(tc["id"], "call_999")
                self.assertEqual(tc["function"]["name"], "search_past_conversations")
                self.assertEqual(tc["function"]["arguments"], {"query": "python"})

    def test_assistant_initialization_with_openai_provider(self):
        config = AppConfig(
            provider="openai",
            openai_api_key="sk-test-key",
            openai_base_url="https://api.openai.com/v1",
            openai_default_model="gpt-4o-mini",
        )
        assistant = AIAssistant(config=config)
        self.assertIsInstance(assistant.provider, OpenAICompatibleProvider)
        self.assertEqual(assistant.provider.name, "OpenAI-Compatible")
        self.assertEqual(assistant.active_model, "gpt-4o-mini")


if __name__ == "__main__":
    unittest.main()
