"""OpenAI-Compatible LLM Provider implementation using httpx streaming.

Supports standard OpenAI endpoints (api.openai.com) as well as any OpenAI-compatible
inference server such as vLLM, LM Studio, LocalAI, Ollama (/v1), Groq, Together AI,
OpenRouter, DeepSeek, and Mistral.
"""
import json
import os
import time
from typing import Any, AsyncGenerator, Dict, List, Optional
import httpx

from .base import BaseLLMProvider, ModelInfo, StreamChunk, StreamStats


class OpenAICompatibleProvider(BaseLLMProvider):
    """Client for any OpenAI-compatible Chat Completions API with streaming telemetry."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 120.0,
        default_model: Optional[str] = None,
    ):
        self.api_key = (
            api_key
            if api_key is not None
            else os.getenv("OPENAI_API_KEY", "")
        )
        # Normalize base URL (strip trailing slash)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.default_model = default_model or os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini")

    @property
    def name(self) -> str:
        return "OpenAI-Compatible"

    def _get_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def health_check(self) -> bool:
        """Confirm OpenAI-compatible API connectivity."""
        url = f"{self.base_url}/models"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.get(url, headers=self._get_headers())
                return res.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> List[ModelInfo]:
        """Fetch available models from the /models endpoint."""
        url = f"{self.base_url}/models"
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                res = await client.get(url, headers=self._get_headers())
                res.raise_for_status()
                data = res.json()
                models: List[ModelInfo] = []
                for item in data.get("data", []):
                    model_id = item.get("id", "")
                    if model_id:
                        models.append(
                            ModelInfo(
                                name=model_id,
                                family=item.get("owned_by", "OpenAI-Compatible"),
                                modified_at=str(item.get("created", "")),
                            )
                        )
                # Sort alphabetically by name
                models.sort(key=lambda m: m.name)
                return models
        except Exception as e:
            raise RuntimeError(
                f"Failed to query models from OpenAI-compatible API at {self.base_url}: {e}"
            ) from e

    def _sanitize_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ensure messages adhere to strict OpenAI chat completion specs."""
        sanitized: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Translate model -> assistant
            if role == "model":
                role = "assistant"

            item: Dict[str, Any] = {
                "role": role,
                "content": content if content is not None else "",
            }

            if role == "assistant" and msg.get("tool_calls"):
                item["tool_calls"] = msg["tool_calls"]

            if role == "tool":
                item["tool_call_id"] = msg.get("tool_call_id", "call_default")
                if "name" in msg:
                    item["name"] = msg["name"]

            sanitized.append(item)
        return sanitized

    async def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        options: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream chat completions from OpenAI-compatible endpoint with telemetry and tool support."""
        url = f"{self.base_url}/chat/completions"
        clean_messages = self._sanitize_messages(messages)

        payload: Dict[str, Any] = {
            "model": model,
            "messages": clean_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        if options:
            for k, v in options.items():
                if k not in payload:
                    payload[k] = v

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        start_time = time.perf_counter()
        first_token_time: Optional[float] = None
        total_eval_tokens = 0
        total_prompt_tokens = 0

        # Accumulated tool calls buffer across stream chunks
        # Key: tool call index, Value: {"id": str, "name": str, "arguments": str}
        accumulated_tools: Dict[int, Dict[str, Any]] = {}

        headers = self._get_headers()

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                # Attempt with stream_options first
                response = await client.post(url, headers=headers, json=payload)
                if response.status_code == 400 and "stream_options" in response.text:
                    # Fallback for servers that reject stream_options
                    payload.pop("stream_options", None)
                    response = await client.post(url, headers=headers, json=payload)

                if response.status_code != 200:
                    error_bytes = response.content
                    error_msg = error_bytes.decode("utf-8", errors="ignore")
                    raise RuntimeError(
                        f"OpenAI-compatible API at {self.base_url} returned HTTP {response.status_code}: {error_msg}"
                    )

                # Stream response lines
                async with client.stream("POST", url, headers=headers, json=payload) as stream_resp:
                    if stream_resp.status_code != 200:
                        err = await stream_resp.aread()
                        raise RuntimeError(
                            f"OpenAI-compatible API returned HTTP {stream_resp.status_code}: {err.decode('utf-8', errors='ignore')}"
                        )

                    async for line in stream_resp.aiter_lines():
                        if not line:
                            continue

                        line_str = line.strip()
                        if not line_str.startswith("data:"):
                            continue

                        data_content = line_str[len("data:"):].strip()
                        if data_content == "[DONE]":
                            break

                        try:
                            chunk_data = json.loads(data_content)
                        except json.JSONDecodeError:
                            continue

                        # Extract usage if returned in stream
                        usage = chunk_data.get("usage")
                        if usage:
                            total_prompt_tokens = usage.get("prompt_tokens", total_prompt_tokens)
                            total_eval_tokens = usage.get("completion_tokens", total_eval_tokens)

                        choices = chunk_data.get("choices", [])
                        if not choices:
                            continue

                        choice = choices[0]
                        delta_dict = choice.get("delta", {})
                        delta_text = delta_dict.get("content") or ""
                        # Capture reasoning/thinking (DeepSeek-R1, vLLM, Qwen)
                        thinking_text = (
                            delta_dict.get("reasoning_content")
                            or delta_dict.get("thinking")
                            or ""
                        )
                        raw_tool_deltas = delta_dict.get("tool_calls") or []

                        if (delta_text or thinking_text or raw_tool_deltas) and first_token_time is None:
                            first_token_time = time.perf_counter()

                        if delta_text:
                            total_eval_tokens += 1

                        # Process streaming tool call fragments
                        for tc in raw_tool_deltas:
                            idx = tc.get("index", 0)
                            if idx not in accumulated_tools:
                                accumulated_tools[idx] = {
                                    "id": tc.get("id", f"call_{idx}"),
                                    "type": "function",
                                    "name": tc.get("function", {}).get("name", ""),
                                    "arguments": "",
                                }
                            else:
                                if tc.get("id"):
                                    accumulated_tools[idx]["id"] = tc["id"]
                                if tc.get("function", {}).get("name"):
                                    accumulated_tools[idx]["name"] += tc["function"]["name"]

                            arg_chunk = tc.get("function", {}).get("arguments", "")
                            if arg_chunk:
                                accumulated_tools[idx]["arguments"] += arg_chunk

                        finish_reason = choice.get("finish_reason")
                        if finish_reason:
                            # Stream has concluded for this choice
                            pass

                        # Yield chunk if there's text or thinking to display
                        if delta_text or thinking_text:
                            yield StreamChunk(
                                delta=delta_text,
                                thinking_delta=thinking_text,
                                is_done=False,
                            )

            except httpx.ConnectError as e:
                raise ConnectionError(
                    f"Could not connect to OpenAI-compatible server at {self.base_url}. Verify server is running and accessible."
                ) from e

        # Finalize and compute telemetry stats
        end_time = time.perf_counter()
        total_duration_ms = (end_time - start_time) * 1000.0
        ttft_ms = (
            (first_token_time - start_time) * 1000.0
            if first_token_time
            else total_duration_ms
        )

        final_tool_calls: Optional[List[Dict[str, Any]]] = None
        if accumulated_tools:
            final_tool_calls = []
            for idx in sorted(accumulated_tools.keys()):
                item = accumulated_tools[idx]
                args_str = item["arguments"]
                try:
                    parsed_args = json.loads(args_str) if args_str else {}
                except json.JSONDecodeError:
                    parsed_args = {"input": args_str}

                final_tool_calls.append({
                    "id": item["id"],
                    "type": "function",
                    "function": {
                        "name": item["name"],
                        "arguments": parsed_args,
                    },
                })

        stats = StreamStats(
            ttft_ms=round(ttft_ms, 2),
            total_duration_ms=round(total_duration_ms, 2),
            eval_count=total_eval_tokens,
            eval_duration_ms=round(total_duration_ms - ttft_ms, 2),
            prompt_eval_count=total_prompt_tokens,
        )
        stats.calculate_speeds()

        # Emit completion chunk
        yield StreamChunk(
            delta="",
            thinking_delta="",
            is_done=True,
            stats=stats,
            tool_calls=final_tool_calls,
        )
