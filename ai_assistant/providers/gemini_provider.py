"""Google Gemini LLM Provider implementation using httpx streaming."""
import json
import os
import time
from typing import Any, AsyncGenerator, Dict, List, Optional
import httpx

from .base import BaseLLMProvider, ModelInfo, StreamChunk, StreamStats


class GeminiProvider(BaseLLMProvider):
    """Google Gemini client using REST SSE streaming with real-time telemetry."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://generativelanguage.googleapis.com",
        timeout: float = 120.0,
    ):
        self.api_key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @property
    def name(self) -> str:
        return "Gemini"

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
        }

    async def health_check(self) -> bool:
        """Confirm Gemini API connectivity with current API key."""
        if not self.api_key:
            return False
        url = f"{self.base_url}/v1beta/models?key={self.api_key}"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.get(url, headers=self._get_headers())
                return res.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> List[ModelInfo]:
        """Fetch available generation models from Gemini API."""
        if not self.api_key:
            raise RuntimeError("Gemini API key is not configured in .env or GEMINI_API_KEY environment variable.")

        url = f"{self.base_url}/v1beta/models?key={self.api_key}"
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                res = await client.get(url, headers=self._get_headers())
                res.raise_for_status()
                data = res.json()
                models: List[ModelInfo] = []
                for item in data.get("models", []):
                    methods = item.get("supportedGenerationMethods", [])
                    if "generateContent" in methods:
                        raw_name = item.get("name", "")
                        clean_name = raw_name.replace("models/", "")
                        models.append(
                            ModelInfo(
                                name=clean_name,
                                family="Gemini",
                                parameter_size=item.get("description", "")[:40],
                            )
                        )
                return models
        except Exception as e:
            raise RuntimeError(f"Failed to query models from Gemini API: {e}") from e

    def _convert_tools(
        self, tools: Optional[List[Dict[str, Any]]]
    ) -> Optional[List[Dict[str, Any]]]:
        """Convert standard OpenAI/Ollama tool schemas into Gemini functionDeclarations."""
        if not tools:
            return None

        declarations: List[Dict[str, Any]] = []
        for t in tools:
            fn = t.get("function", t)
            dec: Dict[str, Any] = {
                "name": fn.get("name"),
                "description": fn.get("description", ""),
            }
            if "parameters" in fn:
                dec["parameters"] = fn["parameters"]
            declarations.append(dec)

        return [{"functionDeclarations": declarations}]

    def _convert_messages(
        self, messages: List[Dict[str, Any]]
    ) -> tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """Convert standard messages into Gemini system instruction and alternating turns."""
        system_instruction: Optional[Dict[str, Any]] = None
        contents: List[Dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                # System prompt is pinned to system_instruction
                system_instruction = {"parts": [{"text": content}]}
                continue

            if role in ("assistant", "model"):
                parts: List[Dict[str, Any]] = []
                if content:
                    parts.append({"text": content})
                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        fn = tc.get("function", {})
                        fn_args = fn.get("arguments", {})
                        if isinstance(fn_args, str):
                            try:
                                fn_args = json.loads(fn_args)
                            except Exception:
                                fn_args = {"input": fn_args}
                        parts.append({
                            "functionCall": {
                                "name": fn.get("name"),
                                "args": fn_args,
                            }
                        })
                if not parts:
                    parts.append({"text": ""})

                if contents and contents[-1]["role"] == "model":
                    contents[-1]["parts"].extend(parts)
                else:
                    contents.append({"role": "model", "parts": parts})
                continue

            if role == "tool":
                func_name = msg.get("name") or "tool"
                part = {
                    "functionResponse": {
                        "name": func_name,
                        "response": {"result": content},
                    }
                }
                if contents and contents[-1]["role"] == "function":
                    contents[-1]["parts"].append(part)
                else:
                    contents.append({"role": "function", "parts": [part]})
                continue

            # Standard user message
            gemini_role = "user"
            part = {"text": content}
            if contents and contents[-1]["role"] == gemini_role:
                contents[-1]["parts"].append(part)
            else:
                contents.append({"role": gemini_role, "parts": [part]})

        return system_instruction, contents

    async def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        options: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream chat completions from Gemini with real-time telemetry and tool call support."""
        if not self.api_key:
            raise RuntimeError(
                "Gemini API key is required. Please set GEMINI_API_KEY in your .env file."
            )

        clean_model = model.replace("models/", "")
        url = f"{self.base_url}/v1beta/models/{clean_model}:streamGenerateContent?alt=sse&key={self.api_key}"

        system_instruction, contents = self._convert_messages(messages)
        payload: Dict[str, Any] = {
            "contents": contents,
        }
        if system_instruction:
            payload["system_instruction"] = system_instruction

        if tools:
            gemini_tools = self._convert_tools(tools)
            if gemini_tools:
                payload["tools"] = gemini_tools

        if options:
            gen_config: Dict[str, Any] = {}
            if "temperature" in options:
                gen_config["temperature"] = options["temperature"]
            if gen_config:
                payload["generationConfig"] = gen_config

        start_time = time.perf_counter()
        first_token_time: Optional[float] = None
        total_eval_tokens = 0
        prompt_tokens = 0

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                async with client.stream(
                    "POST", url, headers=self._get_headers(), json=payload
                ) as response:
                    if response.status_code != 200:
                        error_body = await response.aread()
                        raise RuntimeError(
                            f"Gemini API returned HTTP {response.status_code}: {error_body.decode('utf-8', errors='ignore')}"
                        )

                    async for line in response.aiter_lines():
                        if not line:
                            continue

                        # SSE lines start with "data: "
                        if line.startswith("data: "):
                            raw_json = line[6:].strip()
                            if not raw_json or raw_json == "[DONE]":
                                continue

                            try:
                                chunk_data = json.loads(raw_json)
                            except json.JSONDecodeError:
                                continue

                            usage = chunk_data.get("usageMetadata", {})
                            if usage:
                                total_eval_tokens = usage.get("candidatesTokenCount", total_eval_tokens)
                                prompt_tokens = usage.get("promptTokenCount", prompt_tokens)

                            candidates = chunk_data.get("candidates", [])
                            if not candidates:
                                continue

                            candidate = candidates[0]
                            content_parts = candidate.get("content", {}).get("parts", [])
                            finish_reason = candidate.get("finishReason")
                            is_done = finish_reason is not None and finish_reason != ""

                            for part in content_parts:
                                text_delta = part.get("text", "")
                                thinking_delta = part.get("thought", "")
                                fc = part.get("functionCall")
                                chunk_tool_calls: Optional[List[Dict[str, Any]]] = None

                                if fc:
                                    chunk_tool_calls = [{
                                        "id": f"call_{int(time.time() * 1000)}",
                                        "function": {
                                            "name": fc.get("name"),
                                            "arguments": fc.get("args", {}),
                                        },
                                    }]

                                if (text_delta or thinking_delta or chunk_tool_calls) and first_token_time is None:
                                    first_token_time = time.perf_counter()

                                yield StreamChunk(
                                    delta=text_delta,
                                    thinking_delta=thinking_delta,
                                    tool_calls=chunk_tool_calls,
                                    is_done=False,
                                )

                    # Finished stream, compute final metrics
                    end_time = time.perf_counter()
                    total_duration_ms = (end_time - start_time) * 1000.0
                    ttft_ms = (
                        (first_token_time - start_time) * 1000.0
                        if first_token_time
                        else total_duration_ms
                    )

                    stats = StreamStats(
                        ttft_ms=round(ttft_ms, 2),
                        total_duration_ms=round(total_duration_ms, 2),
                        eval_count=total_eval_tokens,
                        eval_duration_ms=round(total_duration_ms - ttft_ms, 2),
                        prompt_eval_count=prompt_tokens,
                    )
                    stats.calculate_speeds()

                    yield StreamChunk(
                        delta="",
                        is_done=True,
                        stats=stats,
                    )

            except httpx.ConnectError as e:
                raise ConnectionError(
                    f"Could not connect to Gemini API at {self.base_url}. Check network connection."
                ) from e
