"""Ollama LLM Provider implementation using httpx streaming."""
import json
import time
from typing import Any, AsyncGenerator, Dict, List, Optional
import httpx

from .base import BaseLLMProvider, ModelInfo, StreamChunk, StreamStats


class OllamaProvider(BaseLLMProvider):
    """Local Ollama client communicating via REST API."""

    def __init__(self, base_url: str = "http://localhost:11434", timeout: float = 120.0):
        # Normalize trailing slash
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @property
    def name(self) -> str:
        return "Ollama"

    async def health_check(self) -> bool:
        """Ping Ollama API to confirm service is alive."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await client.get(f"{self.base_url}/api/version")
                return res.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> List[ModelInfo]:
        """Fetch all locally downloaded models from Ollama."""
        url = f"{self.base_url}/api/tags"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.get(url)
                res.raise_for_status()
                data = res.json()
                models = []
                for item in data.get("models", []):
                    details = item.get("details", {})
                    models.append(
                        ModelInfo(
                            name=item.get("name", item.get("model", "")),
                            size_bytes=item.get("size", 0),
                            modified_at=item.get("modified_at", ""),
                            family=details.get("family", ""),
                            parameter_size=details.get("parameter_size", ""),
                            quantization=details.get("quantization_level", ""),
                        )
                    )
                return models
        except Exception as e:
            raise RuntimeError(
                f"Failed to query models from Ollama at {self.base_url}: {e}"
            ) from e

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream chat tokens from Ollama with real-time telemetry extraction."""
        url = f"{self.base_url}/api/chat"
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if options:
            payload["options"] = options

        start_time = time.perf_counter()
        first_token_time: Optional[float] = None
        total_tokens_received = 0

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                async with client.stream("POST", url, json=payload) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        raise RuntimeError(
                            f"Ollama API returned HTTP {response.status_code}: {error_text.decode('utf-8', errors='ignore')}"
                        )

                    async for line in response.aiter_lines():
                        if not line:
                            continue

                        try:
                            chunk_data = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        # Extract text delta or thinking delta
                        msg_dict = chunk_data.get("message", {})
                        delta = msg_dict.get("content", "")
                        thinking_delta = msg_dict.get("thinking", "")
                        is_done = chunk_data.get("done", False)

                        # Track Time To First Token
                        if (delta or thinking_delta) and first_token_time is None:
                            first_token_time = time.perf_counter()

                        stats: Optional[StreamStats] = None
                        if is_done:
                            end_time = time.perf_counter()
                            total_duration_ms = (end_time - start_time) * 1000.0
                            ttft_ms = (
                                (first_token_time - start_time) * 1000.0
                                if first_token_time
                                else total_duration_ms
                            )

                            eval_count = chunk_data.get("eval_count", total_tokens_received)
                            eval_duration_ms = chunk_data.get("eval_duration", 0) / 1e6
                            prompt_eval_count = chunk_data.get("prompt_eval_count", 0)
                            prompt_eval_duration_ms = (
                                chunk_data.get("prompt_eval_duration", 0) / 1e6
                            )
                            load_duration_ms = chunk_data.get("load_duration", 0) / 1e6

                            stats = StreamStats(
                                ttft_ms=round(ttft_ms, 2),
                                total_duration_ms=round(total_duration_ms, 2),
                                eval_count=eval_count,
                                eval_duration_ms=round(eval_duration_ms, 2),
                                prompt_eval_count=prompt_eval_count,
                                prompt_eval_duration_ms=round(prompt_eval_duration_ms, 2),
                                load_duration_ms=round(load_duration_ms, 2),
                            )
                            stats.calculate_speeds()

                        yield StreamChunk(
                            delta=delta,
                            thinking_delta=thinking_delta,
                            is_done=is_done,
                            stats=stats,
                        )

            except httpx.ConnectError as e:
                raise ConnectionError(
                    f"Could not connect to Ollama at {self.base_url}. Ensure Ollama is running (`ollama serve`)."
                ) from e
