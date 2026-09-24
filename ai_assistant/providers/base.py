"""Base LLM Provider interface and stream metrics."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional


@dataclass
class ModelInfo:
    """Information about an available model."""
    name: str
    size_bytes: int = 0
    modified_at: str = ""
    family: str = ""
    parameter_size: str = ""
    quantization: str = ""

    @property
    def size_gb(self) -> float:
        return round(self.size_bytes / (1024 ** 3), 2)


@dataclass
class StreamStats:
    """Execution telemetry captured during streaming."""
    ttft_ms: float = 0.0                      # Client Time to First Token
    total_duration_ms: float = 0.0            # Total elapsed time
    eval_count: int = 0                       # Generated token count
    eval_duration_ms: float = 0.0             # Model generation time
    prompt_eval_count: int = 0                # Prompt token count
    prompt_eval_duration_ms: float = 0.0      # Prompt eval time
    load_duration_ms: float = 0.0             # Model load time (if cold)
    tokens_per_second: float = 0.0            # Hardware generation speed
    client_tokens_per_second: float = 0.0     # End-to-end client speed

    def calculate_speeds(self) -> None:
        """Derive speed metrics from counts and durations."""
        if self.eval_duration_ms > 0 and self.eval_count > 0:
            self.tokens_per_second = round(
                self.eval_count / (self.eval_duration_ms / 1000.0), 2
            )
        generation_window_sec = (self.total_duration_ms - self.ttft_ms) / 1000.0
        if generation_window_sec > 0 and self.eval_count > 0:
            self.client_tokens_per_second = round(
                self.eval_count / generation_window_sec, 2
            )


@dataclass
class StreamChunk:
    """A streaming chunk received from the provider."""
    delta: str = ""
    thinking_delta: str = ""
    is_done: bool = False
    stats: Optional[StreamStats] = None
    persona_updated: bool = False
    compression_stats: Optional[Any] = None


class BaseLLMProvider(ABC):
    """Abstract interface for LLM backends (Ollama, OpenAI, Anthropic, etc.)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if provider backend is reachable."""
        pass

    @abstractmethod
    async def list_models(self) -> List[ModelInfo]:
        """Fetch list of models available from the provider."""
        pass

    @abstractmethod
    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream chat completions turn-by-turn."""
        pass
