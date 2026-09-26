"""Configuration settings for Finch.ai."""
import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

# Load .env file automatically into environment variables
load_dotenv()


@dataclass
class AppConfig:
    """Application configuration options."""
    provider: str = field(
        default_factory=lambda: os.getenv("DEFAULT_PROVIDER", "ollama")
    )
    gemini_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY")
    )
    gemini_default_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_DEFAULT_MODEL", "gemini-2.5-flash")
    )
    ollama_host: str = field(
        default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434")
    )
    default_model: str = field(
        default_factory=lambda: os.getenv(
            "DEFAULT_MODEL",
            "gemini-2.5-flash"
            if os.getenv("DEFAULT_PROVIDER", "ollama").lower() == "gemini"
            else os.getenv("OLLAMA_DEFAULT_MODEL", "Gemma4-26000-ctx:latest"),
        )
    )

    @property
    def effective_model(self) -> str:
        """Return provider-appropriate default model."""
        if self.provider.lower() == "gemini":
            if "gemini" in self.default_model.lower():
                return self.default_model
            return self.gemini_default_model
        return self.default_model
    window_size: int = field(
        default_factory=lambda: int(os.getenv("WINDOW_SIZE", "8"))
    )
    default_mode: str = field(
        default_factory=lambda: os.getenv("DEFAULT_MODE", "chat")
    )
    agent_max_turns: int = field(
        default_factory=lambda: int(os.getenv("AGENT_MAX_TURNS", "10"))
    )
    system_prompt: str = field(
        default_factory=lambda: os.getenv(
            "SYSTEM_PROMPT",
            "You are Finch.ai, a helpful, concise, and accurate AI assistant. "
            "Provide clear and direct answers."
        )
    )
    request_timeout: float = field(
        default_factory=lambda: float(os.getenv("REQUEST_TIMEOUT", "120.0"))
    )
    personality_path: str = field(
        default_factory=lambda: os.getenv("PERSONALITY_FILE", "personality.md")
    )
    compression_enabled: bool = field(
        default_factory=lambda: os.getenv("COMPRESSION_ENABLED", "true").lower() in ("true", "1", "yes")
    )
    compression_min_tokens: int = field(
        default_factory=lambda: int(os.getenv("COMPRESSION_MIN_TOKENS", "100"))
    )
    compression_protect_recent: int = field(
        default_factory=lambda: int(os.getenv("COMPRESSION_PROTECT_RECENT", "2"))
    )
    compress_user_messages: bool = field(
        default_factory=lambda: os.getenv("COMPRESS_USER_MESSAGES", "true").lower() in ("true", "1", "yes")
    )
    compress_system_messages: bool = field(
        default_factory=lambda: os.getenv("COMPRESS_SYSTEM_MESSAGES", "false").lower() in ("true", "1", "yes")
    )
    db_path: str = field(
        default_factory=lambda: os.getenv("DATABASE_PATH", "conversations.db")
    )
    auto_summarize_on_exit: bool = field(
        default_factory=lambda: os.getenv("AUTO_SUMMARIZE_ON_EXIT", "true").lower() in ("true", "1", "yes")
    )
    embedding_provider: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "ollama")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    )
    embedding_dim: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "768"))
    )
    rrf_k: int = field(
        default_factory=lambda: int(os.getenv("RRF_K", "60"))
    )


# Global singleton configuration
config = AppConfig()
