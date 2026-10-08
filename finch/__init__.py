"""Finch.ai: Core Architecture and Modular Subsystems.

Submodules:
- storage: Database connection, WAL checkpoints, zstd column compressors, and FTS5 triggers.
- retrieval: Vector embeddings, cosine ranking (sqlite-vec), and Reciprocal Rank Fusion (k=60).
- memory: user_facts manager and personality.md parser.
- agent: Multi-turn execution loop, tool schemas, and 3-state permission engine (OFF/ASK/AUTO).
- compression: Headroom context compression pipeline.
- providers: LLM model providers (Ollama, Gemini, OpenAI-compatible).
- core: Orchestration engine.
- ui: Terminal interface.
"""
from .config import AppConfig, config
from .core.assistant import AIAssistant, AssistantMode, ToolPermissionMode

__all__ = [
    "AppConfig",
    "config",
    "AIAssistant",
    "AssistantMode",
    "ToolPermissionMode",
]
