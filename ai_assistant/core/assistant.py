"""AIAssistant orchestrator managing providers, context memory, and modes."""
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional

from ..config import AppConfig, config as default_config
from ..compression.pipeline import ContextCompressionPipeline, CompressionStats
from ..memory.sliding_window import SlidingWindowBuffer
from ..persona.manager import PersonaManager
from ..providers.base import BaseLLMProvider, ModelInfo, StreamChunk
from ..providers.ollama_provider import OllamaProvider
from ..providers.gemini_provider import GeminiProvider
from ..storage.sqlite_archive import SQLiteArchive, SessionRecord, MessageRecord
from ..storage.session_summarizer import SessionSummarizer
from ..storage.search import (
    SearchResult,
    SessionTranscript,
    load_session_transcript,
    search_keyword,
)


class AssistantMode(str, Enum):
    CHATBOT = "chatbot"
    AGENT = "agent"


class AIAssistant:
    """Central orchestrator for local LLM chat, context compression, and future agentic flows."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        provider: Optional[BaseLLMProvider] = None,
        model: Optional[str] = None,
        archive: Optional[SQLiteArchive] = None,
    ):
        self.config = config or default_config
        if provider:
            self.provider = provider
        elif self.config.provider.lower() == "gemini" and self.config.gemini_api_key:
            self.provider = GeminiProvider(
                api_key=self.config.gemini_api_key,
                timeout=self.config.request_timeout,
            )
        else:
            self.provider = OllamaProvider(
                base_url=self.config.ollama_host,
                timeout=self.config.request_timeout,
            )
        self.active_model = model or self.config.effective_model
        self.mode = AssistantMode(self.config.default_mode)

        # Initialize PersonaManager and inject personality.md as system prompt
        self.persona_manager = PersonaManager(file_path=self.config.personality_path)
        system_prompt = self.persona_manager.build_system_prompt()

        self.memory = SlidingWindowBuffer(
            max_messages=self.config.window_size,
            system_prompt=system_prompt,
        )

        # Initialize Headroom Context Compression Pipeline
        self.compression_pipeline = ContextCompressionPipeline(
            enabled=self.config.compression_enabled,
            compress_user_messages=self.config.compress_user_messages,
            compress_system_messages=self.config.compress_system_messages,
            protect_recent=self.config.compression_protect_recent,
            min_tokens_to_compress=self.config.compression_min_tokens,
        )

        # Initialize SQLite Archive & Session Logging
        self.archive = archive or SQLiteArchive(db_path=self.config.db_path)
        self.summarizer = SessionSummarizer(archive=self.archive, provider=self.provider)

        # Create the initial active session in SQLite
        self.current_session: SessionRecord = self.archive.create_session(
            model=self.active_model,
            mode=self.mode.value,
        )
        if system_prompt:
            self.archive.add_message(
                session_id=self.current_session.id,
                role="system",
                content=system_prompt,
            )

    def reload_persona(self) -> str:
        """Reload personality.md from disk and refresh the pinned system prompt."""
        content = self.persona_manager.load_persona()
        new_system_prompt = self.persona_manager.build_system_prompt()
        self.memory.set_system_prompt(new_system_prompt)
        return content

    def set_mode(self, mode: AssistantMode | str) -> AssistantMode:
        """Switch between Chatbot and Agent modes."""
        if isinstance(mode, str):
            mode = AssistantMode(mode.lower())
        self.mode = mode
        return self.mode

    def toggle_mode(self) -> AssistantMode:
        """Toggle between Chatbot and Agent modes."""
        if self.mode == AssistantMode.CHATBOT:
            self.mode = AssistantMode.AGENT
        else:
            self.mode = AssistantMode.CHATBOT
        return self.mode

    def set_model(self, model_name: str) -> None:
        """Switch the active generation model."""
        self.active_model = model_name.strip()

    def clear_history(self) -> None:
        """Reset conversation memory while preserving system prompt."""
        self.memory.clear(keep_system=True)

    async def list_available_models(self) -> List[ModelInfo]:
        """Query available models from the provider."""
        return await self.provider.list_models()

    async def verify_provider_health(self) -> bool:
        """Check provider connectivity."""
        return await self.provider.health_check()

    async def chat_stream(
        self, user_input: str
    ) -> AsyncGenerator[StreamChunk, None]:
        """Process user input, stream model response, handle persona updates, and append turn to buffer & SQLite."""
        # 1. Add user message to sliding-window memory and persist to SQLite
        self.memory.add_user_message(user_input)
        user_msg_id = self.archive.add_message(
            session_id=self.current_session.id,
            role="user",
            content=user_input,
        )

        # 2. Build provider message payload
        messages_payload = self.memory.to_provider_payload()

        # 3. Compress messages via Headroom pipeline before sending to LLM
        effective_payload, compression_stats = self.compression_pipeline.compress_messages(
            messages=messages_payload,
            model=self.active_model,
        )

        # 4. Stream from provider
        accumulated_content: List[str] = []
        accumulated_thinking: List[str] = []
        last_chunk: Optional[StreamChunk] = None

        try:
            async for chunk in self.provider.stream_chat(
                messages=effective_payload,
                model=self.active_model,
            ):
                if chunk.delta:
                    accumulated_content.append(chunk.delta)
                if chunk.thinking_delta:
                    accumulated_thinking.append(chunk.thinking_delta)

                if chunk.is_done:
                    chunk.compression_stats = compression_stats

                last_chunk = chunk
                yield chunk

            # 5. Once finished, inspect response for model-driven persona update
            full_response = "".join(accumulated_content).strip()
            full_thinking = "".join(accumulated_thinking).strip() or None

            clean_response, updated, _ = self.persona_manager.process_response(full_response)

            if updated:
                # Dynamically update the pinned system prompt in memory
                new_sys_prompt = self.persona_manager.build_system_prompt()
                self.memory.set_system_prompt(new_sys_prompt)

                # Emit a final notification chunk so UI knows persona changed
                yield StreamChunk(
                    delta="",
                    is_done=True,
                    persona_updated=True,
                    stats=last_chunk.stats if last_chunk else None,
                    compression_stats=compression_stats,
                )

            # Record clean conversational turn in sliding window
            self.memory.add_assistant_message(
                content=clean_response,
                thinking=full_thinking,
            )

            # 6. Persist assistant turn to SQLite archive
            eval_tokens = (
                last_chunk.stats.eval_count if (last_chunk and last_chunk.stats) else None
            )
            turn_meta: Dict[str, Any] = {}
            if compression_stats and compression_stats.tokens_before > 0:
                turn_meta["compression"] = {
                    "tokens_before": compression_stats.tokens_before,
                    "tokens_after": compression_stats.tokens_after,
                    "tokens_saved": compression_stats.tokens_saved,
                }
            if last_chunk and last_chunk.stats:
                turn_meta["ttft_ms"] = last_chunk.stats.ttft_ms
                turn_meta["tokens_per_second"] = (
                    last_chunk.stats.tokens_per_second
                    if last_chunk.stats.tokens_per_second > 0
                    else last_chunk.stats.client_tokens_per_second
                )

            self.archive.add_message(
                session_id=self.current_session.id,
                role="assistant",
                content=clean_response,
                thinking=full_thinking,
                tokens=eval_tokens,
                metadata=turn_meta,
            )

        except Exception:
            # If generation fails, remove the last user message from in-memory buffer
            if len(self.memory._messages) > 0 and self.memory._messages[-1].content == user_input:
                self.memory._messages.pop()
            raise

    def new_session(self, title: str = "New Session") -> SessionRecord:
        """Start a fresh session in SQLite and reset the in-memory context buffer."""
        self.clear_history()
        self.current_session = self.archive.create_session(
            title=title,
            model=self.active_model,
            mode=self.mode.value,
        )
        sys_prompt = self.memory.get_system_prompt()
        if sys_prompt:
            self.archive.add_message(
                session_id=self.current_session.id,
                role="system",
                content=sys_prompt,
            )
        return self.current_session

    async def summarize_session(
        self, session_id: Optional[str] = None
    ) -> Tuple[str, str]:
        """Summarize conversational turns for a session and persist title & 2-sentence summary."""
        target_id = session_id or self.current_session.id
        title, summary = await self.summarizer.summarize_session(
            session_id=target_id,
            model=self.active_model,
        )
        # Refresh current session in memory if it was the target
        if target_id == self.current_session.id:
            updated = self.archive.get_session(target_id)
            if updated:
                self.current_session = updated
        return title, summary

    def get_current_session(self) -> Optional[SessionRecord]:
        """Fetch the current session record with updated message counts."""
        return self.archive.get_session(self.current_session.id)

    def list_sessions(self, limit: int = 50, offset: int = 0) -> List[SessionRecord]:
        """Fetch a list of past conversation sessions."""
        return self.archive.list_sessions(limit=limit, offset=offset)

    def search_keyword(
        self,
        query: str,
        session_id: Optional[str] = None,
        limit: int = 10,
        exact_match: bool = True,
    ) -> List[SearchResult]:
        """Execute lexical FTS5 search across archived messages."""
        return search_keyword(
            query=query,
            session_id=session_id,
            limit=limit,
            exact_match=exact_match,
            archive=self.archive,
        )

    def load_session_transcript(
        self, session_id: str, include_system: bool = False
    ) -> Optional[SessionTranscript]:
        """Retrieve full dialogue transcript for an identified session."""
        return load_session_transcript(
            session_id=session_id,
            include_system=include_system,
            archive=self.archive,
        )

    def close(self) -> None:
        """Cleanly close the underlying database connection."""
        self.archive.close()
