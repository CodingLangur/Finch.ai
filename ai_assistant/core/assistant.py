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
    HybridSearchResult,
    SearchResult,
    SessionTranscript,
    load_session_transcript,
    search_hybrid,
    search_keyword,
)
from ..embeddings import BaseEmbedder, get_embedder
from .tools import CHAT_TOOLS, ToolDispatcher


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
        embedder: Optional[BaseEmbedder] = None,
    ):
        self.config = config or default_config
        self.embedder = embedder or get_embedder(self.config)
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
        self.archive = archive or SQLiteArchive(
            db_path=self.config.db_path,
            embedding_dim=self.config.embedding_dim,
        )
        self.summarizer = SessionSummarizer(archive=self.archive, provider=self.provider)
        self.tool_dispatcher = ToolDispatcher(self)

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
        self, user_input: str, enable_tools: bool = True
    ) -> AsyncGenerator[StreamChunk, None]:
        """Process user input, stream model response, autonomously execute tools (single-hop), and persist turn."""
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

        # 4. Stream from provider (Pass 1 with tools enabled)
        active_tools = CHAT_TOOLS if enable_tools else None
        accumulated_content: List[str] = []
        accumulated_thinking: List[str] = []
        accumulated_tool_calls: List[Dict[str, Any]] = []
        pass1_last_chunk: Optional[StreamChunk] = None

        try:
            async for chunk in self.provider.stream_chat(
                messages=effective_payload,
                model=self.active_model,
                tools=active_tools,
            ):
                if chunk.delta:
                    accumulated_content.append(chunk.delta)
                if chunk.thinking_delta:
                    accumulated_thinking.append(chunk.thinking_delta)
                if chunk.tool_calls:
                    accumulated_tool_calls.extend(chunk.tool_calls)

                pass1_last_chunk = chunk

                # Stream out real-time thinking and text as long as no tool calls are triggered
                if not accumulated_tool_calls:
                    if chunk.is_done:
                        chunk.compression_stats = compression_stats
                    yield chunk

            # Check if Pass 1 requested tool execution
            if accumulated_tool_calls:
                # SINGLE-HOP RETRIEVAL CYCLE:
                # User Query -> LLM Tool Call -> DB Fetch -> Final Answer
                first_tool = accumulated_tool_calls[0].get("function", {})
                tool_name = first_tool.get("name", "tool")
                tool_args = first_tool.get("arguments", {})
                arg_desc = ""
                if isinstance(tool_args, dict) and "query" in tool_args:
                    arg_desc = f": '{tool_args['query']}'"
                elif isinstance(tool_args, dict) and "session_id" in tool_args:
                    arg_desc = f": session='{tool_args['session_id']}'"

                notice_text = f"Accessing conversation archive via {tool_name}{arg_desc}..."
                yield StreamChunk(
                    tool_calls=accumulated_tool_calls,
                    tool_call_notice=notice_text,
                )

                # Execute requested tools via ToolDispatcher
                tool_results: List[Dict[str, Any]] = []
                for tc in accumulated_tool_calls:
                    fn_name = tc.get("function", {}).get("name", "")
                    fn_args = tc.get("function", {}).get("arguments", {})
                    output = await self.tool_dispatcher.execute(fn_name, fn_args)
                    tool_results.append({
                        "name": fn_name,
                        "output": output,
                        "id": tc.get("id"),
                    })

                # Build Pass 2 message payload
                pass2_messages = list(effective_payload)
                assistant_msg = {
                    "role": "assistant",
                    "content": "".join(accumulated_content) or "",
                    "tool_calls": accumulated_tool_calls,
                }
                pass2_messages.append(assistant_msg)

                for tr in tool_results:
                    tool_msg: Dict[str, Any] = {
                        "role": "tool",
                        "content": tr["output"],
                    }
                    if tr.get("id"):
                        tool_msg["tool_call_id"] = tr["id"]
                    pass2_messages.append(tool_msg)

                # Pass 2: Stream final answer with tools=None (strict single-hop enforcement)
                pass2_content: List[str] = []
                pass2_thinking: List[str] = []
                pass2_last_chunk: Optional[StreamChunk] = None

                async for chunk in self.provider.stream_chat(
                    messages=pass2_messages,
                    model=self.active_model,
                    tools=None,
                ):
                    if chunk.delta:
                        pass2_content.append(chunk.delta)
                    if chunk.thinking_delta:
                        pass2_thinking.append(chunk.thinking_delta)

                    if chunk.is_done:
                        chunk.compression_stats = compression_stats

                    pass2_last_chunk = chunk
                    yield chunk

                final_content = "".join(pass2_content).strip()
                final_thinking = "".join(pass2_thinking).strip() or None
                last_chunk = pass2_last_chunk
            else:
                final_content = "".join(accumulated_content).strip()
                final_thinking = "".join(accumulated_thinking).strip() or None
                last_chunk = pass1_last_chunk

            # 5. Once finished, inspect response for model-driven persona update
            clean_response, updated, _ = self.persona_manager.process_response(final_content)

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
                thinking=final_thinking,
            )

            # 6. Persist assistant turn to SQLite archive
            if (
                accumulated_tool_calls
                and pass1_last_chunk
                and pass1_last_chunk.stats
                and last_chunk
                and last_chunk.stats
            ):
                eval_tokens = pass1_last_chunk.stats.eval_count + last_chunk.stats.eval_count
            else:
                eval_tokens = (
                    last_chunk.stats.eval_count if (last_chunk and last_chunk.stats) else None
                )

            turn_meta: Dict[str, Any] = {}
            if accumulated_tool_calls:
                turn_meta["tool_calls"] = accumulated_tool_calls
                turn_meta["tools_executed"] = [
                    tc.get("function", {}).get("name") for tc in accumulated_tool_calls
                ]
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
                thinking=final_thinking,
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
        """Summarize conversational turns for a session, persist summary, and generate semantic vector."""
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

        # Generate and store summary vector embedding upon session completion
        if summary and self.archive.has_vec_support():
            try:
                emb_text = f"{title}: {summary}" if title else summary
                emb = await self.embedder.embed_text(emb_text)
                self.archive.store_session_embedding(target_id, emb)
            except Exception:
                pass

        return title, summary

    async def embed_session_summary(self, session_id: Optional[str] = None) -> bool:
        """Generate and store embedding for a session's summary in sessions_vec."""
        target_id = session_id or self.current_session.id
        sess = self.archive.get_session(target_id)
        if not sess or not sess.summary:
            return False
        emb_text = f"{sess.title}: {sess.summary}" if sess.title else sess.summary
        emb = await self.embedder.embed_text(emb_text)
        return self.archive.store_session_embedding(target_id, emb)

    async def search_hybrid(
        self,
        query: str,
        limit: int = 10,
        k: Optional[int] = None,
        lexical_weight: float = 1.0,
        semantic_weight: float = 1.0,
    ) -> List[HybridSearchResult]:
        """Execute Hybrid Search using Reciprocal Rank Fusion (RRF)."""
        return await search_hybrid(
            query=query,
            limit=limit,
            k=k or self.config.rrf_k,
            lexical_weight=lexical_weight,
            semantic_weight=semantic_weight,
            archive=self.archive,
            embedder=self.embedder,
        )

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
