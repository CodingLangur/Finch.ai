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
from .tools import AGENT_TOOLS, CHAT_TOOLS, ToolDispatcher


class AssistantMode(str, Enum):
    CHAT = "chat"
    AGENT = "agent"

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            val = value.strip().lower()
            if val in ("chat", "chatbot"):
                return cls.CHAT
            if val == "agent":
                return cls.AGENT
        return super()._missing_(value)


# Backward-compatibility alias
AssistantMode.CHATBOT = AssistantMode.CHAT


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
        """Switch between Chat and Agent modes."""
        if isinstance(mode, str):
            self.mode = AssistantMode(mode)
        else:
            self.mode = mode
        try:
            self.archive.update_session_mode(self.current_session.id, self.mode.value)
        except Exception:
            pass
        return self.mode

    def toggle_mode(self) -> AssistantMode:
        """Toggle between Chat and Agent modes."""
        if self.mode == AssistantMode.AGENT:
            return self.set_mode(AssistantMode.CHAT)
        else:
            return self.set_mode(AssistantMode.AGENT)

    def _format_tool_notice(self, tool_name: str, tool_args: Any) -> str:
        """Format a human-readable argument preview for tool execution notices."""
        if not isinstance(tool_args, dict):
            return ""
        if "query" in tool_args:
            return f": '{tool_args['query']}'"
        if "session_id" in tool_args:
            return f": session='{tool_args['session_id']}'"
        if "command" in tool_args:
            cmd = str(tool_args["command"]).strip()
            if len(cmd) > 30:
                cmd = cmd[:27] + "..."
            return f": '{cmd}'"
        if "file_path" in tool_args:
            return f": '{tool_args['file_path']}'"
        if "directory_path" in tool_args:
            return f": '{tool_args['directory_path']}'"
        if "code" in tool_args:
            first_line = str(tool_args["code"]).strip().splitlines()[0] if tool_args["code"] else ""
            if len(first_line) > 30:
                first_line = first_line[:27] + "..."
            return f": {first_line}"
        return ""

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
        """Process user input, stream model response, execute tools, and persist turn.
        
        In Chat mode: Restricts tools strictly to memory retrieval and enforces a single retrieval cycle.
        In Agent mode: Executes sequential tool calls in a multi-turn while loop until completion.
        """
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

        # 4. Determine execution flow based on mode
        is_agent_mode = self.mode == AssistantMode.AGENT
        accumulated_tool_calls: List[Dict[str, Any]] = []
        all_tools_executed: List[str] = []
        final_content: str = ""
        final_thinking: Optional[str] = None
        last_chunk: Optional[StreamChunk] = None
        total_eval_tokens = 0
        agent_turns = 1

        try:
            if not is_agent_mode:
                # =========================================================================
                # CHAT MODE: Strict Single-Hop Retrieval Cycle
                # User Query -> LLM Tool Call -> DB Fetch -> Final Answer
                # =========================================================================
                active_tools = CHAT_TOOLS if enable_tools else None
                pass1_content: List[str] = []
                pass1_thinking: List[str] = []
                pass1_tool_calls: List[Dict[str, Any]] = []
                pass1_last_chunk: Optional[StreamChunk] = None

                async for chunk in self.provider.stream_chat(
                    messages=effective_payload,
                    model=self.active_model,
                    tools=active_tools,
                ):
                    if chunk.delta:
                        pass1_content.append(chunk.delta)
                    if chunk.thinking_delta:
                        pass1_thinking.append(chunk.thinking_delta)
                    if chunk.tool_calls:
                        pass1_tool_calls.extend(chunk.tool_calls)

                    pass1_last_chunk = chunk

                    # Stream out real-time thinking and text as long as no tool calls are triggered
                    if not pass1_tool_calls:
                        if chunk.is_done:
                            chunk.compression_stats = compression_stats
                        yield chunk

                if pass1_last_chunk and pass1_last_chunk.stats:
                    total_eval_tokens += pass1_last_chunk.stats.eval_count

                if pass1_tool_calls:
                    accumulated_tool_calls.extend(pass1_tool_calls)
                    first_tool = pass1_tool_calls[0].get("function", {})
                    tool_name = first_tool.get("name", "tool")
                    tool_args = first_tool.get("arguments", {})
                    arg_desc = self._format_tool_notice(tool_name, tool_args)

                    notice_text = f"Accessing conversation archive via {tool_name}{arg_desc}..."
                    yield StreamChunk(
                        tool_calls=pass1_tool_calls,
                        tool_call_notice=notice_text,
                    )

                    # Execute requested tools via ToolDispatcher
                    tool_results: List[Dict[str, Any]] = []
                    for tc in pass1_tool_calls:
                        fn_name = tc.get("function", {}).get("name", "")
                        fn_args = tc.get("function", {}).get("arguments", {})
                        all_tools_executed.append(fn_name)
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
                        "content": "".join(pass1_content) or "",
                        "tool_calls": pass1_tool_calls,
                    }
                    pass2_messages.append(assistant_msg)

                    for tr in tool_results:
                        tool_msg: Dict[str, Any] = {
                            "role": "tool",
                            "name": tr["name"],
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

                    if pass2_last_chunk and pass2_last_chunk.stats:
                        total_eval_tokens += pass2_last_chunk.stats.eval_count

                    final_content = "".join(pass2_content).strip()
                    final_thinking = "".join(pass2_thinking).strip() or None
                    last_chunk = pass2_last_chunk
                else:
                    final_content = "".join(pass1_content).strip()
                    final_thinking = "".join(pass1_thinking).strip() or None
                    last_chunk = pass1_last_chunk

            else:
                # =========================================================================
                # AGENT MODE: Sequential Multi-Turn While Loop
                # Executes tool calls sequentially until completion or max turns reached
                # =========================================================================
                active_tools = AGENT_TOOLS if enable_tools else None
                max_turns = getattr(self.config, "agent_max_turns", 10)
                running_payload = list(effective_payload)
                turn_count = 0
                turn_tool_calls: List[Dict[str, Any]] = []

                while turn_count < max_turns:
                    turn_count += 1
                    agent_turns = turn_count
                    turn_content: List[str] = []
                    turn_thinking: List[str] = []
                    turn_tool_calls = []
                    turn_last_chunk: Optional[StreamChunk] = None

                    async for chunk in self.provider.stream_chat(
                        messages=running_payload,
                        model=self.active_model,
                        tools=active_tools,
                    ):
                        if chunk.delta:
                            turn_content.append(chunk.delta)
                        if chunk.thinking_delta:
                            turn_thinking.append(chunk.thinking_delta)
                        if chunk.tool_calls:
                            turn_tool_calls.extend(chunk.tool_calls)

                        turn_last_chunk = chunk

                        # If no tool calls in this turn, stream out real-time tokens to user
                        if not turn_tool_calls:
                            if chunk.is_done:
                                chunk.compression_stats = compression_stats
                            yield chunk

                    if turn_last_chunk and turn_last_chunk.stats:
                        total_eval_tokens += turn_last_chunk.stats.eval_count
                    last_chunk = turn_last_chunk

                    if not turn_tool_calls:
                        # Model produced final textual answer without calling any further tools
                        final_content = "".join(turn_content).strip()
                        final_thinking = "".join(turn_thinking).strip() or None
                        break

                    # Tool calls were emitted in this step
                    accumulated_tool_calls.extend(turn_tool_calls)
                    for tc in turn_tool_calls:
                        fn_name = tc.get("function", {}).get("name", "tool")
                        fn_args = tc.get("function", {}).get("arguments", {})
                        arg_desc = self._format_tool_notice(fn_name, fn_args)
                        notice_text = f"Agent Step {turn_count}: Executing {fn_name}{arg_desc}..."
                        yield StreamChunk(
                            tool_calls=[tc],
                            tool_call_notice=notice_text,
                        )

                    # Execute requested tools via ToolDispatcher
                    tool_results: List[Dict[str, Any]] = []
                    for tc in turn_tool_calls:
                        fn_name = tc.get("function", {}).get("name", "")
                        fn_args = tc.get("function", {}).get("arguments", {})
                        all_tools_executed.append(fn_name)
                        output = await self.tool_dispatcher.execute(fn_name, fn_args)
                        tool_results.append({
                            "name": fn_name,
                            "output": output,
                            "id": tc.get("id"),
                        })

                    # Append assistant message with tool calls to running payload
                    running_payload.append({
                        "role": "assistant",
                        "content": "".join(turn_content) or "",
                        "tool_calls": turn_tool_calls,
                    })

                    # Append tool result messages
                    for tr in tool_results:
                        tool_msg: Dict[str, Any] = {
                            "role": "tool",
                            "name": tr["name"],
                            "content": tr["output"],
                        }
                        if tr.get("id"):
                            tool_msg["tool_call_id"] = tr["id"]
                        running_payload.append(tool_msg)

                # If max turns reached and model was still calling tools, synthesize final answer
                if not final_content and turn_tool_calls:
                    synth_content: List[str] = []
                    synth_thinking: List[str] = []
                    synth_last_chunk: Optional[StreamChunk] = None

                    async for chunk in self.provider.stream_chat(
                        messages=running_payload,
                        model=self.active_model,
                        tools=None,
                    ):
                        if chunk.delta:
                            synth_content.append(chunk.delta)
                        if chunk.thinking_delta:
                            synth_thinking.append(chunk.thinking_delta)

                        if chunk.is_done:
                            chunk.compression_stats = compression_stats

                        synth_last_chunk = chunk
                        yield chunk

                    if synth_last_chunk and synth_last_chunk.stats:
                        total_eval_tokens += synth_last_chunk.stats.eval_count
                    last_chunk = synth_last_chunk
                    final_content = "".join(synth_content).strip()
                    final_thinking = "".join(synth_thinking).strip() or None

            # 5. Once finished, inspect response for model-driven persona update
            clean_response, updated, _ = self.persona_manager.process_response(final_content)

            if updated:
                new_sys_prompt = self.persona_manager.build_system_prompt()
                self.memory.set_system_prompt(new_sys_prompt)

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
            eval_tokens = total_eval_tokens if total_eval_tokens > 0 else (
                last_chunk.stats.eval_count if (last_chunk and last_chunk.stats) else None
            )

            turn_meta: Dict[str, Any] = {
                "mode": self.mode.value,
            }
            if is_agent_mode:
                turn_meta["agent_turns"] = agent_turns
            if accumulated_tool_calls:
                turn_meta["tool_calls"] = accumulated_tool_calls
                turn_meta["tools_executed"] = all_tools_executed
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
