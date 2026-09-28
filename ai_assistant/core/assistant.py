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
from ..providers.openai_provider import OpenAICompatibleProvider
from ..storage.sqlite_archive import SQLiteArchive, SessionRecord, MessageRecord
from ..storage.session_summarizer import SessionSummarizer
from ..storage.backup import export_backup_bundle, import_backup_bundle
from ..storage.transcript_exporter import export_transcript_markdown, export_transcript_html
from ..storage.search import (
    HybridSearchResult,
    SearchResult,
    SessionTranscript,
    load_session_transcript,
    search_hybrid,
    search_keyword,
)
from ..embeddings import BaseEmbedder, get_embedder
from .tools import (
    AGENT_ACTION_TOOLS,
    AGENT_TOOLS,
    ALL_AGENT_TOOLS,
    CHAT_TOOLS,
    FETCH_WEB_PAGE_TOOL,
    LIST_DIRECTORY_TOOL,
    LOAD_SESSION_TRANSCRIPT_TOOL,
    PYTHON_INTERPRETER_TOOL,
    READ_FILE_TOOL,
    RUN_TERMINAL_COMMAND_TOOL,
    SEARCH_PAST_CONVERSATIONS_TOOL,
    TOOL_CATEGORIES,
    TOOL_NAME_TO_CATEGORY,
    WEB_SEARCH_TOOL,
    WEB_TOOLS,
    WRITE_FILE_TOOL,
    ToolDispatcher,
)


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


class ToolPermissionMode(str):
    """3-state tool permission mode ('off', 'ask', 'auto') with boolean evaluation support.

    Evaluates to False if 'off', and True if 'ask' or 'auto', preserving backward-compatibility
    with binary boolean checks.
    """

    OFF = "off"
    ASK = "ask"
    AUTO = "auto"

    def __bool__(self) -> bool:
        return self.lower() not in ("off", "disabled", "false", "0")


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
        elif self.config.provider.lower() in ("openai", "openai_compatible", "compatible"):
            self.provider = OpenAICompatibleProvider(
                api_key=self.config.openai_api_key,
                base_url=self.config.openai_base_url,
                timeout=self.config.request_timeout,
                default_model=self.config.openai_default_model,
            )
        else:
            self.provider = OllamaProvider(
                base_url=self.config.ollama_host,
                timeout=self.config.request_timeout,
            )
        self.active_model = model or self.config.effective_model
        self.mode = AssistantMode(self.config.default_mode)

        # Initialize SQLite Archive & Session Logging first (needed for facts & memory)
        self.archive = archive or SQLiteArchive(
            db_path=self.config.db_path,
            embedding_dim=self.config.embedding_dim,
            compress_large_messages=getattr(self.config, "compress_message_bodies", False),
            compression_threshold_bytes=getattr(self.config, "message_compression_threshold", 1024),
        )
        self.summarizer = SessionSummarizer(archive=self.archive, provider=self.provider)

        # Initialize PersonaManager and inject combined persona + user facts as system prompt
        self.persona_manager = PersonaManager(file_path=self.config.personality_path)
        system_prompt = self.build_effective_system_prompt()

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

        # Agent mode tool permission policy: 3-state control ('off', 'ask', 'auto')
        def _resolve_tool_mode(flag: Any) -> ToolPermissionMode:
            if isinstance(flag, str):
                v = flag.strip().lower()
                if v in ("auto", "full", "ai_control", "true", "1", "yes"):
                    return ToolPermissionMode("auto")
                if v in ("ask", "confirm", "prompt"):
                    return ToolPermissionMode("ask")
                return ToolPermissionMode("off")
            return ToolPermissionMode("auto" if bool(flag) else "off")

        self.tool_permissions: Dict[str, ToolPermissionMode] = {
            "terminal": _resolve_tool_mode(getattr(self.config, "enable_terminal_tool", True)),
            "python": _resolve_tool_mode(getattr(self.config, "enable_python_tool", True)),
            "web": _resolve_tool_mode(getattr(self.config, "enable_web_tool", False)),
            "files": _resolve_tool_mode(getattr(self.config, "enable_file_tools", True)),
        }
        self.tool_dispatcher = ToolDispatcher(self)
        self.tool_confirmation_callback = None

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

    def get_tool_permission_mode(self, category: str) -> str:
        """Get the current permission mode ('off', 'ask', 'auto') for a category."""
        return str(self.tool_permissions.get(category.lower(), ToolPermissionMode("off")))

    def is_tool_category_enabled(self, category: str) -> bool:
        """Check if a tool category is enabled (either 'ask' or 'auto')."""
        return self.get_tool_permission_mode(category) in ("ask", "auto")

    def set_tool_permission(self, category: str, mode: Any) -> ToolPermissionMode:
        """Set permission mode ('off', 'ask', 'auto') for a tool category."""
        cat = category.lower()
        if cat not in self.tool_permissions:
            raise ValueError(f"Unknown tool category '{category}'. Available: {list(self.tool_permissions.keys())}")

        if isinstance(mode, bool):
            normalized = "auto" if mode else "off"
        elif isinstance(mode, str):
            v = mode.strip().lower()
            if v in ("auto", "full", "ai_control", "on", "enable", "true", "1"):
                normalized = "auto"
            elif v in ("ask", "confirm", "prompt"):
                normalized = "ask"
            elif v in ("off", "disable", "disabled", "false", "0"):
                normalized = "off"
            else:
                normalized = "ask"
        else:
            normalized = "auto" if bool(mode) else "off"

        perm = ToolPermissionMode(normalized)
        self.tool_permissions[cat] = perm
        return perm

    def toggle_tool_permission(self, category: str) -> ToolPermissionMode:
        """Toggle permission for a category cycling through: off -> ask -> auto -> off."""
        cat = category.lower()
        if cat not in self.tool_permissions:
            raise ValueError(f"Unknown tool category '{category}'. Available: {list(self.tool_permissions.keys())}")

        current = str(self.tool_permissions[cat]).lower()
        cycle = {"off": "ask", "ask": "auto", "auto": "off"}
        new_mode = cycle.get(current, "ask")
        perm = ToolPermissionMode(new_mode)
        self.tool_permissions[cat] = perm
        return perm

    def get_tool_permissions(self) -> Dict[str, ToolPermissionMode]:
        """Return a copy of the current tool permissions dict."""
        return dict(self.tool_permissions)

    def get_tool_category(self, tool_name: str) -> str:
        """Return the category name for a given tool."""
        return TOOL_NAME_TO_CATEGORY.get(tool_name, "other")

    def get_active_tools(self, enable_tools: bool = True) -> Optional[List[Dict[str, Any]]]:
        """Resolve active tool definitions based on mode and enabled category toggles."""
        if not enable_tools:
            return None

        if self.mode == AssistantMode.CHAT:
            return list(CHAT_TOOLS)

        # In Agent Mode: dynamically assemble enabled tool categories
        tools: List[Dict[str, Any]] = list(CHAT_TOOLS)
        if self.is_tool_category_enabled("terminal"):
            tools.append(RUN_TERMINAL_COMMAND_TOOL)
        if self.is_tool_category_enabled("python"):
            tools.append(PYTHON_INTERPRETER_TOOL)
        if self.is_tool_category_enabled("web"):
            tools.extend(WEB_TOOLS)
        if self.is_tool_category_enabled("files"):
            tools.extend([READ_FILE_TOOL, WRITE_FILE_TOOL, LIST_DIRECTORY_TOOL])
        return tools

    def build_effective_system_prompt(self) -> str:
        """Compose the complete system prompt including persona guidelines and user memory facts."""
        base_prompt = self.persona_manager.build_system_prompt()
        if hasattr(self, "archive") and hasattr(self.archive, "list_facts"):
            facts = self.archive.list_facts()
            if facts:
                facts_lines = [
                    "\n\n---\n### Verified User Facts & Long-Term Memory",
                    "The following verified facts and preferences about the user and their environment are stored in long-term memory:",
                ]
                for f in facts:
                    cat = f.get("category", "general")
                    facts_lines.append(f"- [Fact #{f['id']}] ({cat}): {f['fact']}")
                return base_prompt + "\n" + "\n".join(facts_lines)
        return base_prompt

    def refresh_system_prompt(self) -> str:
        """Re-synthesize system prompt with persona and facts, and update sliding-window memory."""
        prompt = self.build_effective_system_prompt()
        self.memory.set_system_prompt(prompt)
        return prompt

    def reload_persona(self) -> str:
        """Reload personality.md from disk and refresh the pinned system prompt with user facts."""
        content = self.persona_manager.load_persona()
        self.refresh_system_prompt()
        return content

    def add_user_fact(self, fact: str, category: str = "general") -> int:
        """Store a verified user fact in long-term memory and refresh system prompt."""
        fact_id = self.archive.add_fact(fact=fact, category=category)
        self.refresh_system_prompt()
        return fact_id

    def list_user_facts(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve stored user facts from long-term memory."""
        return self.archive.list_facts(category=category)

    def delete_user_fact(self, fact_id: int) -> bool:
        """Delete a user fact by ID and refresh system prompt."""
        deleted = self.archive.delete_fact(fact_id)
        if deleted:
            self.refresh_system_prompt()
        return deleted

    def wipe_user_facts(self) -> int:
        """Wipe all user facts from long-term memory and refresh system prompt."""
        count = self.archive.wipe_all_facts()
        self.refresh_system_prompt()
        return count

    def export_transcript_markdown(
        self, session_id: Optional[str] = None, output_path: Optional[str] = None
    ) -> str:
        """Export session transcript to a GitHub-flavored Markdown file."""
        target_id = session_id or self.current_session.id
        return export_transcript_markdown(
            archive=self.archive,
            session_id=target_id,
            output_path=output_path,
        )

    def export_transcript_html(
        self, session_id: Optional[str] = None, output_path: Optional[str] = None
    ) -> str:
        """Export session transcript to a standalone, responsive HTML file."""
        target_id = session_id or self.current_session.id
        return export_transcript_html(
            archive=self.archive,
            session_id=target_id,
            output_path=output_path,
        )

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
        if "url" in tool_args:
            u = str(tool_args["url"]).strip()
            if len(u) > 35:
                u = u[:32] + "..."
            return f": '{u}'"
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

    def wipe_persona(self) -> str:
        """Reset personality.md back to default template and refresh in-memory system prompt."""
        content = self.persona_manager.wipe_persona()
        new_system_prompt = self.persona_manager.build_system_prompt()
        self.memory.set_system_prompt(new_system_prompt)
        return content

    def wipe_conversation(self, all_sessions: bool = False) -> Dict[str, Any]:
        """Wipe conversation history.

        Args:
            all_sessions: If True, wipes all sessions and messages across the entire database.
                          If False, wipes only the messages of the current active session.
        """
        self.clear_history()
        if all_sessions:
            stats = self.archive.wipe_all_conversations()
            self.new_session(title="Fresh Session")
            return {"all_sessions": True, **stats}
        else:
            wiped_count = self.archive.wipe_session_messages(self.current_session.id)
            return {
                "all_sessions": False,
                "session_id": self.current_session.id,
                "messages_wiped": wiped_count,
            }

    def wipe_current_conversation(self) -> Dict[str, Any]:
        """Wipe messages from the current active session in SQLite and clear memory buffer."""
        return self.wipe_conversation(all_sessions=False)

    def wipe_all_conversations(self) -> Dict[str, Any]:
        """Wipe all sessions and messages across the entire SQLite database and start a fresh session."""
        return self.wipe_conversation(all_sessions=True)

    def export_backup(self, output_path: Optional[str] = None) -> str:
        """Export conversation data and personality.md into a zip backup bundle."""
        backup_dir = getattr(self.config, "backup_dir", "backups")
        return export_backup_bundle(
            archive=self.archive,
            personality_path=self.config.personality_path,
            output_path=output_path,
            backup_dir=backup_dir,
        )

    def import_backup(self, backup_path: str, mode: str = "replace") -> Dict[str, Any]:
        """Restore conversation data and personality.md from a zip backup bundle."""
        result = import_backup_bundle(
            archive=self.archive,
            personality_path=self.config.personality_path,
            backup_path=backup_path,
            mode=mode,
        )
        self.reload_persona()
        if mode == "replace":
            recent = self.archive.list_sessions(limit=1)
            if recent:
                self.current_session = recent[0]
            else:
                self.new_session(title="Restored Session")
        self.clear_history()
        return result

    async def adapt_persona(
        self, session_ids: Optional[List[str]] = None, lookback_sessions: int = 5
    ) -> Tuple[bool, str]:
        """Analyze recent conversation summaries and adapt personality.md to user preferences."""
        if session_ids:
            sessions = [
                s for sid in session_ids
                if (s := self.archive.get_session(sid)) is not None
            ]
        else:
            sessions = self.archive.list_sessions(limit=lookback_sessions)

        if not sessions:
            return False, "No sessions found in the archive to adapt from."

        summaries_data: List[Dict[str, str]] = []
        for s in sessions:
            title = s.title
            summary = s.summary
            if not summary:
                msgs = self.archive.get_messages(s.id, include_system=False)
                if any(m.role.lower() == "user" for m in msgs):
                    try:
                        title, summary = await self.summarize_session(s.id)
                    except Exception:
                        summary = None
            if summary:
                summaries_data.append({"title": title, "summary": summary})

        if not summaries_data:
            msgs = self.archive.get_messages(self.current_session.id, include_system=False)
            user_msgs = [m for m in msgs if m.role.lower() == "user"]
            if user_msgs:
                sample = "\n".join(f"{m.role}: {m.content[:200]}" for m in msgs[-6:])
                summaries_data.append({"title": "Current Session Context", "summary": sample})
            else:
                return False, "No conversational interactions with user turns found to adapt persona."

        current_persona = self.persona_manager.load_persona()
        prompt = self.persona_manager.build_adaptation_prompt(current_persona, summaries_data)

        messages_payload = [
            {
                "role": "system",
                "content": (
                    "You are an AI persona adaptation specialist. Analyze conversation history and refine "
                    "personality.md to better match the user's communication style, interests, and domain preferences. "
                    "Ensure you emit the complete updated persona inside <personality_update>...</personality_update> tags."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        accumulated: List[str] = []
        try:
            async for chunk in self.provider.stream_chat(
                messages=messages_payload,
                model=self.active_model,
                tools=None,
            ):
                if chunk.delta:
                    accumulated.append(chunk.delta)
        except Exception as e:
            return False, f"LLM provider error during persona adaptation: {e}"

        raw_response = "".join(accumulated).strip()
        clean_text, updated, new_content = self.persona_manager.process_response(raw_response)

        if updated and new_content:
            self.reload_persona()
            return True, "Persona successfully adapted to user interactions and updated on disk!"

        if "# Assistant Persona" in raw_response:
            self.persona_manager.save_persona(raw_response)
            self.reload_persona()
            return True, "Persona updated from model markdown output."

        return False, "Model did not produce an updated persona inside <personality_update> tags."

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
                active_tools = self.get_active_tools(enable_tools=enable_tools)
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

    def run_maintenance(self, vacuum: bool = True) -> Dict[str, Any]:
        """Perform database health check, WAL checkpoint, query planner optimization, and optional VACUUM."""
        return self.archive.run_maintenance(vacuum=vacuum)

    def vacuum(self) -> Dict[str, Any]:
        """Execute SQLite VACUUM to defragment B-trees and reclaim disk space."""
        return self.archive.vacuum()

    def get_storage_stats(self) -> Dict[str, Any]:
        """Retrieve database disk footprint, page counts, WAL size, and record counts."""
        return self.archive.get_storage_stats()

    def archive_sessions(
        self,
        older_than_days: Optional[int] = None,
        session_ids: Optional[List[str]] = None,
        archive_db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Archive inactive sessions to an archive database and vacuum main database."""
        return self.archive.archive_sessions(
            older_than_days=older_than_days,
            session_ids=session_ids,
            archive_db_path=archive_db_path,
        )

    def close(self) -> None:
        """Cleanly close the underlying database connection."""
        self.archive.close()
