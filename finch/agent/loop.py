"""Sequential Multi-Turn Agent Execution Loop for Finch.ai."""
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

from ..providers.base import BaseLLMProvider, StreamChunk


@dataclass
class AgentLoopResult:
    """Outcome and metadata of a multi-turn agent execution run."""
    final_content: str
    final_thinking: Optional[str]
    accumulated_tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    all_tools_executed: List[str] = field(default_factory=list)
    agent_turns: int = 1
    total_eval_tokens: int = 0
    last_chunk: Optional[StreamChunk] = None


class AgentLoop:
    """Executes sequential tool calls in a multi-turn while loop until completion or max turns reached."""

    def __init__(
        self,
        provider: BaseLLMProvider,
        model: str,
        tool_dispatcher: Any,
        format_tool_notice_fn: Optional[Callable[[str, Any], str]] = None,
        max_turns: int = 10,
    ):
        self.provider = provider
        self.model = model
        self.tool_dispatcher = tool_dispatcher
        self.format_tool_notice_fn = format_tool_notice_fn
        self.max_turns = max_turns

    def _format_notice(self, tool_name: str, tool_args: Any) -> str:
        if self.format_tool_notice_fn:
            return self.format_tool_notice_fn(tool_name, tool_args)
        if isinstance(tool_args, dict) and tool_args:
            first_val = next(iter(tool_args.values()))
            return f" ({first_val})"
        return ""

    async def execute_stream(
        self,
        running_payload: List[Dict[str, Any]],
        active_tools: Optional[List[Dict[str, Any]]],
        compression_stats: Optional[Any] = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Run the sequential multi-turn loop and yield stream chunks for live terminal/UI feedback."""
        turn_count = 0
        total_eval_tokens = 0
        accumulated_tool_calls: List[Dict[str, Any]] = []
        all_tools_executed: List[str] = []
        final_content = ""
        final_thinking = None
        last_chunk: Optional[StreamChunk] = None

        while turn_count < self.max_turns:
            turn_count += 1
            turn_content: List[str] = []
            turn_thinking: List[str] = []
            turn_tool_calls: List[Dict[str, Any]] = []
            turn_last_chunk: Optional[StreamChunk] = None

            async for chunk in self.provider.stream_chat(
                messages=running_payload,
                model=self.model,
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
                arg_desc = self._format_notice(fn_name, fn_args)
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
