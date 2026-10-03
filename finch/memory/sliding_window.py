"""Sliding-Window in-memory message buffer."""
from typing import Any, Dict, List, Optional
from .message import ChatMessage, Role


class SlidingWindowBuffer:
    """Maintains an active in-memory list of recent turns.

    Guarantees:
    - System prompt is permanently preserved at the head of the context.
    - Conversation history is strictly bounded to `max_messages` (default 8).
    - When pruning is required, the oldest non-system turns are evicted.
    """

    def __init__(self, max_messages: int = 8, system_prompt: Optional[str] = None):
        if max_messages < 2:
            raise ValueError("max_messages must be at least 2 to hold a conversation turn.")
        self.max_messages = max_messages
        self._system_message: Optional[ChatMessage] = None
        self._messages: List[ChatMessage] = []

        if system_prompt:
            self.set_system_prompt(system_prompt)

    def set_system_prompt(self, content: str) -> None:
        """Set or update the pinned system prompt."""
        self._system_message = ChatMessage.system(content.strip())

    def get_system_prompt(self) -> Optional[str]:
        """Return the current system prompt content, if set."""
        return self._system_message.content if self._system_message else None

    def add_message(self, message: ChatMessage) -> None:
        """Add a ChatMessage to the buffer, triggering sliding prune if necessary."""
        if message.role == Role.SYSTEM:
            self._system_message = message
            return

        self._messages.append(message)
        self._prune()

    def add_user_message(self, content: str) -> ChatMessage:
        """Helper to create and append a user message."""
        msg = ChatMessage.user(content)
        self.add_message(msg)
        return msg

    def add_assistant_message(
        self, content: str, thinking: Optional[str] = None
    ) -> ChatMessage:
        """Helper to create and append an assistant message."""
        msg = ChatMessage.assistant(content, thinking=thinking)
        self.add_message(msg)
        return msg

    def _prune(self) -> None:
        """Trim oldest conversational messages if count exceeds max_messages."""
        while len(self._messages) > self.max_messages:
            # Pop the oldest conversational message
            self._messages.pop(0)

    def clear(self, keep_system: bool = True) -> None:
        """Reset the conversation history."""
        self._messages.clear()
        if not keep_system:
            self._system_message = None

    def get_messages(self) -> List[ChatMessage]:
        """Return full list of messages including pinned system message."""
        result: List[ChatMessage] = []
        if self._system_message:
            result.append(self._system_message)
        result.extend(self._messages)
        return result

    def to_provider_payload(self) -> List[Dict[str, str]]:
        """Return provider-compatible list of message dicts (role, content)."""
        return [msg.to_dict() for msg in self.get_messages()]

    def get_stats(self) -> Dict[str, Any]:
        """Return context buffer statistics."""
        all_msgs = self.get_messages()
        total_chars = sum(len(m.content) for m in all_msgs)
        # Rough token approximation (~4 chars per token for English text)
        approx_tokens = total_chars // 4

        user_count = sum(1 for m in self._messages if m.role == Role.USER)
        assistant_count = sum(1 for m in self._messages if m.role == Role.ASSISTANT)

        return {
            "window_size": self.max_messages,
            "active_messages": len(self._messages),
            "total_messages": len(all_msgs),
            "user_messages": user_count,
            "assistant_messages": assistant_count,
            "has_system_prompt": self._system_message is not None,
            "total_characters": total_chars,
            "approx_tokens": approx_tokens,
        }

    def __len__(self) -> int:
        return len(self._messages) + (1 if self._system_message else 0)

    def __iter__(self):
        return iter(self.get_messages())
