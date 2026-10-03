"""Chat message representation."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ChatMessage:
    """Represents a single conversational turn."""
    role: Role
    content: str
    thinking: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, str]:
        """Convert message to provider-compatible dictionary (role + content)."""
        return {
            "role": self.role.value,
            "content": self.content
        }

    @classmethod
    def system(cls, content: str) -> "ChatMessage":
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> "ChatMessage":
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(cls, content: str, thinking: Optional[str] = None) -> "ChatMessage":
        return cls(role=Role.ASSISTANT, content=content, thinking=thinking)


# Alias for backward compatibility
Message = ChatMessage
