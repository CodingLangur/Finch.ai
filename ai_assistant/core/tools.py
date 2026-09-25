"""Tool schemas and execution dispatcher for autonomous conversation retrieval."""
import json
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .assistant import AIAssistant


# Tool Schemas for Ollama / Function Calling API
SEARCH_PAST_CONVERSATIONS_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_past_conversations",
        "description": (
            "Search archived conversation history using hybrid retrieval (semantic vector search + exact lexical matching) "
            "to recall topics, previous discussions, code snippets, dates, or decisions. "
            "Call this tool ONLY when the user asks about or refers to previous/past conversations or earlier topics. "
            "Do NOT call this tool for general knowledge, coding assistance, or questions about the current active turn."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query, topic, phrase, or question to search for in past conversations.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of relevant past sessions to retrieve (default: 5).",
                },
            },
            "required": ["query"],
        },
    },
}

LOAD_SESSION_TRANSCRIPT_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "load_session_transcript",
        "description": (
            "Retrieve the complete chronological dialogue transcript for an identified past conversation session. "
            "Call this tool when you need the exact dialogue turns, code snippets, or details from a specific session ID."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "The unique session ID (e.g., 'sess_...') to load the transcript for.",
                },
            },
            "required": ["session_id"],
        },
    },
}

CHAT_TOOLS: List[Dict[str, Any]] = [
    SEARCH_PAST_CONVERSATIONS_TOOL,
    LOAD_SESSION_TRANSCRIPT_TOOL,
]


class ToolDispatcher:
    """Dispatches tool calls to Finch.ai storage and search engines."""

    def __init__(self, assistant: "AIAssistant"):
        self.assistant = assistant

    async def execute(self, tool_name: str, arguments: Dict[str, Any] | str) -> str:
        """Execute a tool call and return formatted context string."""
        if isinstance(arguments, str):
            try:
                args = json.loads(arguments)
            except Exception:
                args = {"query": arguments}
        else:
            args = arguments or {}

        if tool_name == "search_past_conversations":
            query = str(args.get("query", "")).strip()
            limit = int(args.get("limit", 5))
            if not query:
                return "Error: Search query cannot be empty."

            results = await self.assistant.search_hybrid(query=query, limit=limit)
            if not results:
                return f"No past conversations found matching '{query}'."

            lines = [f"Found {len(results)} relevant past conversation sessions:"]
            for r in results:
                lines.append(f"\n- Session ID: {r.session_id}")
                lines.append(f"  Title: {r.title}")
                if r.summary:
                    lines.append(f"  Summary: {r.summary}")
                if r.matched_snippets:
                    clean_snips = [
                        s.replace("[MATCH]", "").replace("[/MATCH]", "").strip()
                        for s in r.matched_snippets[:2]
                    ]
                    lines.append(f"  Snippet: {' | '.join(clean_snips)}")
                if r.cosine_similarity is not None:
                    lines.append(f"  Relevance: RRF={r.rrf_score:.4f}, Cosine Similarity={r.cosine_similarity*100:.1f}%")

            lines.append("\nTip: To read full dialogue turns from any session above, call load_session_transcript(session_id).")
            return "\n".join(lines)

        elif tool_name == "load_session_transcript":
            session_id = str(args.get("session_id", "")).strip()
            if not session_id:
                return "Error: session_id is required."

            transcript = self.assistant.load_session_transcript(session_id=session_id)
            if not transcript:
                return f"Session '{session_id}' not found in database."

            return transcript.formatted_transcript

        else:
            return f"Error: Tool '{tool_name}' is not recognized."
