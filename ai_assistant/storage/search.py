"""Lexical Search (FTS5) and Transcript Fetching for Finch.ai."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .sqlite_archive import SQLiteArchive, SessionRecord


@dataclass
class SearchResult:
    """Represents a matched turn from lexical search."""
    message_id: int
    session_id: str
    session_title: str
    role: str
    content: str
    timestamp: str
    snippet: str
    rank: float
    thinking: Optional[str] = None
    tokens: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert search result to dictionary."""
        return {
            "message_id": self.message_id,
            "session_id": self.session_id,
            "session_title": self.session_title,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "snippet": self.snippet,
            "rank": self.rank,
            "thinking": self.thinking,
            "tokens": self.tokens,
        }


@dataclass
class TranscriptTurn:
    """Represents a single conversational turn in a session transcript."""
    turn: int
    role: str
    content: str
    timestamp: str
    thinking: Optional[str] = None
    tokens: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert transcript turn to dictionary."""
        return {
            "turn": self.turn,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "thinking": self.thinking,
            "tokens": self.tokens,
        }


@dataclass
class SessionTranscript:
    """Represents the complete transcript and metadata for a conversation session."""
    session_id: str
    title: str
    summary: Optional[str]
    model: str
    mode: str
    created_at: str
    updated_at: str
    turns: List[TranscriptTurn] = field(default_factory=list)
    formatted_transcript: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert complete session transcript to dictionary."""
        return {
            "session_id": self.session_id,
            "title": self.title,
            "summary": self.summary,
            "model": self.model,
            "mode": self.mode,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "turns": [turn.to_dict() for turn in self.turns],
            "formatted_transcript": self.formatted_transcript,
        }

    def __str__(self) -> str:
        return self.formatted_transcript


def search_keyword(
    query: str,
    session_id: Optional[str] = None,
    limit: int = 10,
    exact_match: bool = True,
    archive: Optional[SQLiteArchive] = None,
    db_path: Optional[str] = None,
) -> List[SearchResult]:
    """
    Search archived messages using SQLite FTS5 lexical matching.

    Supports exact-term queries, code snippets (e.g. 'def foo():', 'config.py'),
    specific dates (e.g. '2026-09-24'), and technical terms.

    Args:
        query: Search term, phrase, date, or code snippet.
        session_id: Optional filter to restrict matches to a specific session.
        limit: Maximum number of matches to return (default: 10).
        exact_match: If True, treats query as exact phrase/term.
                     If False, matches all tokens with AND.
        archive: Optional active SQLiteArchive instance.
        db_path: Database path to use if archive is not provided.

    Returns:
        List of SearchResult objects ordered by relevance rank.
    """
    should_close = False
    if archive is None:
        target_path = db_path or "conversations.db"
        archive = SQLiteArchive(db_path=target_path)
        should_close = True

    try:
        raw_results = archive.search_messages(
            query=query,
            session_id=session_id,
            limit=limit,
            exact_match=exact_match,
        )

        results: List[SearchResult] = []
        for r in raw_results:
            results.append(
                SearchResult(
                    message_id=r["message_id"],
                    session_id=r["session_id"],
                    session_title=r["session_title"],
                    role=r["role"],
                    content=r["content"],
                    timestamp=r["timestamp"],
                    snippet=r["snippet"],
                    rank=r["rank"],
                    thinking=r.get("thinking"),
                    tokens=r.get("tokens"),
                )
            )
        return results
    finally:
        if should_close:
            archive.close()


def load_session_transcript(
    session_id: str,
    include_system: bool = False,
    archive: Optional[SQLiteArchive] = None,
    db_path: Optional[str] = None,
) -> Optional[SessionTranscript]:
    """
    Retrieve full dialogue turns and metadata once a session is identified.

    Args:
        session_id: Target session identifier (e.g. 'sess_...').
        include_system: Whether to include system prompt turn (default: False).
        archive: Optional active SQLiteArchive instance.
        db_path: Database path to use if archive is not provided.

    Returns:
        SessionTranscript object with structured turns and formatted text,
        or None if session does not exist.
    """
    should_close = False
    if archive is None:
        target_path = db_path or "conversations.db"
        archive = SQLiteArchive(db_path=target_path)
        should_close = True

    try:
        data = archive.get_session_transcript(
            session_id=session_id,
            include_system=include_system,
        )
        if not data:
            return None

        session: SessionRecord = data["session"]
        raw_turns = data["turns"]

        turns: List[TranscriptTurn] = []
        for t in raw_turns:
            turns.append(
                TranscriptTurn(
                    turn=t["turn"],
                    role=t["role"],
                    content=t["content"],
                    timestamp=t["timestamp"],
                    thinking=t.get("thinking"),
                    tokens=t.get("tokens"),
                )
            )

        return SessionTranscript(
            session_id=session.id,
            title=session.title,
            summary=session.summary,
            model=session.model,
            mode=session.mode,
            created_at=session.created_at,
            updated_at=session.updated_at,
            turns=turns,
            formatted_transcript=data["formatted_transcript"],
        )
    finally:
        if should_close:
            archive.close()
