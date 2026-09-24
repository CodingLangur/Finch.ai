"""SQLite Persistent Archive for Conversations and Session Logging."""
import json
import os
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def utc_now_iso() -> str:
    """Return current UTC time in ISO 8601 string format."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SessionRecord:
    """Represents a chat session stored in SQLite."""
    id: str
    title: str
    summary: Optional[str]
    model: str
    mode: str
    created_at: str
    updated_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    message_count: int = 0


@dataclass
class MessageRecord:
    """Represents a chat turn message stored in SQLite."""
    id: int
    session_id: str
    role: str
    content: str
    thinking: Optional[str]
    tokens: Optional[int]
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class SQLiteArchive:
    """Relational SQLite storage manager for chat sessions and conversational history."""

    def __init__(self, db_path: str = "conversations.db"):
        self.db_path = db_path
        self._lock = threading.RLock()

        # Ensure parent directory exists if db_path is not in-memory
        if self.db_path != ":memory:":
            parent_dir = os.path.dirname(os.path.abspath(self.db_path))
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)

        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get or initialize thread-safe SQLite connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0,
            )
            self._conn.row_factory = sqlite3.Row
            with self._conn:
                self._conn.execute("PRAGMA foreign_keys = ON;")
                if self.db_path != ":memory:":
                    self._conn.execute("PRAGMA journal_mode = WAL;")
        return self._conn

    def _init_db(self) -> None:
        """Create relational tables and indexes if they do not exist."""
        with self._lock:
            conn = self._get_connection()
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sessions (
                        id TEXT PRIMARY KEY,
                        title TEXT NOT NULL DEFAULT 'New Session',
                        summary TEXT,
                        model TEXT NOT NULL,
                        mode TEXT NOT NULL DEFAULT 'chatbot',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        metadata TEXT NOT NULL DEFAULT '{}'
                    );
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        thinking TEXT,
                        tokens INTEGER,
                        timestamp TEXT NOT NULL,
                        metadata TEXT NOT NULL DEFAULT '{}',
                        FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                    );
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON sessions(created_at);"
                )

    def create_session(
        self,
        session_id: Optional[str] = None,
        title: str = "New Session",
        model: str = "default",
        mode: str = "chatbot",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SessionRecord:
        """Create a new conversation session record."""
        sid = session_id or f"sess_{uuid.uuid4().hex[:12]}"
        now = utc_now_iso()
        meta_json = json.dumps(metadata or {})

        with self._lock:
            conn = self._get_connection()
            with conn:
                conn.execute(
                    """
                    INSERT INTO sessions (id, title, summary, model, mode, created_at, updated_at, metadata)
                    VALUES (?, ?, NULL, ?, ?, ?, ?, ?)
                    """,
                    (sid, title, model, mode, now, now, meta_json),
                )

        return SessionRecord(
            id=sid,
            title=title,
            summary=None,
            model=model,
            mode=mode,
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
            message_count=0,
        )

    def get_session(self, session_id: str) -> Optional[SessionRecord]:
        """Fetch session by ID with message count."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT s.*, COUNT(m.id) as msg_count
                FROM sessions s
                LEFT JOIN messages m ON s.id = m.session_id
                WHERE s.id = ?
                GROUP BY s.id
                """,
                (session_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None

            try:
                meta = json.loads(row["metadata"])
            except Exception:
                meta = {}

            return SessionRecord(
                id=row["id"],
                title=row["title"],
                summary=row["summary"],
                model=row["model"],
                mode=row["mode"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                metadata=meta,
                message_count=row["msg_count"],
            )

    def list_sessions(self, limit: int = 50, offset: int = 0) -> List[SessionRecord]:
        """List sessions ordered by most recently updated."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT s.*, COUNT(m.id) as msg_count
                FROM sessions s
                LEFT JOIN messages m ON s.id = m.session_id
                GROUP BY s.id
                ORDER BY s.updated_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            )
            rows = cursor.fetchall()
            results: List[SessionRecord] = []
            for row in rows:
                try:
                    meta = json.loads(row["metadata"])
                except Exception:
                    meta = {}
                results.append(
                    SessionRecord(
                        id=row["id"],
                        title=row["title"],
                        summary=row["summary"],
                        model=row["model"],
                        mode=row["mode"],
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                        metadata=meta,
                        message_count=row["msg_count"],
                    )
                )
            return results

    def update_session_summary(
        self,
        session_id: str,
        title: Optional[str] = None,
        summary: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update the title, summary, and/or metadata for a session."""
        now = utc_now_iso()
        with self._lock:
            conn = self._get_connection()
            # Fetch current session to merge metadata if needed
            current = self.get_session(session_id)
            if not current:
                return False

            new_title = title if title is not None else current.title
            new_summary = summary if summary is not None else current.summary
            new_meta = current.metadata
            if metadata:
                new_meta.update(metadata)

            with conn:
                cursor = conn.execute(
                    """
                    UPDATE sessions
                    SET title = ?, summary = ?, updated_at = ?, metadata = ?
                    WHERE id = ?
                    """,
                    (new_title, new_summary, now, json.dumps(new_meta), session_id),
                )
                return cursor.rowcount > 0

    def delete_session(self, session_id: str) -> bool:
        """Delete session and cascade delete all its messages."""
        with self._lock:
            conn = self._get_connection()
            with conn:
                cursor = conn.execute(
                    "DELETE FROM sessions WHERE id = ?",
                    (session_id,),
                )
                return cursor.rowcount > 0

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        thinking: Optional[str] = None,
        tokens: Optional[int] = None,
        timestamp: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Persist a conversation turn to the messages table and touch session updated_at."""
        now = timestamp or utc_now_iso()
        meta_json = json.dumps(metadata or {})

        with self._lock:
            conn = self._get_connection()
            with conn:
                cursor = conn.execute(
                    """
                    INSERT INTO messages (session_id, role, content, thinking, tokens, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (session_id, role, content, thinking, tokens, now, meta_json),
                )
                msg_id = cursor.lastrowid

                # Update parent session updated_at timestamp
                conn.execute(
                    "UPDATE sessions SET updated_at = ? WHERE id = ?",
                    (now, session_id),
                )

                return msg_id

    def get_messages(
        self, session_id: str, include_system: bool = True
    ) -> List[MessageRecord]:
        """Fetch all messages for a session in chronological order."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            if include_system:
                cursor.execute(
                    """
                    SELECT * FROM messages
                    WHERE session_id = ?
                    ORDER BY id ASC
                    """,
                    (session_id,),
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM messages
                    WHERE session_id = ? AND role != 'system'
                    ORDER BY id ASC
                    """,
                    (session_id,),
                )
            rows = cursor.fetchall()
            messages: List[MessageRecord] = []
            for row in rows:
                try:
                    meta = json.loads(row["metadata"])
                except Exception:
                    meta = {}
                messages.append(
                    MessageRecord(
                        id=row["id"],
                        session_id=row["session_id"],
                        role=row["role"],
                        content=row["content"],
                        thinking=row["thinking"],
                        tokens=row["tokens"],
                        timestamp=row["timestamp"],
                        metadata=meta,
                    )
                )
            return messages

    def get_session_message_count(self, session_id: str) -> int:
        """Return message count for a session."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ?",
                (session_id,),
            )
            return cursor.fetchone()[0]

    def close(self) -> None:
        """Close database connection cleanly."""
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None

    def __enter__(self) -> "SQLiteArchive":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
