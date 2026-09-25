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

    def __init__(self, db_path: str = "conversations.db", embedding_dim: int = 768):
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self._lock = threading.RLock()
        self._vec_enabled: bool = False

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

            # Attempt loading sqlite-vec extension
            try:
                import sqlite_vec
                self._conn.enable_load_extension(True)
                sqlite_vec.load(self._conn)
                self._conn.enable_load_extension(False)
                self._vec_enabled = True
            except Exception:
                self._vec_enabled = False

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

                # FTS5 Virtual Table for Fast Lexical Search
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                        content,
                        role,
                        session_id UNINDEXED,
                        content='messages',
                        content_rowid='id'
                    );
                    """
                )

                # FTS5 Automatic Synchronization Triggers
                conn.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
                        INSERT INTO messages_fts(rowid, content, role, session_id)
                        VALUES (new.id, new.content, new.role, new.session_id);
                    END;
                    """
                )
                conn.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
                        INSERT INTO messages_fts(messages_fts, rowid, content, role, session_id)
                        VALUES('delete', old.id, old.content, old.role, old.session_id);
                    END;
                    """
                )
                conn.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
                        INSERT INTO messages_fts(messages_fts, rowid, content, role, session_id)
                        VALUES('delete', old.id, old.content, old.role, old.session_id);
                        INSERT INTO messages_fts(rowid, content, role, session_id)
                        VALUES (new.id, new.content, new.role, new.session_id);
                    END;
                    """
                )

                # Rebuild FTS index if messages already existed before FTS5 table was created
                try:
                    c = conn.cursor()
                    c.execute("SELECT COUNT(*) FROM messages;")
                    m_cnt = c.fetchone()[0]
                    c.execute("SELECT COUNT(*) FROM messages_fts;")
                    f_cnt = c.fetchone()[0]
                    if m_cnt > 0 and f_cnt == 0:
                        conn.execute("INSERT INTO messages_fts(messages_fts) VALUES('rebuild');")
                except Exception:
                    pass

                # sqlite-vec Virtual Table for Session Summary Semantic Embeddings
                if getattr(self, "_vec_enabled", False):
                    try:
                        conn.execute(
                            f"""
                            CREATE VIRTUAL TABLE IF NOT EXISTS sessions_vec USING vec0(
                                session_id text,
                                summary_embedding float[{self.embedding_dim}] distance_metric=cosine
                            );
                            """
                        )
                    except Exception:
                        pass

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
        """Delete session and cascade delete all its messages and vector embeddings."""
        with self._lock:
            conn = self._get_connection()
            with conn:
                if getattr(self, "_vec_enabled", False):
                    try:
                        conn.execute("DELETE FROM sessions_vec WHERE session_id = ?", (session_id,))
                    except Exception:
                        pass
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

    def rebuild_fts(self) -> None:
        """Force rebuild of the messages_fts virtual table index."""
        with self._lock:
            conn = self._get_connection()
            with conn:
                conn.execute("INSERT INTO messages_fts(messages_fts) VALUES('rebuild');")

    def search_messages(
        self,
        query: str,
        session_id: Optional[str] = None,
        limit: int = 20,
        exact_match: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Execute lexical search over archived messages using SQLite FTS5.

        Args:
            query: Exact term, phrase, code snippet, or keywords to search.
            session_id: Optional filter for a specific session.
            limit: Maximum number of matched messages to return.
            exact_match: When True, treats query as an exact phrase/term.
        """
        cleaned = query.strip()
        if not cleaned:
            return []

        # Prepare FTS5 formatted query string
        if exact_match:
            if (cleaned.startswith('"') and cleaned.endswith('"')) or (
                cleaned.startswith("'") and cleaned.endswith("'")
            ):
                cleaned = cleaned[1:-1].strip()
            escaped = cleaned.replace('"', '""')
            fts_query = f'"{escaped}"'
        else:
            tokens = [t.strip('"\'') for t in cleaned.split() if t.strip('"\'')]
            if not tokens:
                return []
            sanitized = ['"' + t.replace('"', '""') + '"' for t in tokens]
            fts_query = " AND ".join(sanitized)

        sql = """
            SELECT m.id, m.session_id, s.title as session_title, m.role, m.content,
                   m.thinking, m.tokens, m.timestamp,
                   snippet(messages_fts, 0, '[MATCH]', '[/MATCH]', '...', 15) as snippet,
                   bm25(messages_fts) as rank
            FROM messages_fts
            JOIN messages m ON m.id = messages_fts.rowid
            LEFT JOIN sessions s ON s.id = m.session_id
            WHERE messages_fts MATCH ?
        """
        params: List[Any] = [fts_query]

        if session_id:
            sql += " AND m.session_id = ?"
            params.append(session_id)

        sql += " ORDER BY rank ASC LIMIT ?"
        params.append(limit)

        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            try:
                cursor.execute(sql, tuple(params))
                rows = cursor.fetchall()
            except sqlite3.OperationalError:
                # Graceful fallback to escaped phrase
                try:
                    fallback_query = '"' + cleaned.replace('"', '""') + '"'
                    params[0] = fallback_query
                    cursor.execute(sql, tuple(params))
                    rows = cursor.fetchall()
                except Exception:
                    return []

            results = []
            for r in rows:
                results.append(
                    {
                        "message_id": r["id"],
                        "session_id": r["session_id"],
                        "session_title": r["session_title"] or "Untitled Session",
                        "role": r["role"],
                        "content": r["content"],
                        "thinking": r["thinking"],
                        "tokens": r["tokens"],
                        "timestamp": r["timestamp"],
                        "snippet": r["snippet"],
                        "rank": float(r["rank"]),
                    }
                )
            return results

    def get_session_transcript(
        self, session_id: str, include_system: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Fetch full session transcript and formatted dialogue turns for a session."""
        with self._lock:
            session = self.get_session(session_id)
            if not session:
                return None

            messages = self.get_messages(session_id, include_system=include_system)
            turns = []
            formatted_lines = [
                f"=== Session Transcript: {session.title} ({session.id}) ===",
                f"Model: {session.model} | Mode: {session.mode} | Messages: {len(messages)}",
                f"Created: {session.created_at} | Updated: {session.updated_at}",
            ]
            if session.summary:
                formatted_lines.append(f"Summary: {session.summary}")
            formatted_lines.append("=" * 60)

            for idx, msg in enumerate(messages, 1):
                turns.append(
                    {
                        "turn": idx,
                        "role": msg.role,
                        "content": msg.content,
                        "thinking": msg.thinking,
                        "tokens": msg.tokens,
                        "timestamp": msg.timestamp,
                    }
                )
                formatted_lines.append(f"\n[{msg.role.upper()} | {msg.timestamp}]:")
                formatted_lines.append(msg.content)

            return {
                "session": session,
                "turns": turns,
                "formatted_transcript": "\n".join(formatted_lines),
            }

    def has_vec_support(self) -> bool:
        """Return whether sqlite-vec extension and sessions_vec are active."""
        return getattr(self, "_vec_enabled", False)

    def store_session_embedding(self, session_id: str, embedding: List[float]) -> bool:
        """Store or replace session summary vector in sessions_vec virtual table."""
        if not getattr(self, "_vec_enabled", False):
            return False

        with self._lock:
            conn = self._get_connection()
            with conn:
                try:
                    conn.execute("DELETE FROM sessions_vec WHERE session_id = ?", (session_id,))
                    conn.execute(
                        "INSERT INTO sessions_vec(session_id, summary_embedding) VALUES (?, ?)",
                        (session_id, json.dumps(embedding)),
                    )
                    return True
                except Exception:
                    return False

    def search_sessions_semantic(
        self, query_embedding: List[float], limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        KNN semantic search over session summary embeddings using cosine distance.

        Returns list of matched session dictionaries ordered by descending cosine similarity.
        """
        if not getattr(self, "_vec_enabled", False):
            return []

        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            try:
                cursor.execute(
                    """
                    SELECT v.session_id, s.title, s.summary, s.model, s.mode,
                           s.created_at, s.updated_at, v.distance
                    FROM sessions_vec v
                    LEFT JOIN sessions s ON s.id = v.session_id
                    WHERE v.summary_embedding MATCH ?
                      AND k = ?
                    ORDER BY v.distance ASC
                    """,
                    (json.dumps(query_embedding), limit),
                )
                rows = cursor.fetchall()
                results = []
                for r in rows:
                    dist = float(r["distance"])
                    # Cosine distance: 0 is exact match, 1 is orthogonal, 2 is opposite
                    sim = max(0.0, 1.0 - dist)
                    results.append(
                        {
                            "session_id": r["session_id"],
                            "title": r["title"] or "Untitled Session",
                            "summary": r["summary"],
                            "distance": dist,
                            "similarity": round(sim, 4),
                            "model": r["model"],
                            "mode": r["mode"],
                            "created_at": r["created_at"],
                            "updated_at": r["updated_at"],
                        }
                    )
                return results
            except Exception:
                return []

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
