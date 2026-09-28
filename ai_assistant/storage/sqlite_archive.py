"""SQLite Persistent Archive for Conversations and Session Logging."""
import json
import os
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .compression import ColumnCompressor, default_column_compressor


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

    def __init__(
        self,
        db_path: str = "conversations.db",
        embedding_dim: int = 768,
        compress_large_messages: bool = False,
        compression_threshold_bytes: int = 1024,
        compressor: Optional[ColumnCompressor] = None,
    ):
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self.compress_large_messages = compress_large_messages
        self.compression_threshold_bytes = compression_threshold_bytes
        self.compressor = compressor or default_column_compressor
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

    def update_session_mode(self, session_id: str, mode: str) -> bool:
        """Update the mode for a session."""
        now = utc_now_iso()
        with self._lock:
            conn = self._get_connection()
            with conn:
                cursor = conn.execute(
                    """
                    UPDATE sessions
                    SET mode = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (mode, now, session_id),
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

    def wipe_session_messages(self, session_id: str) -> int:
        """Delete all messages for a specific session without deleting the session itself."""
        with self._lock:
            conn = self._get_connection()
            with conn:
                cursor = conn.execute(
                    "DELETE FROM messages WHERE session_id = ?",
                    (session_id,),
                )
                deleted_count = cursor.rowcount
                conn.execute(
                    "UPDATE sessions SET summary = NULL, updated_at = ? WHERE id = ?",
                    (utc_now_iso(), session_id),
                )
                if getattr(self, "_vec_enabled", False):
                    try:
                        conn.execute("DELETE FROM sessions_vec WHERE session_id = ?", (session_id,))
                    except Exception:
                        pass
                return deleted_count

    def wipe_all_conversations(self) -> Dict[str, int]:
        """Wipe all sessions, messages, full-text index entries, and vector embeddings."""
        with self._lock:
            conn = self._get_connection()
            with conn:
                c1 = conn.execute("SELECT COUNT(*) FROM sessions;").fetchone()[0]
                c2 = conn.execute("SELECT COUNT(*) FROM messages;").fetchone()[0]

                conn.execute("DELETE FROM messages;")
                conn.execute("DELETE FROM sessions;")
                try:
                    conn.execute("DELETE FROM messages_fts;")
                except Exception:
                    pass
                if getattr(self, "_vec_enabled", False):
                    try:
                        conn.execute("DELETE FROM sessions_vec;")
                    except Exception:
                        pass

                return {"sessions_wiped": c1, "messages_wiped": c2}

    def export_all_conversations(self) -> List[Dict[str, Any]]:
        """Export all sessions and their messages as a list of dictionaries."""
        with self._lock:
            sessions = self.list_sessions(limit=100000)
            exported = []
            for s in sessions:
                messages = self.get_messages(s.id, include_system=True)
                exported.append({
                    "session": {
                        "id": s.id,
                        "title": s.title,
                        "summary": s.summary,
                        "model": s.model,
                        "mode": s.mode,
                        "created_at": s.created_at,
                        "updated_at": s.updated_at,
                        "metadata": s.metadata,
                    },
                    "messages": [
                        {
                            "id": m.id,
                            "session_id": m.session_id,
                            "role": m.role,
                            "content": m.content,
                            "thinking": m.thinking,
                            "tokens": m.tokens,
                            "timestamp": m.timestamp,
                            "metadata": m.metadata,
                        }
                        for m in messages
                    ],
                })
            return exported

    def import_conversations(self, data: List[Dict[str, Any]], mode: str = "merge") -> Dict[str, int]:
        """Import sessions and messages from exported dictionaries."""
        with self._lock:
            conn = self._get_connection()
            sessions_imported = 0
            messages_imported = 0

            with conn:
                if mode == "replace":
                    self.wipe_all_conversations()

                for entry in data:
                    s_data = entry.get("session", {})
                    s_id = s_data.get("id")
                    if not s_id:
                        continue

                    existing = self.get_session(s_id)
                    if not existing:
                        conn.execute(
                            """
                            INSERT OR IGNORE INTO sessions (id, title, summary, model, mode, created_at, updated_at, metadata)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                s_id,
                                s_data.get("title", "Imported Session"),
                                s_data.get("summary"),
                                s_data.get("model", "default"),
                                s_data.get("mode", "chat"),
                                s_data.get("created_at", utc_now_iso()),
                                s_data.get("updated_at", utc_now_iso()),
                                json.dumps(s_data.get("metadata", {})),
                            ),
                        )
                        sessions_imported += 1

                    for m in entry.get("messages", []):
                        if mode == "merge":
                            cursor = conn.execute(
                                "SELECT id FROM messages WHERE session_id = ? AND role = ? AND timestamp = ? LIMIT 1;",
                                (s_id, m.get("role"), m.get("timestamp")),
                            )
                            if cursor.fetchone():
                                continue

                        self.add_message(
                            session_id=s_id,
                            role=m.get("role", "user"),
                            content=m.get("content", ""),
                            thinking=m.get("thinking"),
                            tokens=m.get("tokens"),
                            timestamp=m.get("timestamp"),
                            metadata=m.get("metadata"),
                        )
                        messages_imported += 1

            return {"sessions_imported": sessions_imported, "messages_imported": messages_imported}

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
        content_to_store = content
        effective_meta = dict(metadata or {})

        is_compressed = False
        if self.compress_large_messages and len(content.encode("utf-8")) >= self.compression_threshold_bytes:
            comp_content, was_comp = self.compressor.compress_text(content)
            if was_comp:
                content_to_store = comp_content
                is_compressed = True
                effective_meta["compressed"] = True
                effective_meta["algo"] = self.compressor.algorithm
                effective_meta["raw_chars"] = len(content)

        meta_json = json.dumps(effective_meta)

        with self._lock:
            conn = self._get_connection()
            with conn:
                cursor = conn.execute(
                    """
                    INSERT INTO messages (session_id, role, content, thinking, tokens, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (session_id, role, content_to_store, thinking, tokens, now, meta_json),
                )
                msg_id = cursor.lastrowid

                # If stored in compressed form, update FTS index with uncompressed plain text
                # so full-text lexical search continues to index keywords transparently.
                if is_compressed:
                    try:
                        conn.execute(
                            "INSERT INTO messages_fts(messages_fts, rowid, content, role, session_id) VALUES('delete', ?, ?, ?, ?);",
                            (msg_id, content_to_store, role, session_id),
                        )
                        conn.execute(
                            "INSERT INTO messages_fts(rowid, content, role, session_id) VALUES (?, ?, ?, ?);",
                            (msg_id, content, role, session_id),
                        )
                    except Exception:
                        pass

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
                decomp_content = self.compressor.decompress_text(row["content"])
                messages.append(
                    MessageRecord(
                        id=row["id"],
                        session_id=row["session_id"],
                        role=row["role"],
                        content=decomp_content,
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
                decomp_content = self.compressor.decompress_text(r["content"])
                snippet_text = r["snippet"]
                if (
                    self.compressor.is_compressed(r["content"])
                    or self.compressor.is_compressed(snippet_text)
                    or "ZSTD" in snippet_text
                    or "ZLIB" in snippet_text
                ):
                    clean_preview = decomp_content.replace("\n", " ").strip()
                    q_lower = cleaned.lower()
                    p_lower = clean_preview.lower()
                    idx = p_lower.find(q_lower)
                    if idx >= 0:
                        start_idx = max(0, idx - 40)
                        end_idx = min(len(clean_preview), idx + len(cleaned) + 40)
                        prefix = "..." if start_idx > 0 else ""
                        suffix = "..." if end_idx < len(clean_preview) else ""
                        snippet_text = f"{prefix}{clean_preview[start_idx:end_idx]}{suffix}"
                    else:
                        snippet_text = clean_preview[:120] + "..." if len(clean_preview) > 120 else clean_preview
                results.append(
                    {
                        "message_id": r["id"],
                        "session_id": r["session_id"],
                        "session_title": r["session_title"] or "Untitled Session",
                        "role": r["role"],
                        "content": decomp_content,
                        "thinking": r["thinking"],
                        "tokens": r["tokens"],
                        "timestamp": r["timestamp"],
                        "snippet": snippet_text,
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

    # =========================================================================
    # Phase 8: SQLite Housekeeping & Archival Maintenance Routines
    # =========================================================================

    def checkpoint(self) -> Dict[str, Any]:
        """Checkpoint WAL pages into the database file and truncate WAL."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            row = cursor.fetchone()
            return {
                "busy": row[0] if row else 0,
                "log": row[1] if row else 0,
                "checkpointed": row[2] if row else 0,
            }

    def optimize(self) -> None:
        """Run PRAGMA optimize to update SQLite query planner index statistics."""
        with self._lock:
            conn = self._get_connection()
            conn.execute("PRAGMA optimize;")

    def integrity_check(self) -> str:
        """Run PRAGMA integrity_check to verify database file health."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("PRAGMA integrity_check;")
            row = cursor.fetchone()
            return row[0] if row else "ok"

    def get_storage_stats(self) -> Dict[str, Any]:
        """Collect disk usage, page counts, WAL size, and record counts."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("PRAGMA page_size;")
            page_size = cursor.fetchone()[0]

            cursor.execute("PRAGMA page_count;")
            page_count = cursor.fetchone()[0]

            cursor.execute("PRAGMA freelist_count;")
            freelist_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM sessions;")
            session_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM messages;")
            message_count = cursor.fetchone()[0]

            vec_count = 0
            if getattr(self, "_vec_enabled", False):
                try:
                    cursor.execute("SELECT COUNT(*) FROM sessions_vec;")
                    vec_count = cursor.fetchone()[0]
                except Exception:
                    pass

            file_size = 0
            wal_size = 0
            if self.db_path != ":memory:" and os.path.exists(self.db_path):
                file_size = os.path.getsize(self.db_path)
                wal_path = f"{self.db_path}-wal"
                if os.path.exists(wal_path):
                    wal_size = os.path.getsize(wal_path)

            return {
                "db_path": self.db_path,
                "file_size_bytes": file_size,
                "wal_size_bytes": wal_size,
                "total_size_bytes": file_size + wal_size,
                "page_size": page_size,
                "page_count": page_count,
                "freelist_count": freelist_count,
                "reclaimable_bytes": freelist_count * page_size,
                "session_count": session_count,
                "message_count": message_count,
                "vec_entry_count": vec_count,
            }

    def vacuum(self) -> Dict[str, Any]:
        """Execute SQLite VACUUM to defragment B-trees and reclaim disk space."""
        with self._lock:
            stats_before = self.get_storage_stats()
            conn = self._get_connection()
            # VACUUM cannot be run inside an open transaction
            conn.commit()
            conn.execute("VACUUM;")
            stats_after = self.get_storage_stats()

            reclaimed = max(0, stats_before["file_size_bytes"] - stats_after["file_size_bytes"])
            return {
                "bytes_before": stats_before["file_size_bytes"],
                "bytes_after": stats_after["file_size_bytes"],
                "bytes_reclaimed": reclaimed,
                "pages_before": stats_before["page_count"],
                "pages_after": stats_after["page_count"],
                "freelist_before": stats_before["freelist_count"],
                "freelist_after": stats_after["freelist_count"],
            }

    def run_maintenance(self, vacuum: bool = True) -> Dict[str, Any]:
        """Perform end-to-end database housekeeping: WAL checkpoint, integrity check, PRAGMA optimize, and VACUUM."""
        with self._lock:
            stats_before = self.get_storage_stats()
            ckpt = self.checkpoint()
            integrity = self.integrity_check()
            self.optimize()
            vac_res = None
            if vacuum:
                vac_res = self.vacuum()

            stats_after = self.get_storage_stats()
            bytes_saved = max(0, stats_before["total_size_bytes"] - stats_after["total_size_bytes"])

            return {
                "status": "success" if integrity == "ok" else "warning",
                "integrity": integrity,
                "checkpoint": ckpt,
                "vacuum": vac_res,
                "total_bytes_before": stats_before["total_size_bytes"],
                "total_bytes_after": stats_after["total_size_bytes"],
                "bytes_reclaimed": bytes_saved,
                "storage_stats": stats_after,
            }

    def archive_sessions(
        self,
        older_than_days: Optional[int] = None,
        session_ids: Optional[List[str]] = None,
        archive_db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Move selected inactive or older sessions and their messages into an archive database, then vacuum.
        
        Args:
            older_than_days: Archive sessions created more than N days ago.
            session_ids: Explicit list of session IDs to archive.
            archive_db_path: Destination SQLite database path (default: <db_name>_archive.db).
        """
        with self._lock:
            stats_before = self.get_storage_stats()
            all_sessions = self.list_sessions(limit=10000)
            target_sessions: List[SessionRecord] = []

            now_dt = datetime.now(timezone.utc)
            for s in all_sessions:
                should_archive = False
                if session_ids and s.id in session_ids:
                    should_archive = True
                elif older_than_days is not None:
                    try:
                        s_dt = datetime.fromisoformat(s.created_at)
                        age_days = (now_dt - s_dt).total_seconds() / 86400.0
                        if age_days >= older_than_days:
                            should_archive = True
                    except Exception:
                        pass

                if should_archive:
                    target_sessions.append(s)

            if not target_sessions:
                return {
                    "archived_sessions": 0,
                    "archived_messages": 0,
                    "archive_db_path": archive_db_path,
                    "bytes_reclaimed": 0,
                }

            # Determine archive database destination
            if not archive_db_path:
                if self.db_path == ":memory:":
                    archive_db_path = ":memory:"
                elif self.db_path.endswith(".db"):
                    archive_db_path = self.db_path[:-3] + "_archive.db"
                else:
                    archive_db_path = self.db_path + "_archive"

            # Connect to archive destination
            dest_archive = SQLiteArchive(
                db_path=archive_db_path,
                embedding_dim=self.embedding_dim,
                compress_large_messages=self.compress_large_messages,
            )

            total_messages_archived = 0

            try:
                for s in target_sessions:
                    # 1. Copy session record
                    dest_archive.create_session(
                        session_id=s.id,
                        title=s.title,
                        model=s.model,
                        mode=s.mode,
                        metadata=s.metadata,
                    )
                    if s.summary:
                        dest_archive.update_session_summary(
                            session_id=s.id,
                            title=s.title,
                            summary=s.summary,
                            metadata=s.metadata,
                        )

                    # 2. Copy all messages
                    msgs = self.get_messages(s.id, include_system=True)
                    for m in msgs:
                        dest_archive.add_message(
                            session_id=s.id,
                            role=m.role,
                            content=m.content,
                            thinking=m.thinking,
                            tokens=m.tokens,
                            timestamp=m.timestamp,
                            metadata=m.metadata,
                        )
                        total_messages_archived += 1

                    # 3. Purge session from main database (cascades messages and vectors)
                    self.delete_session(s.id)

            finally:
                dest_archive.close()

            # Run housekeeping on main DB to reclaim space
            self.checkpoint()
            vac_res = self.vacuum()
            stats_after = self.get_storage_stats()

            return {
                "archived_sessions": len(target_sessions),
                "archived_messages": total_messages_archived,
                "archive_db_path": archive_db_path,
                "bytes_reclaimed": vac_res["bytes_reclaimed"],
                "main_db_size_before": stats_before["total_size_bytes"],
                "main_db_size_after": stats_after["total_size_bytes"],
            }

    def close(self) -> None:
        """Close database connection cleanly after running PRAGMA optimize and checkpoint."""
        with self._lock:
            if self._conn is not None:
                try:
                    self.optimize()
                    self.checkpoint()
                except Exception:
                    pass
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

