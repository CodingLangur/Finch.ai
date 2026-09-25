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


@dataclass
class HybridSearchResult:
    """Represents a fused session match from Lexical (FTS5) and Semantic (sqlite-vec) search."""
    session_id: str
    title: str
    summary: Optional[str]
    rrf_score: float
    lexical_rank: Optional[int] = None
    semantic_rank: Optional[int] = None
    cosine_similarity: Optional[float] = None
    matched_snippets: List[str] = field(default_factory=list)
    model: str = ""
    mode: str = ""
    updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert hybrid search result to dictionary."""
        return {
            "session_id": self.session_id,
            "title": self.title,
            "summary": self.summary,
            "rrf_score": round(self.rrf_score, 6),
            "lexical_rank": self.lexical_rank,
            "semantic_rank": self.semantic_rank,
            "cosine_similarity": self.cosine_similarity,
            "matched_snippets": self.matched_snippets,
            "model": self.model,
            "mode": self.mode,
            "updated_at": self.updated_at,
        }


async def search_hybrid(
    query: str,
    limit: int = 10,
    k: int = 60,
    lexical_weight: float = 1.0,
    semantic_weight: float = 1.0,
    archive: Optional[SQLiteArchive] = None,
    embedder: Optional[Any] = None,
    db_path: Optional[str] = None,
) -> List[HybridSearchResult]:
    """
    Execute Hybrid Search using Reciprocal Rank Fusion (RRF) combining
    FTS5 lexical retrieval and sqlite-vec semantic vector similarity.

    Args:
        query: User search text or thematic description.
        limit: Number of hybrid results to return (default: 10).
        k: Smoothing constant for RRF (default: 60).
        lexical_weight: Multiplier weight for lexical ranking.
        semantic_weight: Multiplier weight for vector similarity ranking.
        archive: Optional active SQLiteArchive instance.
        embedder: Optional BaseEmbedder instance for generating query vectors.
        db_path: Database path to use if archive is not provided.

    Returns:
        List of HybridSearchResult instances ranked by descending RRF score.
    """
    should_close = False
    if archive is None:
        target_path = db_path or "conversations.db"
        archive = SQLiteArchive(db_path=target_path)
        should_close = True

    if embedder is None:
        from ..embeddings import get_embedder
        embedder = get_embedder()

    try:
        # 1. Lexical Retrieval (FTS5) over message content
        lexical_raw = archive.search_messages(query=query, limit=50, exact_match=False)
        if not lexical_raw:
            lexical_raw = archive.search_messages(query=query, limit=50, exact_match=True)

        lexical_sessions: Dict[str, Dict[str, Any]] = {}
        for r in lexical_raw:
            sid = r["session_id"]
            if sid not in lexical_sessions:
                lexical_sessions[sid] = {
                    "rank": len(lexical_sessions) + 1,
                    "title": r["session_title"],
                    "snippets": [r["snippet"]],
                }
            else:
                if len(lexical_sessions[sid]["snippets"]) < 3:
                    lexical_sessions[sid]["snippets"].append(r["snippet"])

        # 2. Semantic Retrieval (sqlite-vec) over session summary embeddings
        semantic_sessions: Dict[str, Dict[str, Any]] = {}
        if archive.has_vec_support():
            try:
                query_vector = await embedder.embed_text(query)
                sem_raw = archive.search_sessions_semantic(query_vector, limit=limit * 3)
                for idx, r in enumerate(sem_raw, start=1):
                    sid = r["session_id"]
                    semantic_sessions[sid] = {
                        "rank": idx,
                        "title": r["title"],
                        "summary": r["summary"],
                        "similarity": r["similarity"],
                        "distance": r["distance"],
                        "model": r["model"],
                        "mode": r["mode"],
                        "updated_at": r["updated_at"],
                    }
            except Exception:
                pass

        # 3. Reciprocal Rank Fusion (RRF)
        all_session_ids = set(lexical_sessions.keys()).union(semantic_sessions.keys())
        scored: List[HybridSearchResult] = []

        for sid in all_session_ids:
            lex_info = lexical_sessions.get(sid)
            sem_info = semantic_sessions.get(sid)

            lex_rank = lex_info["rank"] if lex_info else None
            sem_rank = sem_info["rank"] if sem_info else None

            rrf = 0.0
            if lex_rank is not None:
                rrf += lexical_weight / (k + lex_rank)
            if sem_rank is not None:
                rrf += semantic_weight / (k + sem_rank)

            title = (sem_info and sem_info["title"]) or (lex_info and lex_info["title"])
            summary = sem_info["summary"] if sem_info else None
            sim = sem_info["similarity"] if sem_info else None
            snippets = lex_info["snippets"] if lex_info else []
            model = sem_info["model"] if sem_info else ""
            mode = sem_info["mode"] if sem_info else ""
            updated_at = sem_info["updated_at"] if sem_info else ""

            if not summary or not title:
                sess_rec = archive.get_session(sid)
                if sess_rec:
                    title = title or sess_rec.title
                    summary = summary or sess_rec.summary
                    model = model or sess_rec.model
                    mode = mode or sess_rec.mode
                    updated_at = updated_at or sess_rec.updated_at

            scored.append(
                HybridSearchResult(
                    session_id=sid,
                    title=title or "Untitled Session",
                    summary=summary,
                    rrf_score=rrf,
                    lexical_rank=lex_rank,
                    semantic_rank=sem_rank,
                    cosine_similarity=sim,
                    matched_snippets=snippets,
                    model=model,
                    mode=mode,
                    updated_at=updated_at,
                )
            )

        scored.sort(key=lambda x: x.rrf_score, reverse=True)
        return scored[:limit]

    finally:
        if should_close:
            archive.close()


def search_hybrid_sync(
    query: str,
    limit: int = 10,
    k: int = 60,
    lexical_weight: float = 1.0,
    semantic_weight: float = 1.0,
    archive: Optional[SQLiteArchive] = None,
    embedder: Optional[Any] = None,
    db_path: Optional[str] = None,
) -> List[HybridSearchResult]:
    """Synchronously execute search_hybrid."""
    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return executor.submit(
                asyncio.run,
                search_hybrid(
                    query=query,
                    limit=limit,
                    k=k,
                    lexical_weight=lexical_weight,
                    semantic_weight=semantic_weight,
                    archive=archive,
                    embedder=embedder,
                    db_path=db_path,
                ),
            ).result()
    else:
        return asyncio.run(
            search_hybrid(
                query=query,
                limit=limit,
                k=k,
                lexical_weight=lexical_weight,
                semantic_weight=semantic_weight,
                archive=archive,
                embedder=embedder,
                db_path=db_path,
            )
        )

