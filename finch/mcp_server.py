"""Finch FastMCP Service.

Exposes Finch's memory (facts & persona), retrieval (hybrid RRF search),
context compression (Headroom), and assistant query capabilities over
the Model Context Protocol (MCP) using an STDIO-based transport.

Operational logging is strictly routed to finch_mcp.log and stderr so that
stdout remains clean for JSON-RPC frames.
"""
import asyncio
import json
import logging
import os
import sys
import warnings
from typing import Any, Dict, List, Optional, Union

# Ensure operational logs go exclusively to finch_mcp.log and stderr (NEVER stdout)
LOG_FILE = os.getenv("FINCH_MCP_LOG_FILE", "finch_mcp.log")
logging.basicConfig(
    level=getattr(logging, os.getenv("FINCH_MCP_LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stderr),
    ],
)
logger = logging.getLogger("finch.mcp")

# Suppress third-party warnings and library chatter
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

from fastmcp import FastMCP

from finch.config import AppConfig
from finch.storage import SQLiteArchive
from finch.memory import FactsManager, PersonaManager
from finch.retrieval import search_hybrid, load_session_transcript, get_embedder
from finch.compression import ContextCompressionPipeline
from finch.core.assistant import AIAssistant, AssistantMode


# Initialize FastMCP Server instance
mcp = FastMCP(
    "Finch",
    instructions=(
        "Finch.ai FastMCP Service providing persistent long-term memory, "
        "hybrid FTS5+vector search with Reciprocal Rank Fusion, "
        "Headroom context compression, and AI assistant query capabilities."
    ),
)

# Shared service components (initialized lazily or per-session)
_config: Optional[AppConfig] = None
_archive: Optional[SQLiteArchive] = None
_facts_manager: Optional[FactsManager] = None
_persona_manager: Optional[PersonaManager] = None
_compression_pipeline: Optional[ContextCompressionPipeline] = None


def get_config() -> AppConfig:
    global _config
    if _config is None:
        _config = AppConfig()
    return _config


def get_archive() -> SQLiteArchive:
    global _archive
    if _archive is None:
        cfg = get_config()
        _archive = SQLiteArchive(
            db_path=cfg.db_path,
            embedding_dim=cfg.embedding_dim,
            compress_large_messages=getattr(cfg, "compress_message_bodies", False),
            compression_threshold_bytes=getattr(cfg, "message_compression_threshold", 1024),
        )
        logger.info("SQLiteArchive initialized at %s", cfg.db_path)
    return _archive


def get_facts_manager() -> FactsManager:
    global _facts_manager
    if _facts_manager is None:
        _facts_manager = FactsManager(get_archive())
    return _facts_manager


def get_persona_manager() -> PersonaManager:
    global _persona_manager
    if _persona_manager is None:
        cfg = get_config()
        _persona_manager = PersonaManager(file_path=cfg.personality_path)
    return _persona_manager


def get_compression_pipeline() -> ContextCompressionPipeline:
    global _compression_pipeline
    if _compression_pipeline is None:
        cfg = get_config()
        _compression_pipeline = ContextCompressionPipeline(
            enabled=cfg.compression_enabled,
            compress_user_messages=cfg.compress_user_messages,
            compress_system_messages=cfg.compress_system_messages,
            protect_recent=cfg.compression_protect_recent,
            min_tokens_to_compress=cfg.compression_min_tokens,
        )
    return _compression_pipeline


# =============================================================================
# MCP Tools: Memory & Long-Term Facts (Task 2.2)
# =============================================================================

@mcp.tool()
def finch_remember_fact(fact: str, category: str = "general") -> Dict[str, Any]:
    """Inserts new verified facts directly into Finch's persistent table.
    
    Args:
        fact: Fact statement or user preference to record.
        category: Category tag for the fact (default: 'general').
    """
    logger.info("finch_remember_fact called: fact=%r, category=%r", fact, category)
    mgr = get_facts_manager()
    fact_id = mgr.add_fact(fact=fact, category=category)
    return {
        "id": fact_id,
        "fact": fact.strip(),
        "category": category.strip().lower() or "general",
        "status": "stored",
    }


@mcp.tool()
def finch_get_user_facts(category: str = "") -> List[Dict[str, Any]]:
    """Fetches persistent, verified user profile facts from user_facts.
    
    Args:
        category: Optional category filter. If empty or omitted, returns all facts.
    """
    logger.info("finch_get_user_facts called: category=%r", category)
    mgr = get_facts_manager()
    cat_filter = category.strip() if category else None
    return mgr.list_facts(category=cat_filter, limit=100)


@mcp.tool()
def remember_fact(fact: str, category: str = "general") -> str:
    """Store an enduring user fact, preference, or environment detail in Finch long-term memory.
    
    Args:
        fact: The factual statement or preference to record.
        category: Optional category tag (e.g. 'preference', 'tech_stack', 'project').
    """
    res = finch_remember_fact(fact=fact, category=category)
    return f"Fact successfully stored in long-term memory (ID: {res['id']}, Category: {res['category']})."


@mcp.tool()
def list_facts(category: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieve verified user facts and preferences stored in Finch long-term memory.
    
    Args:
        category: Optional category filter.
        limit: Maximum number of facts to return (default: 50).
    """
    logger.info("list_facts called: category=%s", category)
    mgr = get_facts_manager()
    return mgr.list_facts(category=category, limit=limit)


@mcp.tool()
def forget_fact(fact_id: int) -> str:
    """Delete a specific user fact from long-term memory by ID.
    
    Args:
        fact_id: The unique integer ID of the fact to remove.
    """
    logger.info("forget_fact called: fact_id=%d", fact_id)
    mgr = get_facts_manager()
    deleted = mgr.delete_fact(fact_id)
    if deleted:
        return f"Fact ID {fact_id} removed from long-term memory."
    return f"Fact ID {fact_id} not found."


# =============================================================================
# MCP Tools: Conversation Retrieval & Search (Task 2.2)
# =============================================================================

@mcp.tool()
async def finch_search_history(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Executes hybrid FTS5 + sqlite-vec RRF search across archived messages.
    
    Args:
        query: Search keywords, question, code snippet, or topic.
        limit: Maximum number of matching sessions to retrieve (default: 5).
    """
    logger.info("finch_search_history called: query=%r, limit=%d", query, limit)
    archive = get_archive()
    cfg = get_config()
    embedder = get_embedder(cfg)
    results = await search_hybrid(
        query=query,
        limit=limit,
        k=cfg.rrf_k,
        archive=archive,
        embedder=embedder,
    )
    return [r.to_dict() for r in results]


@mcp.tool()
def finch_get_transcript(session_id: str) -> Dict[str, Any]:
    """Decompresses zstd records and returns structured dialogue turns.
    
    Args:
        session_id: The unique session identifier (e.g. 'sess_...').
    """
    logger.info("finch_get_transcript called: session_id=%s", session_id)
    archive = get_archive()
    data = archive.get_session_transcript(session_id)
    if not data:
        return {
            "session_id": session_id,
            "found": False,
            "turns": [],
            "error": f"Session '{session_id}' not found.",
        }
    session = data["session"]
    return {
        "session_id": session.id,
        "title": session.title,
        "summary": session.summary,
        "model": session.model,
        "mode": session.mode,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "total_turns": len(data["turns"]),
        "turns": data["turns"],
        "formatted_transcript": data.get("formatted_transcript", ""),
    }


@mcp.tool()
async def search_past_conversations(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Search archived conversation history using hybrid retrieval (FTS5 lexical + sqlite-vec semantic with RRF).
    
    Args:
        query: Search keywords, question, code snippet, or topic.
        limit: Maximum number of matching sessions to retrieve (default: 5).
    """
    return await finch_search_history(query=query, limit=limit)


@mcp.tool()
def load_transcript(session_id: str) -> str:
    """Load the full turn-by-turn dialogue transcript for a past conversation session.
    
    Args:
        session_id: The unique session identifier (e.g. 'sess_...').
    """
    logger.info("load_transcript called: session_id=%s", session_id)
    archive = get_archive()
    transcript = load_session_transcript(session_id=session_id, archive=archive)
    if not transcript:
        return f"Session '{session_id}' not found."
    return transcript.formatted_transcript


# =============================================================================
# MCP Tools: Context Compression
# =============================================================================

@mcp.tool()
def compress_context(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Compress conversational message payload using Headroom, saving tokens while preserving semantic integrity.
    
    Args:
        messages: List of message dicts with 'role' and 'content' keys.
        model: Optional model name for token estimation.
    """
    logger.info("compress_context called: %d messages", len(messages))
    pipeline = get_compression_pipeline()
    compressed_msgs, stats = pipeline.compress_messages(messages=messages, model=model)
    return {
        "compressed_messages": compressed_msgs,
        "tokens_before": stats.tokens_before if stats else 0,
        "tokens_after": stats.tokens_after if stats else 0,
        "tokens_saved": stats.tokens_saved if stats else 0,
        "savings_pct": stats.savings_pct if stats else 0.0,
        "transforms_applied": stats.transforms_applied if stats else [],
    }


# =============================================================================
# MCP Tools: Assistant Execution & Storage Telemetry
# =============================================================================

@mcp.tool()
def get_storage_stats() -> Dict[str, Any]:
    """Retrieve SQLite database storage telemetry, message counts, and WAL file metrics."""
    logger.info("get_storage_stats called")
    archive = get_archive()
    return archive.get_storage_stats()


@mcp.tool()
async def run_finch_query(prompt: str, mode: str = "chat") -> Dict[str, Any]:
    """Execute a query through the Finch AI Assistant orchestrator in chat or agent mode.
    
    Args:
        prompt: User question or instruction.
        mode: Execution mode ('chat' for single-hop dialogue, 'agent' for autonomous tools).
    """
    logger.info("run_finch_query called: mode=%s", mode)
    cfg = get_config()
    archive = get_archive()
    assistant = AIAssistant(config=cfg, archive=archive)
    assistant.set_mode(mode)

    content_parts = []
    thinking_parts = []
    tools_called = []
    notices = []

    try:
        async for chunk in assistant.chat_stream(prompt):
            if chunk.delta:
                content_parts.append(chunk.delta)
            if chunk.thinking_delta:
                thinking_parts.append(chunk.thinking_delta)
            if chunk.tool_calls:
                tools_called.extend(chunk.tool_calls)
            if chunk.tool_call_notice:
                notices.append(chunk.tool_call_notice)
    finally:
        assistant.close()

    return {
        "response": "".join(content_parts).strip(),
        "thinking": "".join(thinking_parts).strip() or None,
        "tools_called": tools_called,
        "notices": notices,
        "session_id": assistant.current_session.id,
    }


@mcp.tool()
async def finch_run_subagent(
    instruction: str,
    max_turns: int = 8,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Spins up Finch's autonomous while loop headlessly in mode = "agent".
    
    Allows external orchestrators to delegate local filesystem, Python REPL,
    or shell execution tasks to Finch, applying Finch's Headroom compression
    and logging execution metadata directly to conversations.db.
    
    Args:
        instruction: Task instruction, goal, or execution request for the subagent.
        max_turns: Maximum autonomous multi-turn execution steps (default: 8).
        model: Optional model name override for this subagent execution.
    """
    logger.info("finch_run_subagent called: instruction=%r, max_turns=%d, model=%r", instruction, max_turns, model)
    cfg = get_config()
    archive = get_archive()
    assistant = AIAssistant(config=cfg, archive=archive)
    assistant.set_mode("agent")
    if model:
        assistant.set_model(model)

    # Set descriptive session title in conversations.db
    clean_instr = instruction.strip().replace("\n", " ")
    subagent_title = f"Subagent: {clean_instr[:45]}..." if len(clean_instr) > 45 else f"Subagent: {clean_instr}"
    archive.update_session_summary(assistant.current_session.id, title=subagent_title)

    # Configure autonomous headless tool permissions (auto-allow filesystem, python, terminal)
    assistant.set_tool_permission("terminal", "auto")
    assistant.set_tool_permission("python", "auto")
    assistant.set_tool_permission("files", "auto")
    if cfg.enable_web_tool:
        assistant.set_tool_permission("web", "auto")

    content_parts: List[str] = []
    thinking_parts: List[str] = []
    tools_called: List[Dict[str, Any]] = []
    notices: List[str] = []
    last_compression_stats = None

    try:
        async for chunk in assistant.chat_stream(instruction, max_turns=max_turns):
            if chunk.delta:
                content_parts.append(chunk.delta)
            if chunk.thinking_delta:
                thinking_parts.append(chunk.thinking_delta)
            if chunk.tool_calls:
                tools_called.extend(chunk.tool_calls)
            if chunk.tool_call_notice:
                notices.append(chunk.tool_call_notice)
            if chunk.compression_stats:
                last_compression_stats = chunk.compression_stats

        msgs = archive.get_messages(assistant.current_session.id)
        assistant_msg = next((m for m in reversed(msgs) if m.role == "assistant"), None)
        execution_metadata = assistant_msg.metadata if (assistant_msg and assistant_msg.metadata) else {}
    finally:
        assistant.close()

    return {
        "response": "".join(content_parts).strip(),
        "thinking": "".join(thinking_parts).strip() or None,
        "tools_called": tools_called,
        "tools_executed": execution_metadata.get("tools_executed", []),
        "agent_turns": execution_metadata.get("agent_turns", 1),
        "max_turns": max_turns,
        "notices": notices,
        "session_id": assistant.current_session.id,
        "compression": execution_metadata.get("compression") or (
            {
                "tokens_before": last_compression_stats.tokens_before,
                "tokens_after": last_compression_stats.tokens_after,
                "tokens_saved": last_compression_stats.tokens_saved,
                "savings_pct": last_compression_stats.savings_pct,
            }
            if last_compression_stats
            else None
        ),
        "execution_metadata": execution_metadata,
        "status": "completed",
    }


@mcp.tool()
async def run_subagent(
    instruction: str,
    max_turns: int = 8,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute an autonomous subagent task headlessly in agent mode (alias for finch_run_subagent).
    
    Args:
        instruction: Task instruction, goal, or execution request for the subagent.
        max_turns: Maximum autonomous multi-turn execution steps (default: 8).
        model: Optional model name override for this subagent execution.
    """
    return await finch_run_subagent(instruction=instruction, max_turns=max_turns, model=model)


# =============================================================================
# MCP Resources
# =============================================================================

@mcp.resource("finch://facts")
def resource_facts() -> str:
    """Active verified user facts and memory preferences formatted as Markdown."""
    mgr = get_facts_manager()
    return mgr.format_facts_for_prompt() or "No user facts recorded yet."


@mcp.resource("finch://persona")
def resource_persona() -> str:
    """Active Finch AI personality guidelines and operational rules."""
    mgr = get_persona_manager()
    return mgr.load_persona()


@mcp.resource("finch://storage")
def resource_storage() -> str:
    """Current database storage and telemetry metrics."""
    stats = get_storage_stats()
    return json.dumps(stats, indent=2)


# =============================================================================
# Main Entry Point
# =============================================================================

def run_server() -> None:
    """Run the FastMCP server over STDIO transport without polluting stdout."""
    logger.info("Starting Finch FastMCP server on STDIO transport...")
    try:
        mcp.run(transport="stdio", show_banner=False)
    except Exception as e:
        logger.exception("Error in Finch FastMCP server: %s", e)
        raise


if __name__ == "__main__":
    run_server()
