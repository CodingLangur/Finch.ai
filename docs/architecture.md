# Finch.ai Architecture & Deep-Dive Technical Specifications

This document contains the complete technical specifications, developmental phase logs (Phases 3 through 8), database schemas, SQLite trigger definitions, and subsystem mechanics for **Finch.ai**.

---

## Table of Contents

1. [Architectural Overview](#1-architectural-overview)
2. [Technical Phase Logs (Phase 3 – Phase 8)](#2-technical-phase-logs-phase-3--phase-8)
   - [Phase 3: SQLite Conversation Archive & Session Logging](#phase-3-sqlite-conversation-archive--session-logging)
   - [Phase 4: Full-Text Search (FTS5) & Lexical Retrieval](#phase-4-full-text-search-fts5--lexical-retrieval)
   - [Phase 5: Semantic & Hybrid Search (sqlite-vec + RRF)](#phase-5-semantic--hybrid-search-sqlite-vec--rrf)
   - [Phase 6: Function Calling in Chat Mode & Single-Hop Retrieval](#phase-6-function-calling-in-chat-mode--single-hop-retrieval)
   - [Phase 7: Chat vs. Agent Mode & 3-State Tool Guardrails](#phase-7-chat-vs-agent-mode--3-state-tool-guardrails)
   - [Phase 8: Optimization, Housekeeping & zstd Compression](#phase-8-optimization-housekeeping--zstd-compression)
3. [Production SQLite Schemas & Triggers](#3-production-sqlite-schemas--triggers)
   - [Connection Pragmas](#connection-pragmas)
   - [Relational Tables (sessions, messages, user_facts)](#relational-tables)
   - [Indexes](#indexes)
   - [FTS5 Full-Text Virtual Table & Synchronization Triggers](#fts5-virtual-table--synchronization-triggers)
   - [sqlite-vec Vector Table](#sqlite-vec-vector-table)
4. [Model Context Protocol (FastMCP) Subsystem](#4-model-context-protocol-fastmcp-subsystem)
5. [Storage Footprint & Maintenance Economics](#5-storage-footprint--maintenance-economics)

---

## 1. Architectural Overview

Finch.ai is built on a modular, decoupled architecture organized around an asynchronous central orchestrator (`AIAssistant`). It serves dual roles:
1. **Interactive Local CLI Harness**: Terminal interface using `prompt_toolkit` and `rich` for low-latency streaming chat and agent operations.
2. **Model Context Protocol (MCP) Server**: Headless FastMCP service communicating over STDIO to provide external runtimes (Claude Desktop, Hermes Agent, Cursor) with context compression, hybrid memory search, and subagent delegation.

```
+-------------------------------------------------------------------------+
|                                Clients                                  |
|  [Interactive CLI]   [Claude Desktop]   [Hermes Agent]   [IDE / Cursor] |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                           Transport & Ingress                           |
|       Terminal Loop (prompt_toolkit)  /  FastMCP Server (STDIO)         |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                        Core Orchestrator Engine                         |
|   - AIAssistant State Coordinator                                       |
|   - Headroom Context Compression Pipeline (15%–60% token savings)       |
|   - 3-State Guardrail Dispatcher (OFF | ASK | AUTO)                     |
|   - Autonomous Subagent Execution Loop (max_turns bound)                |
+-------------------------------------------------------------------------+
             |                                    |
             v                                    v
+---------------------------+       +-------------------------------------+
|    Inference Backends     |       |          Storage & Memory           |
| - Local Ollama            |       | - SQLite (WAL mode, foreign keys)   |
| - Google Gemini REST/SSE  |       | - FTS5 BM25 Lexical Index           |
| - OpenAI-Compatible APIs  |       | - sqlite-vec 768-dim Vector Store   |
|                           |       | - Reciprocal Rank Fusion (RRF)      |
|                           |       | - Level-3 zstd Message Compression  |
|                           |       | - Persistent user_facts Store       |
+---------------------------+       +-------------------------------------+
```

---

## 2. Technical Phase Logs (Phase 3 – Phase 8)

### Phase 3: SQLite Conversation Archive & Session Logging

- **Objective**: Implement durable, zero-overhead relational persistence for sessions, user turns, assistant responses, model telemetry, and thinking traces.
- **Implementation**:
  - Provisioned relational SQLite database `conversations.db` with Write-Ahead Logging (`PRAGMA journal_mode=WAL`) and `PRAGMA synchronous=NORMAL` for concurrent reader/writer safety.
  - Engineered the `sessions` table capturing unique session IDs, titles, summaries, active model, execution mode (`chat` vs `agent`), creation timestamps, and JSON metadata.
  - Engineered the `messages` table capturing turn role, text content, thinking/reasoning blocks, token counts, timestamps, and tool execution traces.
  - Enforced foreign-key cascades (`ON DELETE CASCADE`) to eliminate orphaned message turns.
  - Implemented background session summarization on shutdown or via `/summarize`: runs a fast background LLM prompt to generate a 2-sentence summary and 5-word title saved directly into the session record.

### Phase 4: Full-Text Search (FTS5) & Lexical Retrieval

- **Objective**: Deliver sub-millisecond lexical retrieval for exact code tokens, error messages, filenames, and historical dialogue turns.
- **Implementation**:
  - Provisioned an external-content SQLite FTS5 virtual table (`messages_fts`) mapping to the underlying `messages` table.
  - Built automatic SQLite write-triggers (`messages_ai`, `messages_ad`, `messages_au`) ensuring ACID synchronization between normal tables and the FTS5 index.
  - Added query sanitization routines removing malformed SQLite FTS5 boolean syntax characters while preserving phrase quotes and prefixes.
  - Implemented BM25 relevance ranking with highlighted match snippet generation (`highlight(messages_fts, 0, '...', '...')`).
  - Added dialogue restoration engine (`load_session_transcript`) retrieving full sequential turns for any identified session.

### Phase 5: Semantic & Hybrid Search (sqlite-vec + RRF)

- **Objective**: Marry lexical exact-match search with dense vector semantic search to handle conceptual queries lacking exact keyword overlap.
- **Implementation**:
  - Integrated local Ollama embedding pipeline using `nomic-embed-text` generating 768-dimensional normalized floating-point vectors.
  - Indexed session summary vectors using the `sqlite-vec` virtual table (`sessions_vec` using `vec0`) configured with cosine distance metric.
  - Implemented **Reciprocal Rank Fusion (RRF)** uniting FTS5 BM25 ranks ($r_{\text{lex}}$) and vector distance ranks ($r_{\text{vec}}$):
    $$RRF(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
    Where $k = 60$, dynamically ranking and interleaving top-$N$ conversation results.
  - Created automated retrieval benchmarks comparing thematic vs. lexical queries (`python -m benchmarks.benchmark_search`).

### Phase 6: Function Calling in Chat Mode & Single-Hop Retrieval

- **Objective**: Grant the LLM native access to past conversational history during active chat without compromising latency or causing multi-hop loops.
- **Implementation**:
  - Formatted JSON Schema tool declarations for `search_past_conversations` (hybrid semantic + lexical query) and `load_session_transcript` (turn-by-turn dialogue fetch).
  - Enforced strict **Single-Hop Tool Execution**:
    $$\text{User Prompt} \longrightarrow \text{Model Tool Call} \longrightarrow \text{DB Fetch} \longrightarrow \text{Model Synthesis}$$
    Second synthesis pass is invoked with `tools=None`, prohibiting iterative tool loops during conversational chat.
  - Implemented terminal streaming notices notifying the user when memory archives are referenced.
  - Verified routing accuracy: queries unrelated to history bypass tool execution completely, preserving sub-200ms initial response time.

### Phase 7: Chat vs. Agent Mode & 3-State Tool Guardrails

- **Objective**: Enable full autonomous multi-turn agency with tool execution while providing deterministic human-in-the-loop safety.
- **Implementation**:
  - Implemented two-state runtime execution engine:
    - **Chat Mode**: Restricted exclusively to conversation memory retrieval with single-hop execution.
    - **Agent Mode**: Multi-turn autonomous loop executing sequential tool calls until task completion.
  - Integrated full tool inventory:
    - `run_terminal_command`: Local shell execution with directory isolation, configurable timeouts, and stdout/stderr capture.
    - `python_interpreter`: Sandboxed Python subshell for mathematical calculations, data transformation, and code evaluation.
    - `web_search` & `fetch_web_page`: Live search querying and HTML/markdown content scraping.
    - `read_file`, `write_file`, `list_directory`: Local filesystem operations.
  - Upgraded tool toggles to a **3-State Permission Guardrail Model**:
    - **`OFF`**: Capability completely disabled; omitted from LLM prompt schemas; rejected if invoked.
    - **`ASK`**: Human-in-the-loop interactive confirmation prompt (`[y/N]`) before execution.
    - **`AUTO`**: Autonomous execution with prominent safety banners and activity logging.
  - Enforced step limits (`AGENT_MAX_TURNS`) and real-time execution feedback.

### Phase 8: Optimization, Housekeeping & zstd Compression

- **Objective**: Prevent database bloat, optimize search indexes, and minimize disk footprint over thousands of sessions.
- **Implementation**:
  - **Column-Level Zstandard (`zstd`) Compression**:
    - Transparent compression (level 3) with zlib fallback for all archived message payloads.
    - Achieves **85%–92%** compression ratio on code, tool logs, and system prompts.
    - Decompression latency measured at $<0.1\text{ ms}$, ensuring zero perceptual lag.
  - **SQLite Housekeeping Procedures**:
    - `VACUUM`: Defragments database B-trees and reclaims unallocated freelist pages.
    - `PRAGMA optimize`: Generates query planner statistics for updated indexes.
    - `PRAGMA wal_checkpoint(TRUNCATE)`: Flushes uncommitted WAL transactions and truncates log files.
    - `PRAGMA integrity_check`: Validates file structures and indexes against corruption.
  - **Cold-Storage Archival**:
    - Routine to migrate sessions older than $N$ days into `conversations_archive.db`.
  - **Storage Telemetry**:
    - On-demand inspection via `/storage` reporting database size, WAL size, page count, and compression statistics.

---

## 3. Production SQLite Schemas & Triggers

### Connection Pragmas

Applied on every database connection handle before executing operations:

```sql
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;
PRAGMA busy_timeout = 5000;
```

---

### Relational Tables

#### 1. `sessions` Table
Stores conversation sessions, model settings, and background-generated summaries:

```sql
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
```

#### 2. `messages` Table
Stores chronological conversational turns, reasoning traces, token telemetry, and tool metadata:

```sql
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
```

#### 3. `user_facts` Table
Stores verified user preferences, technical stack constraints, and long-term project facts independent of `personality.md`:

```sql
CREATE TABLE IF NOT EXISTS user_facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fact TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'general',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

---

### Indexes

Standard B-Tree indexes for fast session lookups, time-series sorting, and category filtering:

```sql
CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);
CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON sessions(created_at);
CREATE INDEX IF NOT EXISTS idx_user_facts_category ON user_facts(category);
```

---

### FTS5 Virtual Table & Synchronization Triggers

#### Virtual Table Definition
Uses an external-content table configured against the `messages` table:

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content,
    role,
    session_id UNINDEXED,
    content='messages',
    content_rowid='id'
);
```

#### Synchronization Triggers
Ensures automatic, atomic updates to the FTS5 index upon `INSERT`, `DELETE`, or `UPDATE` on the `messages` table:

```sql
-- Trigger: Synchronize on INSERT
CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content, role, session_id)
    VALUES (new.id, new.content, new.role, new.session_id);
END;

-- Trigger: Synchronize on DELETE
CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content, role, session_id)
    VALUES('delete', old.id, old.content, old.role, old.session_id);
END;

-- Trigger: Synchronize on UPDATE
CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content, role, session_id)
    VALUES('delete', old.id, old.content, old.role, old.session_id);
    INSERT INTO messages_fts(rowid, content, role, session_id)
    VALUES (new.id, new.content, new.role, new.session_id);
END;
```

---

### sqlite-vec Vector Table

Virtual table for storing 768-dimensional normalized dense vectors of session summaries:

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS sessions_vec USING vec0(
    session_id TEXT,
    summary_embedding FLOAT[768] distance_metric=cosine
);
```

---

## 4. Model Context Protocol (FastMCP) Subsystem

Finch implements the official **Model Context Protocol (MCP)** via `FastMCP` over STDIO transport (`finch.mcp_server`).

### Exposed Tools & Specifications

| Tool Name | Parameters | Return Type | Description |
| :--- | :--- | :--- | :--- |
| `finch_run_subagent` | `instruction: str`<br/>`max_turns: int = 8`<br/>`model: Optional[str] = None` | `str` | Executes an autonomous multi-turn agent loop with shell, python, and file capabilities. Logs to SQLite archive. *(Alias: `run_subagent`)* |
| `finch_remember_fact` | `fact: str`<br/>`category: str = "general"` | `str` | Commits verified user fact directly to the persistent `user_facts` relational table. |
| `finch_get_user_facts` | `category: Optional[str] = None` | `str` | Retrieves formatted list of user facts and preferences. |
| `finch_search_history` | `query: str`<br/>`limit: int = 5` | `str` | Executes dual-stage FTS5 + `sqlite-vec` RRF hybrid search over historical conversations. |
| `finch_get_transcript` | `session_id: str` | `str` | Decompresses Level-3 zstd payloads and returns complete turn-by-turn dialogue transcript. |
| `compress_context` | `messages: list`<br/>`model: str = "llama3.2"` | `dict` | Runs Headroom compression pipeline on message payloads, returning compressed text and telemetry. |
| `run_finch_query` | `prompt: str`<br/>`mode: str = "chat"` | `str` | Dispatches single query through Finch core assistant engine. |
| `get_storage_stats` | *None* | `dict` | Reports SQLite file size, WAL size, page count, and compression statistics. |

---

## 5. Storage Footprint & Maintenance Economics

### Compression Comparison Benchmark

| Data Payload | Raw Size | Level-3 zstd Size | Savings | Decompression Latency |
| :--- | :--- | :--- | :--- | :--- |
| Agent Execution Logs | 142 KB | 13.8 KB | **90.3%** | 0.04 ms |
| Python Stacktraces | 48 KB | 4.9 KB | **89.8%** | 0.02 ms |
| Conversational Turns | 22 KB | 3.4 KB | **84.5%** | 0.01 ms |

### Automated Housekeeping Lifecycle

1. **On Session Exit**:
   - Truncates WAL log via `PRAGMA wal_checkpoint(TRUNCATE)`.
   - Runs `PRAGMA optimize` to refresh query plan statistics.
2. **On Explicit `/vacuum`**:
   - Executes full database compaction and B-Tree defragmentation.
   - Cleans freelist pages and reclaims disk space.
3. **On Archival (`/archive [days]`)**:
   - Copies older sessions and messages to `conversations_archive.db`.
   - Purges copied rows and executes `VACUUM`.
