# Finch.ai (Chatbot & Agent Foundation)

A high-performance, low-latency AI Assistant interface designed for local Ollama and Google Gemini runtimes with **Headroom Context Compression**, dynamic persona management, SQLite conversation persistence, and modular agent architecture.

## Features

- **Local Ollama Integration**: Built with `httpx` async streaming for minimal latency.
- **Default Model**: Configured out-of-the-box to use `Gemma4-26000-ctx:latest`.
- **Headroom Context Compression Pipeline**:
  - Intercepts and compresses tool outputs, JSON structures, logs, AST code, and conversational history before sending prompts to the LLM.
  - Reduces token consumption by **15% to 60%+** while preserving critical details and semantic integrity.
  - **Persona Protection**: Pinned `personality.md` system instructions remain completely untouched and byte-exact (`compress_system_messages=False`).
  - **Recent Turn Protection**: The latest active conversational turn is preserved with full fidelity (`protect_recent=2`).
  - **Live Compression Telemetry**: Displays token counts before/after, tokens saved, percentage reduction, and applied transforms.
- **Dynamic Persona Management (`personality.md`)**:
  - **File-First Injection**: Loads identity, style, and rules from `personality.md` on startup as the pinned system prompt.
  - **Model-Driven Updates**: When asked to adapt its persona, the model emits `<personality_update>...</personality_update>` tags.
  - **Automatic Persistence & Backup**: Captures and validates the update, backs up previous version to `personality.md.bak`, and live-updates in-memory context without restarting.
- **Model Discovery & Switching**: Query installed Ollama models (`/models`) and switch on-the-fly (`/use <model>`).
- **Sliding-Window Message Buffer**: In-memory context retention (default: 8 messages / 4 turns) with pinned system prompt preservation.
- **Real-Time Telemetry & Profiling**: Live token streaming with exact per-turn stats:
  - **Generation Speed (tokens/second)**
  - **Time to First Token (TTFT)**
  - **Token Counts (Prompt & Eval)**
  - **Process Memory (RAM / RSS in MB)**
  - **Buffer Utilization (active vs max)**
  - **Headroom Savings (tokens before/after, % saved)**
- **Multi-Provider Support (Ollama & Google Gemini)**:
  - Supports local Ollama models (`Gemma4-26000-ctx:latest`) and Google Gemini API (`gemini-2.5-flash`, `gemini-2.5-pro`).
  - Configurable via `.env` (`GEMINI_API_KEY`, `DEFAULT_PROVIDER=gemini|ollama`).
  - Native SSE streaming with real-time TTFT and tokens/sec telemetry.
- **SQLite Conversation Archive & Session Logging (Phase 3)**:
  - Persistent relational storage on disk (`conversations.db`) using WAL mode and enforced foreign keys.
  - Relational schema with `sessions` and `messages` tables.
  - Automatically logs every user and assistant turn with timestamps, token counts, reasoning/thinking traces, and compression metadata.
  - **Fast Session Summarization**: Runs a fast background prompt on exit or via `/summarize` to generate a 2-sentence summary and concise title, saved directly into the database.
- **Fast Lexical Search with FTS5 & Transcript Fetching (Phase 4)**:
  - SQLite `messages_fts` virtual table synced in real-time via `AFTER INSERT`, `AFTER DELETE`, and `AFTER UPDATE` triggers.
  - Sub-millisecond exact-term matching for code snippets (e.g. `def calculate_fibonacci(n):`), dates (`2026-09-24`), technical keywords, and filenames (`config.py`).
  - Automatic query sanitization, BM25 relevance ranking, and contextual match snippet extraction.
  - Structured transcript retrieval (`load_session_transcript`) restoring full conversational dialogue turns once a session is identified.
- **Semantic & Hybrid Search with sqlite-vec + Reciprocal Rank Fusion (Phase 5)**:
  - **Local Embedder**: Generates 768-dimensional vector representations of session summaries upon completion using local Ollama (`nomic-embed-text`).
  - **`sqlite-vec` Vector Storage**: Summary vectors are indexed in the `sessions_vec` virtual table (`vec0`) using cosine distance.
  - **Reciprocal Rank Fusion (RRF)**: Combines FTS5 lexical signals and `sqlite-vec` cosine similarity into the unified `search_hybrid` engine ($k=60$).
  - **Benchmark Suite**: Compare retrieval accuracy for vague thematic queries vs. specific keywords (`python -m benchmarks.benchmark_search`).
- **Function Calling in Chat Mode & Autonomous Retrieval (Phase 6)**:
  - **Tool Schemas**: Exposes `search_past_conversations` (hybrid semantic + lexical search) and `load_session_transcript` (dialogue turn fetching) via Ollama's tool-calling API.
  - **Single-Hop Tool Execution**: Enforces a strict single retrieval cycle per turn:
    $$\text{User Query} \longrightarrow \text{LLM Tool Call} \longrightarrow \text{DB Fetch} \longrightarrow \text{Final Answer}$$
    Pass 2 is invoked with `tools=None`, strictly preventing multi-hop looping and preserving conversational responsiveness.
  - **Live UI Feedback**: Streams real-time notices (`⚡ Accessing conversation archive via ...`) to the terminal when tools are executed.
  - **Adherence Verified**: Model strictly adheres to reaching into history only when relevant to the user query, answering general queries directly without calling search.
- **Chat vs. Agent Mode Toggle & Multi-Turn Loop (Phase 7)**:
  - **Mode State Machine**: Runtime flag (`mode = "chat"` vs. `mode = "agent"`), controllable via `/mode [chat|agent]` or `--mode <chat|agent>`.
  - **Chat Mode (Safety & Low Latency)**: Tools are restricted strictly to conversation memory retrieval (`search_past_conversations`, `load_session_transcript`) with single-hop enforcement to preserve minimal latency and prevent looping.
  - **Agent Mode Action Capabilities**: Exposes action tools:
    - `run_terminal_command` - Local shell command execution with timeout and output capture
    - `python_interpreter` - Standalone Python REPL subshell for calculations and data processing
    - `read_file` - Inspect local file contents
    - `write_file` - Create or overwrite local files with automatic directory creation
    - `list_directory` - Inspect directory entries, types, and sizes
  - **Sequential Multi-Turn Execution Loop**: In agent mode, an autonomous `while` loop executes sequential tool calls across multiple turns until the task is complete, reporting live step progress (`⚡ Agent Step N: Executing ...`) and logging full multi-turn metadata to SQLite.
  - **Multi-Provider Tool Calling**: Full tool-calling and multi-turn execution support across both local Ollama and Google Gemini (`gemini-2.5-flash`).
- **Interactive Slash Commands**:
  - `/summarize` - Generate 2-sentence summary and title for current session
  - `/sessions [limit]` - List past archived sessions with message counts and summary snippets
  - `/session` - Display details and metadata of active session
  - `/search <query>` - Exact-term & keyword search across past messages (FTS5)
  - `/hybrid <query>` - Semantic & Hybrid Search combining FTS5 and sqlite-vec (RRF)
  - `/transcript [id]` - View full dialogue transcript for an identified session
  - `/new [title]` - Start a fresh session with clean context
  - `/compress [on|off|stats]` - Toggle or inspect Headroom context compression
  - `/persona [reload|edit]` - View, reload, or see edit path for `personality.md`
  - `/models` - List available models with sizes and parameter counts
  - `/use <model>` - Switch active model dynamically
  - `/mode [chat|agent]` - Inspect or toggle between Chat and Agent modes
  - `/buffer` - View active context window and character/token breakdown
  - `/clear` - Reset conversation history while preserving system prompt
  - `/system [prompt]` - View or modify pinned system prompt directly
  - `/help` - Show command reference
  - `/exit` or `/quit` - Quit session (auto-summarizes & vector indexes unless empty)

---

## Quick Start

### 1. Activate Environment
```bash
source .venv/bin/activate
# or install requirements:
pip install -r requirements.txt
```

### 2. Launch Interactive Chat
```bash
python main.py
```

### 3. Optional Arguments
```bash
# Specify custom model or Ollama host:
python main.py --model Gemma4-26000-ctx:latest --host http://localhost:11434

# Change sliding window size:
python main.py --window-size 10

# Disable context compression:
python main.py --no-compression

# Run a quick one-off query:
python main.py -q "Explain context compression in 2 sentences."
```
