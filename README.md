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
- **Interactive Slash Commands**:
  - `/summarize` - Generate 2-sentence summary and title for current session
  - `/sessions [limit]` - List past archived sessions with message counts and summary snippets
  - `/session` - Display details and metadata of active session
  - `/new [title]` - Start a fresh session with clean context
  - `/compress [on|off|stats]` - Toggle or inspect Headroom context compression
  - `/persona [reload|edit]` - View, reload, or see edit path for `personality.md`
  - `/models` - List available models with sizes and parameter counts
  - `/use <model>` - Switch active model dynamically
  - `/mode [chatbot|agent]` - Inspect or toggle between Chatbot and Agent modes
  - `/buffer` - View active context window and character/token breakdown
  - `/clear` - Reset conversation history while preserving system prompt
  - `/system [prompt]` - View or modify pinned system prompt directly
  - `/help` - Show command reference
  - `/exit` or `/quit` - Quit session (auto-summarizes unless empty)

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
