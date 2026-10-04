<div align="center">

# Finch.ai

<img src="assets/finch_mascot.jpg" alt="Finch.ai Mascot" width="260" style="border-radius: 12px;" />

<p>
A high-performance, low-latency AI assistant and autonomous agent supporting local Ollama, Google Gemini, and OpenAI-compatible runtimes.
</p>

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Documentation](https://img.shields.io/badge/Docs-LaTeX%2FPDF-orange.svg)](docs/main.pdf)

</div>

---

## Key Features

- **Multi-Provider Support**: Seamlessly connect to local Ollama models (`llama3.2`, `mistral`, `qwen2.5`), Google Gemini (`gemini-2.5-flash`), or OpenAI-compatible endpoints (vLLM, LM Studio, Groq, OpenRouter).
- **Headroom Context Compression**: Intelligently compresses tool outputs, logs, code, and history to save **15%–60%+ tokens** while preserving persona rules and recent turns.
- **Dual Execution Modes**:
  - **Chat Mode**: Fast conversational assistant with single-hop memory retrieval.
  - **Agent Mode**: Autonomous multi-turn loop with tools for terminal execution, Python REPL, web search, and filesystem access.
- **3-State Tool Guardrails**: Configure tool categories (`terminal`, `python`, `web`, `files`) to `OFF`, `ASK` (interactive human confirmation), or `AUTO`.
- **Hybrid Long-Term Memory**: SQLite conversation store with full-text search (FTS5) and semantic vector search (`sqlite-vec` + Reciprocal Rank Fusion).
- **Dynamic Persona & User Facts**: Identity and behavior adapt via `personality.md` alongside an isolated persistent fact store (`/remember`).
- **Telemetry & Maintenance**: Real-time streaming metrics (tokens/sec, TTFT, RAM), transparent `zstd` message compression, and WAL maintenance routines.
- **Rich Session Exports**: Export chats to self-contained styled HTML reports, GitHub-flavored Markdown, or full backup archives.

---

## Quick Start

### 1. Installation
```bash
# Clone repository and enter directory
cd Ai_Assistant

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration
Copy `.env.example` to `.env` to configure your default provider and API keys:
```bash
cp .env.example .env
```
*(Ollama works locally out-of-the-box. Add keys to `.env` if using Gemini or OpenAI-compatible APIs.)*

### 3. Launch
```bash
# Start interactive session (default Ollama model)
python main.py

# Switch provider or model on launch
python main.py --provider gemini --model gemini-2.5-flash
python main.py --provider openai --model gpt-4o-mini

# Start directly in Agent Mode
python main.py --mode agent

# Run a one-off query
python main.py -q "Explain context compression in two sentences."
```

---

## Interactive Slash Commands

Control Finch.ai inside an active session using slash commands:

| Command | Description |
| :--- | :--- |
| `/mode [chat\|agent]` | Toggle between Chat mode and autonomous Agent mode |
| `/tools [cat] [off\|ask\|auto]` | Inspect or update tool permissions (`terminal`, `python`, `web`, `files`) |
| `/remember <fact>` / `/facts` | Store or inspect persistent user preferences and project facts |
| `/search <query>` / `/hybrid <q>` | Search past conversations via keyword (FTS5) or semantic hybrid search |
| `/persona [reload\|adapt\|edit]` | View, evolve, or reload dynamic persona from `personality.md` |
| `/models` / `/use <model>` | List available models or switch the active model on the fly |
| `/export <md\|html\|bundle>` | Export session as formatted Markdown, styled HTML, or a backup zip |
| `/compress [on\|off\|stats]` | Toggle or inspect Headroom context compression statistics |
| `/storage` / `/vacuum` | Check SQLite disk footprint or trigger database maintenance |
| `/help` | Display the complete command reference list |

---

## Model Context Protocol (MCP) Server

Finch exposes its memory, hybrid search, context compression, and autonomous agent capabilities as a standard FastMCP service over STDIO transport:

```bash
# Run Finch FastMCP service
python -m finch.mcp_server
```

### Exposed MCP Tools

| Tool | Description |
| :--- | :--- |
| `finch_run_subagent(instruction, max_turns=8)` | Headless multi-turn agent execution delegating filesystem, Python REPL, or shell tasks with Headroom compression and DB logging |
| `run_subagent(instruction, max_turns=8)` | Alias for autonomous subagent delegation |
| `finch_remember_fact(fact, category)` | Insert verified facts directly into Finch long-term persistent memory |
| `finch_get_user_facts(category)` | Fetch persistent verified user profile facts from `user_facts` table |
| `finch_search_history(query, limit=5)` | Hybrid FTS5 + `sqlite-vec` RRF search across archived conversations |
| `finch_get_transcript(session_id)` | Retrieve and decompress zstd turn-by-turn dialogue transcripts |
| `compress_context(messages, model)` | Run Headroom context compression pipeline on message payloads |
| `run_finch_query(prompt, mode)` | Execute queries through Finch orchestrator in chat or agent mode |
| `get_storage_stats()` | Retrieve SQLite database storage, WAL, and telemetry metrics |

---

## Technical Documentation & Architecture Reference

For detailed subsystem architecture, database schemas, mathematical formulations (Reciprocal Rank Fusion, token compression economics), benchmarks, and developer guides, refer to the technical reference manual in [`docs/`](docs/):

- **Compiled PDF Manual**: [`docs/main.pdf`](docs/main.pdf)
- **Modular Chapters**: [`docs/sections/`](docs/sections/)
- **Build Instructions**: See [`docs/README.md`](docs/README.md) (`make` or `./build.sh`)
