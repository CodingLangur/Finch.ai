"""Tool schemas and execution dispatcher for autonomous conversation retrieval and agent actions."""
import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .assistant import AIAssistant


# --- Tool Schemas for Memory Retrieval (Chat & Agent Modes) ---

SEARCH_PAST_CONVERSATIONS_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_past_conversations",
        "description": (
            "Search archived conversation history using hybrid retrieval (semantic vector search + exact lexical matching) "
            "to recall topics, previous discussions, code snippets, dates, or decisions. "
            "Call this tool ONLY when the user asks about or refers to previous/past conversations or earlier topics. "
            "Do NOT call this tool for general knowledge, coding assistance, or questions about the current active turn."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query, topic, phrase, or question to search for in past conversations.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of relevant past sessions to retrieve (default: 5).",
                },
            },
            "required": ["query"],
        },
    },
}

LOAD_SESSION_TRANSCRIPT_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "load_session_transcript",
        "description": (
            "Retrieve the complete chronological dialogue transcript for an identified past conversation session. "
            "Call this tool when you need the exact dialogue turns, code snippets, or details from a specific session ID."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "The unique session ID (e.g., 'sess_...') to load the transcript for.",
                },
            },
            "required": ["session_id"],
        },
    },
}

SAVE_USER_FACT_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "save_user_fact",
        "description": (
            "Save an enduring fact, preference, or piece of knowledge about the user into persistent long-term memory. "
            "Call this when the user shares personal preferences, project specifics, habits, or environment facts "
            "that should be remembered across future sessions (e.g. 'I work in Ubuntu', 'My favorite language is Rust')."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "fact": {
                    "type": "string",
                    "description": "The concise fact to store in memory.",
                },
                "category": {
                    "type": "string",
                    "description": "Optional category (e.g. 'tech_stack', 'preference', 'project', 'general').",
                },
            },
            "required": ["fact"],
        },
    },
}

LIST_USER_FACTS_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "list_user_facts",
        "description": (
            "Retrieve verified facts and preferences about the user stored in persistent long-term memory."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Optional category filter.",
                },
            },
        },
    },
}

# --- Action Tool Schemas for Agent Mode (Phase 7) ---

RUN_TERMINAL_COMMAND_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "run_terminal_command",
        "description": (
            "Execute a shell or terminal command locally and return its stdout, stderr, and exit code. "
            "Use this tool to inspect system state, run shell commands, check git status, or build projects. "
            "Available in agent mode."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command line to execute.",
                },
                "cwd": {
                    "type": "string",
                    "description": "Optional working directory in which to execute the command.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Maximum execution time in seconds (default: 30).",
                },
            },
            "required": ["command"],
        },
    },
}

PYTHON_INTERPRETER_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "python_interpreter",
        "description": (
            "Execute Python code in a standalone Python subshell and return stdout, stderr, and exit code. "
            "Use this tool for mathematical calculations, data parsing, algorithmic problem solving, or testing logic. "
            "Available in agent mode."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python source code to execute.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Maximum execution time in seconds (default: 30).",
                },
            },
            "required": ["code"],
        },
    },
}

READ_FILE_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": (
            "Read and return the text contents of a file from the local filesystem. "
            "Available in agent mode."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to read (relative to workspace or absolute).",
                },
                "max_lines": {
                    "type": "integer",
                    "description": "Optional maximum number of lines to read (default: 500).",
                },
            },
            "required": ["file_path"],
        },
    },
}

WRITE_FILE_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": (
            "Write or overwrite text content to a file on the local filesystem. "
            "Parent directories will be created automatically if they do not exist. "
            "Available in agent mode."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to create or overwrite.",
                },
                "content": {
                    "type": "string",
                    "description": "The text content to write into the file.",
                },
            },
            "required": ["file_path", "content"],
        },
    },
}

LIST_DIRECTORY_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "list_directory",
        "description": (
            "List files and subdirectories within a directory on the local filesystem, including types and sizes. "
            "Available in agent mode."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "directory_path": {
                    "type": "string",
                    "description": "Path to the directory to list (default: current directory '.').",
                },
            },
        },
    },
}


# --- Web Access Tool Schemas ---

WEB_SEARCH_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the live web for recent information, documentation, news, or articles. "
            "Available in agent mode when web access is enabled."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query or terms to look up on the web.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of search results to return (default: 5).",
                },
            },
            "required": ["query"],
        },
    },
}

FETCH_WEB_PAGE_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "fetch_web_page",
        "description": (
            "Fetch and extract readable text content from a web URL. "
            "Available in agent mode when web access is enabled."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The HTTP or HTTPS URL of the web page to fetch.",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Optional maximum number of characters to extract (default: 5000).",
                },
            },
            "required": ["url"],
        },
    },
}


# --- Tool Groupings & Category Mappings ---

TOOL_CATEGORIES: Dict[str, List[Dict[str, Any]]] = {
    "terminal": [RUN_TERMINAL_COMMAND_TOOL],
    "python": [PYTHON_INTERPRETER_TOOL],
    "web": [WEB_SEARCH_TOOL, FETCH_WEB_PAGE_TOOL],
    "files": [READ_FILE_TOOL, WRITE_FILE_TOOL, LIST_DIRECTORY_TOOL],
    "memory": [
        SEARCH_PAST_CONVERSATIONS_TOOL,
        LOAD_SESSION_TRANSCRIPT_TOOL,
    ],
}

TOOL_NAME_TO_CATEGORY: Dict[str, str] = {
    "run_terminal_command": "terminal",
    "python_interpreter": "python",
    "web_search": "web",
    "fetch_web_page": "web",
    "read_file": "files",
    "write_file": "files",
    "list_directory": "files",
    "search_past_conversations": "memory",
    "load_session_transcript": "memory",
    "save_user_fact": "memory",
    "list_user_facts": "memory",
}

CHAT_TOOLS: List[Dict[str, Any]] = [
    SEARCH_PAST_CONVERSATIONS_TOOL,
    LOAD_SESSION_TRANSCRIPT_TOOL,
]

MEMORY_FACT_TOOLS: List[Dict[str, Any]] = [
    SAVE_USER_FACT_TOOL,
    LIST_USER_FACTS_TOOL,
]

AGENT_ACTION_TOOLS: List[Dict[str, Any]] = [
    RUN_TERMINAL_COMMAND_TOOL,
    PYTHON_INTERPRETER_TOOL,
    READ_FILE_TOOL,
    WRITE_FILE_TOOL,
    LIST_DIRECTORY_TOOL,
]

WEB_TOOLS: List[Dict[str, Any]] = [
    WEB_SEARCH_TOOL,
    FETCH_WEB_PAGE_TOOL,
]

AGENT_TOOLS: List[Dict[str, Any]] = (
    CHAT_TOOLS + AGENT_ACTION_TOOLS
)

ALL_AGENT_TOOLS: List[Dict[str, Any]] = (
    CHAT_TOOLS + AGENT_ACTION_TOOLS + WEB_TOOLS + MEMORY_FACT_TOOLS
)


class ToolDispatcher:
    """Dispatches tool calls to Finch.ai storage engines and agent action executors."""

    def __init__(self, assistant: "AIAssistant"):
        self.assistant = assistant
        self.confirmation_callback = None

    async def execute(self, tool_name: str, arguments: Dict[str, Any] | str) -> str:
        """Execute a tool call and return formatted context string."""
        if isinstance(arguments, str):
            try:
                args = json.loads(arguments)
            except Exception:
                args = {
                    "query": arguments,
                    "command": arguments,
                    "code": arguments,
                    "file_path": arguments,
                    "directory_path": arguments,
                    "url": arguments,
                    "fact": arguments,
                }
        else:
            args = arguments or {}

        # 0. Check category permission policy and human-in-the-loop confirmation
        cat = TOOL_NAME_TO_CATEGORY.get(tool_name)
        if cat and cat != "memory":
            if hasattr(self.assistant, "is_tool_category_enabled") and not self.assistant.is_tool_category_enabled(cat):
                return (
                    f"Error: Access to tool '{tool_name}' is disabled. "
                    f"The '{cat}' tool capability is currently turned off by permission policy. "
                    f"Use /tools {cat} [ask|auto] to allow access."
                )

            # Check for Human-in-the-Loop confirmation mode
            if hasattr(self.assistant, "get_tool_permission_mode"):
                mode = self.assistant.get_tool_permission_mode(cat)
                if mode == "ask":
                    cb = getattr(self.assistant, "tool_confirmation_callback", None) or self.confirmation_callback
                    if cb:
                        import inspect
                        if inspect.iscoroutinefunction(cb):
                            confirmed = await cb(tool_name, args)
                        else:
                            res = cb(tool_name, args)
                            if inspect.isawaitable(res):
                                confirmed = await res
                            else:
                                confirmed = res
                        if not confirmed:
                            return f"Execution cancelled: User denied permission to execute tool '{tool_name}'."

        # 1. Past Conversation Search
        if tool_name == "search_past_conversations":
            query = str(args.get("query", "")).strip()
            limit = int(args.get("limit", 5))
            if not query:
                return "Error: Search query cannot be empty."

            results = await self.assistant.search_hybrid(query=query, limit=limit)
            if not results:
                return f"No past conversations found matching '{query}'."

            lines = [f"Found {len(results)} relevant past conversation sessions:"]
            for r in results:
                lines.append(f"\n- Session ID: {r.session_id}")
                lines.append(f"  Title: {r.title}")
                if r.summary:
                    lines.append(f"  Summary: {r.summary}")
                if r.matched_snippets:
                    clean_snips = [
                        s.replace("[MATCH]", "").replace("[/MATCH]", "").strip()
                        for s in r.matched_snippets[:2]
                    ]
                    lines.append(f"  Snippet: {' | '.join(clean_snips)}")
                if r.cosine_similarity is not None:
                    lines.append(f"  Relevance: RRF={r.rrf_score:.4f}, Cosine Similarity={r.cosine_similarity*100:.1f}%")

            lines.append("\nTip: To read full dialogue turns from any session above, call load_session_transcript(session_id).")
            return "\n".join(lines)

        # 2. Session Transcript Loader
        elif tool_name == "load_session_transcript":
            session_id = str(args.get("session_id", "")).strip()
            if not session_id:
                return "Error: session_id is required."

            transcript = self.assistant.load_session_transcript(session_id=session_id)
            if not transcript:
                return f"Session '{session_id}' not found in database."

            return transcript.formatted_transcript

        # 3. Long-Term Fact Store Tools
        elif tool_name == "save_user_fact":
            fact_text = str(args.get("fact", "")).strip()
            category = str(args.get("category", "general")).strip()
            if not fact_text:
                return "Error: Fact content cannot be empty."
            if hasattr(self.assistant, "add_user_fact"):
                fact_id = self.assistant.add_user_fact(fact=fact_text, category=category)
                return f"Fact successfully stored in long-term memory (ID: {fact_id}, Category: {category})."
            return "Error: Fact store not available on assistant."

        elif tool_name == "list_user_facts":
            cat_filter = args.get("category")
            if hasattr(self.assistant, "list_user_facts"):
                facts = self.assistant.list_user_facts(category=cat_filter)
                if not facts:
                    return "No long-term user facts found."
                lines = [f"Found {len(facts)} user facts:"]
                for f in facts:
                    lines.append(f"- [ID {f['id']}] ({f['category']}): {f['fact']}")
                return "\n".join(lines)
            return "Error: Fact store not available on assistant."

        # 3. Terminal Command Execution
        elif tool_name == "run_terminal_command":
            command = str(args.get("command", "")).strip()
            if not command:
                return "Error: command is required."
            cwd = args.get("cwd") or None
            if cwd:
                cwd = os.path.abspath(os.path.expanduser(str(cwd).strip()))
                if not os.path.exists(cwd):
                    return f"Error: Working directory '{cwd}' does not exist."
                if not os.path.isdir(cwd):
                    return f"Error: '{cwd}' is a file, not a directory."

            default_timeout = getattr(getattr(self.assistant, "config", None), "terminal_timeout", 30.0)
            timeout = float(args.get("timeout") or default_timeout)

            try:
                proc = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=cwd,
                    start_new_session=True if os.name == "posix" else False,
                )
                try:
                    stdout_data, stderr_data = await asyncio.wait_for(
                        proc.communicate(), timeout=timeout
                    )
                except asyncio.TimeoutError:
                    if os.name == "posix":
                        import signal
                        try:
                            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                        except (ProcessLookupError, OSError):
                            try:
                                proc.kill()
                            except ProcessLookupError:
                                pass
                    else:
                        try:
                            proc.kill()
                        except ProcessLookupError:
                            pass
                    await proc.wait()
                    return f"Error: Command timed out after {timeout} seconds."

                stdout_str = stdout_data.decode("utf-8", errors="replace").strip()
                stderr_str = stderr_data.decode("utf-8", errors="replace").strip()

                if len(stdout_str) > 8000:
                    stdout_str = stdout_str[:8000] + "\n...[Output truncated]"
                if len(stderr_str) > 4000:
                    stderr_str = stderr_str[:4000] + "\n...[Stderr truncated]"

                res_lines = [f"Exit Code: {proc.returncode}"]
                if stdout_str:
                    res_lines.append(f"STDOUT:\n{stdout_str}")
                else:
                    res_lines.append("STDOUT: (none)")
                if stderr_str:
                    res_lines.append(f"STDERR:\n{stderr_str}")
                return "\n".join(res_lines)
            except Exception as e:
                return f"Error executing command: {e}"

        # 4. Python Interpreter
        elif tool_name == "python_interpreter":
            code = str(args.get("code", "")).strip()
            if not code:
                return "Error: code is required."
            timeout = float(args.get("timeout", 30))

            try:
                proc = await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-c",
                    code,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout_data, stderr_data = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
                stdout_str = stdout_data.decode("utf-8", errors="replace").strip()
                stderr_str = stderr_data.decode("utf-8", errors="replace").strip()

                if len(stdout_str) > 8000:
                    stdout_str = stdout_str[:8000] + "\n...[Output truncated]"
                if len(stderr_str) > 4000:
                    stderr_str = stderr_str[:4000] + "\n...[Stderr truncated]"

                res_lines = [f"Exit Code: {proc.returncode}"]
                if stdout_str:
                    res_lines.append(f"Output:\n{stdout_str}")
                else:
                    res_lines.append("Output: (no output)")
                if stderr_str:
                    res_lines.append(f"Errors/Traceback:\n{stderr_str}")
                return "\n".join(res_lines)
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                return f"Error: Python execution timed out after {timeout} seconds."
            except Exception as e:
                return f"Error executing Python code: {e}"

        # 5. Read File
        elif tool_name == "read_file":
            file_path = str(args.get("file_path", "")).strip()
            if not file_path:
                return "Error: file_path is required."
            max_lines = int(args.get("max_lines", 500))

            try:
                if not os.path.exists(file_path):
                    return f"Error: File '{file_path}' does not exist."
                if os.path.isdir(file_path):
                    return f"Error: '{file_path}' is a directory, not a file. Use list_directory instead."

                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    lines = [f.readline() for _ in range(max_lines + 1)]

                truncated = len(lines) > max_lines
                content = "".join(lines[:max_lines])
                if truncated:
                    content += f"\n...[Truncated after {max_lines} lines]"
                return content if content else "(empty file)"
            except Exception as e:
                return f"Error reading file '{file_path}': {e}"

        # 6. Write File
        elif tool_name == "write_file":
            file_path = str(args.get("file_path", "")).strip()
            if not file_path:
                return "Error: file_path is required."
            content = str(args.get("content", ""))

            try:
                parent_dir = os.path.dirname(os.path.abspath(file_path))
                if parent_dir:
                    os.makedirs(parent_dir, exist_ok=True)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                return f"Successfully wrote {len(content)} characters to '{file_path}'."
            except Exception as e:
                return f"Error writing file '{file_path}': {e}"

        # 7. List Directory
        elif tool_name == "list_directory":
            dir_path = str(args.get("directory_path", ".")).strip() or "."

            try:
                if not os.path.exists(dir_path):
                    return f"Error: Directory '{dir_path}' does not exist."
                if not os.path.isdir(dir_path):
                    return f"Error: '{dir_path}' is a file, not a directory. Use read_file instead."

                entries = os.listdir(dir_path)
                lines = [f"Directory listing for '{dir_path}' ({len(entries)} items):"]
                for entry in sorted(entries)[:100]:
                    full = os.path.join(dir_path, entry)
                    is_dir = os.path.isdir(full)
                    type_str = "DIR" if is_dir else "FILE"
                    size_str = ""
                    if not is_dir:
                        try:
                            size_str = f" ({os.path.getsize(full)} bytes)"
                        except OSError:
                            pass
                    lines.append(f"  [{type_str}] {entry}{size_str}")

                if len(entries) > 100:
                    lines.append(f"  ...[Truncated {len(entries) - 100} remaining items]")
                return "\n".join(lines)
            except Exception as e:
                return f"Error listing directory '{dir_path}': {e}"

        # 8. Web Search
        elif tool_name == "web_search":
            query = str(args.get("query", "")).strip()
            if not query:
                return "Error: query is required."
            max_results = int(args.get("max_results", 5))

            try:
                import httpx
                import re
                import urllib.parse

                encoded_q = urllib.parse.quote_plus(query)
                search_url = f"https://html.duckduckgo.com/html/?q={encoded_q}"
                async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                    resp = await client.post(
                        search_url,
                        data={"q": query},
                        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
                    )
                    html = resp.text
                    snippets = re.findall(r'<a class="result__snippet[^>]*>(.*?)</a>', html, flags=re.DOTALL)
                    titles = re.findall(r'<a class="result__url[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html, flags=re.DOTALL)

                    results = []
                    for i in range(min(max_results, len(snippets))):
                        clean_snip = re.sub(r"<[^>]+>", "", snippets[i]).strip()
                        url_item = titles[i][0] if i < len(titles) else ""
                        results.append(f"{i+1}. {clean_snip}\n   URL: {url_item}")

                    if results:
                        return f"Web search results for '{query}':\n\n" + "\n\n".join(results)
                    return f"No web search results found for '{query}'."
            except Exception as e:
                return f"Error executing web search for '{query}': {e}"

        # 9. Fetch Web Page
        elif tool_name == "fetch_web_page":
            url = str(args.get("url", "")).strip()
            if not url:
                return "Error: url is required."
            if not url.startswith(("http://", "https://")):
                url = "https://" + url
            max_chars = int(args.get("max_chars", 5000))

            try:
                import httpx
                import re

                async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                    resp = await client.get(
                        url,
                        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
                    )
                    resp.raise_for_status()
                    html_text = resp.text
                    clean_html = re.sub(
                        r"<(script|style)[^>]*>.*?</\1>", "", html_text, flags=re.DOTALL | re.IGNORECASE
                    )
                    plain_text = re.sub(r"<[^>]+>", " ", clean_html)
                    plain_text = re.sub(r"\s+", " ", plain_text).strip()
                    if len(plain_text) > max_chars:
                        plain_text = plain_text[:max_chars] + f"\n...[Truncated after {max_chars} chars]"
                    return f"Content of {url}:\n{plain_text}"
            except Exception as e:
                return f"Error fetching web page '{url}': {e}"

        else:
            return f"Error: Tool '{tool_name}' is not recognized."
