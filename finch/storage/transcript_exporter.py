"""Markdown and HTML transcript exporter for conversation sessions."""
import html
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .sqlite_archive import MessageRecord, SessionRecord, SQLiteArchive


def utc_now_compact() -> str:
    """Return compact UTC timestamp string for filenames."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def export_transcript_markdown(
    archive: SQLiteArchive,
    session_id: str,
    output_path: Optional[str] = None,
    transcript_dir: str = "transcripts",
) -> str:
    """Export a session transcript to a GitHub-flavored Markdown file.

    Returns:
        Absolute path to the created markdown transcript.
    """
    session = archive.get_session(session_id)
    if not session:
        raise ValueError(f"Session '{session_id}' not found in archive.")

    messages = archive.get_messages(session_id, include_system=True)

    if not output_path:
        os.makedirs(transcript_dir, exist_ok=True)
        filename = f"transcript_{session.id}_{utc_now_compact()}.md"
        target_path = os.path.abspath(os.path.join(transcript_dir, filename))
    else:
        target_path = os.path.abspath(output_path)
        parent = os.path.dirname(target_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    lines = [
        f"# Conversation Transcript: {session.title}",
        "",
        f"- **Session ID**: `{session.id}`",
        f"- **Model**: `{session.model}`",
        f"- **Mode**: `{session.mode.upper()}`",
        f"- **Created At**: {session.created_at}",
        f"- **Updated At**: {session.updated_at}",
        f"- **Total Messages**: {len(messages)}",
    ]

    if session.summary:
        lines.extend([
            f"- **Summary**: {session.summary}",
        ])

    lines.extend(["", "---", ""])

    for msg in messages:
        role = msg.role.lower()
        time_str = msg.timestamp[:19].replace("T", " ")

        if role == "user":
            lines.append(f"### 👤 User `[{time_str}]`")
            lines.append("")
            lines.append(msg.content.strip())
            lines.append("")

        elif role == "assistant":
            lines.append(f"### 🤖 Assistant (Finch.ai) `[{time_str}]`")
            lines.append("")

            thinking_text = msg.thinking
            body_content = msg.content or ""
            if not thinking_text and "<think>" in body_content and "</think>" in body_content:
                import re
                m = re.search(r"<think>(.*?)</think>", body_content, flags=re.DOTALL)
                if m:
                    thinking_text = m.group(1).strip()
                    body_content = re.sub(r"<think>.*?</think>", "", body_content, flags=re.DOTALL).strip()

            # If thinking process is present
            if thinking_text:
                lines.append("> **💭 Thinking Process:**")
                for tline in thinking_text.strip().splitlines():
                    lines.append(f"> {tline}")
                lines.append("")

            # If metadata contains tool calls
            meta = msg.metadata or {}
            tool_calls = meta.get("tool_calls", [])
            if tool_calls:
                lines.append("> **⚡ Tools Executed:**")
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    fn_name = fn.get("name", "tool")
                    fn_args = fn.get("arguments", {})
                    lines.append(f"> - `{fn_name}`: `{json.dumps(fn_args)}`")
                lines.append("")

            lines.append(body_content.strip())
            lines.append("")

        elif role == "system":
            lines.append(f"### ⚙️ System Prompt `[{time_str}]`")
            lines.append("")
            lines.append(f"```markdown\n{msg.content.strip()}\n```")
            lines.append("")

        elif role == "tool":
            lines.append(f"### 🛠️ Tool Output `[{time_str}]`")
            lines.append("")
            lines.append(f"```\n{msg.content.strip()}\n```")
            lines.append("")

    content = "\n".join(lines).strip() + "\n"
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(content)

    return target_path


def export_transcript_html(
    archive: SQLiteArchive,
    session_id: str,
    output_path: Optional[str] = None,
    transcript_dir: str = "transcripts",
) -> str:
    """Export a session transcript to a standalone, modern HTML page.

    Returns:
        Absolute path to the created HTML transcript.
    """
    session = archive.get_session(session_id)
    if not session:
        raise ValueError(f"Session '{session_id}' not found in archive.")

    messages = archive.get_messages(session_id, include_system=True)

    if not output_path:
        os.makedirs(transcript_dir, exist_ok=True)
        filename = f"transcript_{session.id}_{utc_now_compact()}.html"
        target_path = os.path.abspath(os.path.join(transcript_dir, filename))
    else:
        target_path = os.path.abspath(output_path)
        parent = os.path.dirname(target_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    def escape_text(text: str) -> str:
        return html.escape(text or "")

    def format_body(text: str) -> str:
        escaped = html.escape(text or "")
        return escaped.replace("\n", "<br>")

    message_blocks = []
    for msg in messages:
        role = msg.role.lower()
        time_str = msg.timestamp[:19].replace("T", " ")

        if role == "user":
            block = f"""
            <div class="message user-message">
                <div class="message-header">
                    <span class="avatar user-avatar">👤</span>
                    <span class="sender-name">User</span>
                    <span class="timestamp">{escape_text(time_str)}</span>
                </div>
                <div class="message-content">{format_body(msg.content)}</div>
            </div>
            """
        elif role == "assistant":
            thinking_text = msg.thinking
            body_content = msg.content or ""
            if not thinking_text and "<think>" in body_content and "</think>" in body_content:
                import re
                m = re.search(r"<think>(.*?)</think>", body_content, flags=re.DOTALL)
                if m:
                    thinking_text = m.group(1).strip()
                    body_content = re.sub(r"<think>.*?</think>", "", body_content, flags=re.DOTALL).strip()

            thinking_html = ""
            if thinking_text:
                thinking_html = f"""
                <details class="thinking-box">
                    <summary>💭 Thinking Process</summary>
                    <div class="thinking-content">{format_body(thinking_text)}</div>
                </details>
                """

            tools_html = ""
            meta = msg.metadata or {}
            tool_calls = meta.get("tool_calls", [])
            if tool_calls:
                tool_lines = []
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    fn_name = fn.get("name", "tool")
                    fn_args = json.dumps(fn.get("arguments", {}), indent=2)
                    tool_lines.append(f"<strong>{escape_text(fn_name)}</strong>:\n{escape_text(fn_args)}")
                tools_html = f"""
                <details class="tool-box">
                    <summary>⚡ Autonomous Tool Calls ({len(tool_calls)})</summary>
                    <pre><code>{escape_text(chr(10).join(tool_lines))}</code></pre>
                </details>
                """

            block = f"""
            <div class="message assistant-message">
                <div class="message-header">
                    <span class="avatar assistant-avatar">🤖</span>
                    <span class="sender-name">Finch.ai Assistant</span>
                    <span class="timestamp">{escape_text(time_str)}</span>
                </div>
                {thinking_html}
                {tools_html}
                <div class="message-content">{format_body(body_content)}</div>
            </div>
            """
        elif role == "system":
            block = f"""
            <details class="system-message">
                <summary>⚙️ Pinned System Prompt [{escape_text(time_str)}]</summary>
                <pre class="system-content"><code>{escape_text(msg.content)}</code></pre>
            </details>
            """
        elif role == "tool":
            block = f"""
            <details class="tool-result-message">
                <summary>🛠️ Tool Execution Output [{escape_text(time_str)}]</summary>
                <pre class="tool-output"><code>{escape_text(msg.content)}</code></pre>
            </details>
            """
        else:
            block = f"""
            <div class="message other-message">
                <div class="message-header">
                    <span class="sender-name">{escape_text(role.capitalize())}</span>
                    <span class="timestamp">{escape_text(time_str)}</span>
                </div>
                <div class="message-content">{format_body(msg.content)}</div>
            </div>
            """

        message_blocks.append(block)

    summary_html = ""
    if session.summary:
        summary_html = f"""
        <div class="summary-card">
            <div class="summary-title">📝 Session Summary</div>
            <div class="summary-text">{escape_text(session.summary)}</div>
        </div>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Transcript: {escape_text(session.title)}</title>
    <style>
        :root {{
            --bg: #0f172a;
            --surface: #1e293b;
            --surface-hover: #334155;
            --border: #334155;
            --text-main: #f8fafc;
            --text-dim: #94a3b8;
            --accent-cyan: #38bdf8;
            --accent-blue: #60a5fa;
            --accent-green: #34d399;
            --accent-purple: #c084fc;
            --accent-yellow: #fbbf24;
        }}
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            background-color: var(--bg);
            color: var(--text-main);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            padding: 2rem 1rem;
        }}
        .container {{
            max-width: 860px;
            margin: 0 auto;
        }}
        header {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        }}
        h1 {{
            font-size: 1.75rem;
            font-weight: 700;
            margin-bottom: 0.75rem;
            color: var(--accent-cyan);
        }}
        .badges {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-bottom: 0.75rem;
        }}
        .badge {{
            font-size: 0.8rem;
            padding: 0.25rem 0.6rem;
            border-radius: 9999px;
            background: rgba(255, 255, 255, 0.08);
            color: var(--text-dim);
            border: 1px solid var(--border);
        }}
        .badge strong {{
            color: var(--text-main);
        }}
        .summary-card {{
            background: rgba(56, 189, 248, 0.08);
            border: 1px solid rgba(56, 189, 248, 0.3);
            border-radius: 8px;
            padding: 1rem;
            margin-top: 1rem;
        }}
        .summary-title {{
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--accent-cyan);
            margin-bottom: 0.25rem;
        }}
        .summary-text {{
            font-size: 0.95rem;
            color: var(--text-main);
        }}
        .dialogue {{
            display: flex;
            flex-direction: column;
            gap: 1.25rem;
        }}
        .message {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 1.25rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }}
        .user-message {{
            border-left: 4px solid var(--accent-blue);
        }}
        .assistant-message {{
            border-left: 4px solid var(--accent-green);
        }}
        .message-header {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.75rem;
        }}
        .avatar {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 28px;
            height: 28px;
            border-radius: 6px;
            background: rgba(255, 255, 255, 0.1);
            font-size: 1rem;
        }}
        .sender-name {{
            font-weight: 600;
            font-size: 0.95rem;
        }}
        .timestamp {{
            font-size: 0.8rem;
            color: var(--text-dim);
            margin-left: auto;
        }}
        .message-content {{
            font-size: 0.95rem;
            word-break: break-word;
            white-space: pre-wrap;
        }}
        details {{
            background: rgba(0, 0, 0, 0.25);
            border: 1px solid var(--border);
            border-radius: 6px;
            margin: 0.75rem 0;
            padding: 0.5rem 0.75rem;
        }}
        summary {{
            cursor: pointer;
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-dim);
            outline: none;
        }}
        summary:hover {{
            color: var(--accent-cyan);
        }}
        .thinking-content {{
            margin-top: 0.5rem;
            padding-top: 0.5rem;
            border-top: 1px dashed var(--border);
            font-size: 0.85rem;
            color: #cbd5e1;
            font-style: italic;
        }}
        pre {{
            background: #090d16;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 6px;
            padding: 0.75rem;
            overflow-x: auto;
            font-size: 0.85rem;
            color: #e2e8f0;
            margin-top: 0.5rem;
        }}
        code {{
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        }}
        footer {{
            text-align: center;
            margin-top: 3rem;
            font-size: 0.85rem;
            color: var(--text-dim);
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{escape_text(session.title)}</h1>
            <div class="badges">
                <span class="badge">Session ID: <strong>{escape_text(session.id)}</strong></span>
                <span class="badge">Model: <strong>{escape_text(session.model)}</strong></span>
                <span class="badge">Mode: <strong>{escape_text(session.mode.upper())}</strong></span>
                <span class="badge">Messages: <strong>{len(messages)}</strong></span>
                <span class="badge">Created: <strong>{escape_text(session.created_at[:19].replace('T', ' '))}</strong></span>
            </div>
            {summary_html}
        </header>

        <div class="dialogue">
            {''.join(message_blocks)}
        </div>

        <footer>
            Exported by Finch.ai Assistant • Standalone Transcript
        </footer>
    </div>
</body>
</html>
"""
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return target_path
