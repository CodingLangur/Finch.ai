"""Interactive Rich CLI for streaming chat, telemetry, and buffer monitoring."""
import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional
import psutil
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory, InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from rich.box import ROUNDED
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..compression.pipeline import CompressionStats
from ..core.assistant import AIAssistant, AssistantMode
from ..providers.base import StreamStats


def get_process_memory_mb() -> float:
    """Return current process resident memory (RSS) in megabytes."""
    process = psutil.Process(os.getpid())
    return round(process.memory_info().rss / (1024 * 1024), 2)


def create_cli_keybindings() -> KeyBindings:
    """Create key bindings for prompt_toolkit with native multiline editing and submit triggers.
    
    Submits on:
    - Alt+Enter (or Escape then Enter)
    - Double Enter (when cursor is at end of buffer and last line is blank)
    - Single Enter for slash commands (/help, /exit, etc.)
    - Ctrl+J / Ctrl+Enter
    """
    kb = KeyBindings()

    @kb.add("c-m")  # Standard Enter key
    def _(event):
        buf = event.current_buffer
        text = buf.text

        # 1. Single-line slash commands submit immediately on single Enter
        if text.strip().startswith("/"):
            buf.validate_and_handle()
            return

        # 2. Double Enter: If cursor is at the end of the buffer and last line is empty, submit
        if buf.cursor_position == len(text) and text.endswith("\n"):
            buf.text = text.rstrip("\n")
            buf.validate_and_handle()
            return
        elif not text.strip():
            # Empty input on Enter
            buf.validate_and_handle()
            return

        # Otherwise insert a newline for native multiline editing
        buf.insert_text("\n")

    @kb.add("escape", "enter")  # Alt+Enter (POSIX terminal escape sequence for Meta+Enter)
    def _(event):
        """Immediately submit multiline buffer on Alt+Enter."""
        event.current_buffer.validate_and_handle()

    @kb.add("c-j")  # Ctrl+Enter / LineFeed
    def _(event):
        """Submit multiline buffer on Ctrl+Enter / Ctrl+J."""
        event.current_buffer.validate_and_handle()

    return kb


def prompt_continuation(width: int, line_number: int, is_soft_wrap: bool):
    """Render visually distinct prompt continuation indicator for multiline input."""
    return HTML("<ansibrightblack>  │ </ansibrightblack>")


class InteractiveCLI:
    """Rich interactive streaming terminal interface with prompt_toolkit multiline editing."""

    def __init__(self, assistant: AIAssistant):
        self.assistant = assistant
        self.console = Console()
        self.assistant.tool_confirmation_callback = self.prompt_tool_confirmation

        # Initialize prompt_toolkit session with native POSIX cursor handling and multiline support
        self.key_bindings = create_cli_keybindings()
        history_path = os.path.expanduser("~/.finch_history")
        try:
            self.history = FileHistory(history_path)
        except Exception:
            self.history = InMemoryHistory()

        self.prompt_session: PromptSession = PromptSession(
            multiline=True,
            key_bindings=self.key_bindings,
            history=self.history,
            prompt_continuation=prompt_continuation,
            enable_history_search=True,
        )

    async def prompt_tool_confirmation(self, tool_name: str, args: Dict[str, Any]) -> bool:
        """Prompt user for interactive confirmation before executing a tool in 'ask' mode."""
        self.console.print()
        arg_preview = self.assistant._format_tool_notice(tool_name, args)
        self.console.print(
            f"[bold yellow]⚠️  Tool Confirmation Requested:[/bold yellow] "
            f"The agent requests to run [bold cyan]{tool_name}[/bold cyan][white]{arg_preview}[/white]"
        )
        if isinstance(args, dict) and args:
            try:
                compact_args = json.dumps(args, indent=2)
                if len(compact_args) > 240:
                    compact_args = compact_args[:235] + "\n..."
                self.console.print(f"[dim]{compact_args}[/dim]")
            except Exception:
                pass

        try:
            conf_session = PromptSession(multiline=False)
            resp = await conf_session.prompt_async(HTML("<ansiyellow>➤ Allow execution? [y/N]: </ansiyellow>"))
            confirmed = resp.strip().lower() in ("y", "yes")
        except (KeyboardInterrupt, EOFError):
            confirmed = False

        if confirmed:
            self.console.print("[bold green]✓ Execution approved by user.[/bold green]\n")
        else:
            self.console.print("[bold red]✗ Execution cancelled by user.[/bold red]\n")
        return confirmed

    def print_welcome_banner(self) -> None:
        """Display an eye-catching welcome header with session parameters."""
        mem_mb = get_process_memory_mb()
        comp_active = self.assistant.compression_pipeline.enabled
        comp_status = "[bold green]Headroom (ON)[/bold green]" if comp_active else "[dim]OFF[/dim]"

        banner = Table.grid(padding=1)
        banner.add_column(style="cyan", justify="left")
        banner.add_column(style="bold white", justify="left")

        if self.assistant.provider.name == "Gemini":
            key_suffix = f"...{self.assistant.config.gemini_api_key[-6:]}" if self.assistant.config.gemini_api_key else "Not Set"
            provider_desc = f"Google Gemini (Key: {key_suffix})"
        else:
            provider_desc = f"{self.assistant.provider.name} ({self.assistant.config.ollama_host})"

        banner.add_row("Session ID:", f"[bold cyan]{self.assistant.current_session.id}[/bold cyan]")
        banner.add_row("Storage:", f"[white]{self.assistant.config.db_path}[/white] (SQLite)")
        banner.add_row("Provider:", provider_desc)
        banner.add_row("Active Model:", f"[bold green]{self.assistant.active_model}[/bold green]")
        mode_label = (
            f"[bold magenta]{self.assistant.mode.value.upper()}[/bold magenta] [dim](Conversational Dialogue)[/dim]"
            if self.assistant.mode == AssistantMode.CHAT
            else f"[bold magenta]{self.assistant.mode.value.upper()}[/bold magenta] [dim](Autonomous Tools)[/dim]"
        )
        banner.add_row("Mode:", mode_label)
        banner.add_row("Persona:", f"[yellow]{self.assistant.config.personality_path}[/yellow]")
        banner.add_row("Compression:", comp_status)
        banner.add_row("Sliding Window:", f"{self.assistant.memory.max_messages} messages (pinned system prompt)")
        banner.add_row("Process Memory:", f"{mem_mb} MB")

        panel = Panel(
            banner,
            title="[bold blue]🤖 Finch.ai Terminal[/bold blue]",
            subtitle="[dim]Type /help for commands, Alt+Enter or Double Enter to send, /exit to quit[/dim]",
            border_style="cyan",
            box=ROUNDED,
        )
        self.console.print(panel)

    def print_help(self) -> None:
        """Display command help table."""
        table = Table(title="Available Commands", box=ROUNDED, header_style="bold cyan")
        table.add_column("Command", style="yellow")
        table.add_column("Description", style="white")

        table.add_row("/summarize", "Generate a 2-sentence summary and title for the current session")
        table.add_row("/sessions [limit]", "List archived conversation sessions from SQLite")
        table.add_row("/session", "Display metadata and stats for the active session")
        table.add_row("/search <query>", "Exact-term & keyword search across past messages (FTS5)")
        table.add_row("/hybrid <query>", "Semantic & Hybrid Search combining FTS5 and sqlite-vec (RRF)")
        table.add_row("/transcript [id]", "View full dialogue transcript for an identified session")
        table.add_row("/new [title]", "Start a fresh session and clear in-memory context")
        table.add_row("/models", "List available models with metadata")
        table.add_row("/use <name>", "Switch active model (e.g. /use gemini-2.5-flash)")
        table.add_row("/mode [chat|agent]", "Toggle or set assistant mode (chat vs agent)")
        table.add_row("/tools [cat] [off|ask|auto]", "Tool policy: OFF, ASK (Confirm with User), or AUTO (Full Control)")
        table.add_row("/remember <fact>", "Store an enduring fact or preference in long-term memory")
        table.add_row("/forget <id>", "Remove a fact from long-term memory")
        table.add_row("/facts, /memory", "List all persistent facts stored in long-term memory")
        table.add_row("/compress [on|off|stats]", "Toggle or inspect Headroom context compression")
        table.add_row("/persona [reload|edit|adapt|wipe]", "View, reload, adapt, or reset personality.md")
        table.add_row("/wipe [current|all|persona|memory]", "Wipe conversation history (active or all), persona, or memory facts")
        table.add_row("/export [backup|md|html] [path]", "Export backup bundle (.zip), Markdown transcript, or HTML")
        table.add_row("/import <filepath> [mode]", "Restore conversation data and personality.md from backup")
        table.add_row("/buffer", "Inspect current sliding-window message buffer")
        table.add_row("/clear [all|history]", "Clear in-memory buffer, or wipe SQLite conversation history")
        table.add_row("/vacuum, /maintenance", "Run SQLite housekeeping (checkpoint, optimize, VACUUM)")
        table.add_row("/storage", "Inspect database disk footprint, page count, and WAL size")
        table.add_row("/archive [days]", "Archive inactive sessions older than N days to archive DB")
        table.add_row("/system [text]", "View or update the pinned system prompt directly")
        table.add_row("/help", "Show this help table")
        table.add_row("/exit, /quit", "Exit session (auto-summarizes unless empty)")
        self.console.print(table)

    async def handle_models_command(self) -> None:
        """Fetch and display available models."""
        prov_name = self.assistant.provider.name
        with self.console.status(f"[bold green]Fetching models from {prov_name}..."):
            try:
                models = await self.assistant.list_available_models()
            except Exception as e:
                self.console.print(f"[bold red]Error fetching models:[/bold red] {e}")
                return

        if not models:
            self.console.print(f"[yellow]No models found in {prov_name}.[/yellow]")
            return

        table = Table(title=f"Available Models in {prov_name}", box=ROUNDED, header_style="bold cyan")
        table.add_column("Status", justify="center", width=8)
        table.add_column("Model Name", style="bold white")
        table.add_column("Size", justify="right")
        table.add_column("Family")
        table.add_column("Params", justify="right")
        table.add_column("Quant")

        for m in models:
            is_active = m.name == self.assistant.active_model
            status = "[bold green]ACTIVE[/bold green]" if is_active else ""
            table.add_row(
                status,
                f"[green]{m.name}[/green]" if is_active else m.name,
                f"{m.size_gb} GB",
                m.family or "-",
                m.parameter_size or "-",
                m.quantization or "-",
            )

        self.console.print(table)
        self.console.print(f"[dim]Use [bold cyan]/use <model>[/bold cyan] to switch.[/dim]\n")

    async def handle_persona_command(self, arg: str) -> None:
        """View, reload, adapt, or reset personality.md."""
        arg = arg.strip().lower()
        if arg == "reload":
            content = self.assistant.reload_persona()
            self.console.print(
                f"[bold green]✓ Successfully reloaded {self.assistant.config.personality_path}![/bold green]\n"
            )
        elif arg == "edit":
            abs_path = os.path.abspath(self.assistant.config.personality_path)
            self.console.print(
                f"[cyan]You can edit the personality file directly in your editor:[/cyan]\n[bold white]{abs_path}[/bold white]\n"
                "[dim]After saving, run [bold cyan]/persona reload[/bold cyan] to apply changes without restarting.[/dim]\n"
            )
        elif arg in ("wipe", "reset"):
            self.handle_wipe_command("persona")
        elif arg in ("adapt", "evolve"):
            await self.handle_persona_adapt_command()
        else:
            content = self.assistant.persona_manager.load_persona()
            panel = Panel(
                Markdown(content),
                title=f"[bold yellow]Active Persona ({self.assistant.config.personality_path})[/bold yellow]",
                border_style="yellow",
                box=ROUNDED,
            )
            self.console.print(panel)
            self.console.print(
                "[dim]Commands: [bold cyan]/persona reload[/bold cyan] | "
                "[bold cyan]/persona edit[/bold cyan] | "
                "[bold cyan]/persona adapt[/bold cyan] | "
                "[bold cyan]/persona wipe[/bold cyan][/dim]\n"
            )

    async def handle_persona_adapt_command(self) -> None:
        """Analyze recent conversation summaries and adapt personality.md."""
        with self.console.status("[bold cyan]Analyzing conversations to adapt personality...[/bold cyan]"):
            success, msg = await self.assistant.adapt_persona()

        if success:
            self.console.print(f"[bold green]✓ {msg}[/bold green]\n")
            content = self.assistant.persona_manager.load_persona()
            panel = Panel(
                Markdown(content),
                title=f"[bold green]Updated Persona ({self.assistant.config.personality_path})[/bold green]",
                border_style="green",
                box=ROUNDED,
            )
            self.console.print(panel)
            self.console.print()
        else:
            self.console.print(f"[bold yellow]Persona adaptation notice:[/bold yellow] {msg}\n")

    def handle_wipe_command(self, arg: str) -> None:
        """Wipe conversation history, personality.md, or long-term memory facts."""
        raw_parts = arg.strip().split()
        if not raw_parts:
            self.console.print(
                "[bold yellow]⚠️  Wipe Options:[/bold yellow]\n"
                "  [1] [cyan]Current conversation[/cyan] - Wipe active session messages from DB & memory\n"
                "  [2] [red]All conversations[/red]    - Wipe ALL conversation sessions & messages from DB\n"
                "  [3] [magenta]Personality[/magenta]          - Reset personality.md to default template (backed up to .bak)\n"
                "  [4] [blue]Memory facts[/blue]         - Clear all stored user facts & preferences\n"
                "  [5] Cancel\n"
            )
            try:
                conf_session = PromptSession(multiline=False)
                choice = conf_session.prompt("Select an option [1-5]: ").strip()
            except (KeyboardInterrupt, EOFError):
                self.console.print("\n[dim]Wipe cancelled.[/dim]\n")
                return
            if choice == "1":
                parts = ["conversation", "current"]
            elif choice == "2":
                parts = ["conversation", "all"]
            elif choice == "3":
                parts = ["persona"]
            elif choice == "4":
                parts = ["memory"]
            else:
                self.console.print("[dim]Wipe cancelled.[/dim]\n")
                return
        else:
            parts = raw_parts

        target = parts[0].lower()

        # Handle direct shorthand: /wipe all, /wipe current, /wipe session
        if target in ("all", "everything", "full"):
            target = "conversation"
            parts = ["conversation", "all"]
        elif target in ("current", "active", "session"):
            target = "conversation"
            parts = ["conversation", "current"]

        if target in ("persona", "personality"):
            content = self.assistant.wipe_persona()
            self.console.print(
                f"[bold green]✓ Successfully wiped {self.assistant.config.personality_path}![/bold green]\n"
                f"[dim]Reset to default template. Previous version saved to {self.assistant.config.personality_path}.bak[/dim]\n"
            )
        elif target in ("conversation", "conversations", "conv", "history", "chat", "messages", "dialogue"):
            scope = parts[1].lower() if len(parts) > 1 else "current"
            all_sessions = scope in ("all", "full", "everything")
            stats = self.assistant.wipe_conversation(all_sessions=all_sessions)

            if all_sessions:
                self.console.print(
                    f"[bold green]✓ Successfully wiped ALL conversation history![/bold green]\n"
                    f"Cleared [bold white]{stats.get('sessions_wiped', 0)}[/bold white] sessions and "
                    f"[bold white]{stats.get('messages_wiped', 0)}[/bold white] messages from SQLite.\n"
                    f"Started fresh session: [cyan]{self.assistant.current_session.id}[/cyan]\n"
                )
            else:
                self.console.print(
                    f"[bold green]✓ Successfully wiped conversation messages for active session![/bold green]\n"
                    f"Removed [bold white]{stats.get('messages_wiped', 0)}[/bold white] message(s). In-memory buffer cleared.\n"
                )
        elif target in ("memory", "facts", "fact"):
            count = self.assistant.wipe_user_facts()
            self.console.print(
                f"[bold green]✓ Successfully wiped all user facts from long-term memory![/bold green]\n"
                f"Removed [bold white]{count}[/bold white] fact(s).\n"
            )
        else:
            self.console.print(
                "[yellow]Usage: /wipe [target][/yellow]\n"
                "  [bold cyan]/wipe[/bold cyan]                         - Open interactive wipe menu\n"
                "  [bold cyan]/wipe conversation [current|all][/bold cyan] - Wipe active session or all conversation history\n"
                "  [bold cyan]/wipe current[/bold cyan]                 - Wipe active conversation messages\n"
                "  [bold cyan]/wipe all[/bold cyan]                     - Wipe ALL conversation sessions & messages from DB\n"
                "  [bold cyan]/wipe persona[/bold cyan]                 - Reset personality.md to default template (backs up to .bak)\n"
                "  [bold cyan]/wipe memory[/bold cyan]                  - Clear all stored user facts and preferences\n"
            )

    def handle_remember_command(self, arg: str) -> None:
        """Store a fact or preference in long-term memory."""
        fact = arg.strip()
        if not fact:
            self.console.print("[yellow]Usage: /remember <fact or preference to store>[/yellow]\n")
            return
        fact_id = self.assistant.add_user_fact(fact=fact)
        self.console.print(
            f"[bold green]✓ Stored in long-term memory (ID #{fact_id}):[/bold green] {fact}\n"
            "[dim]This fact is now pinned in system memory across all sessions.[/dim]\n"
        )

    def handle_forget_command(self, arg: str) -> None:
        """Remove a fact from long-term memory by ID."""
        if not arg.strip():
            self.console.print("[yellow]Usage: /forget <fact_id>[/yellow]\n")
            return
        try:
            fid = int(arg.strip())
        except ValueError:
            self.console.print("[bold red]Error: Fact ID must be an integer.[/bold red]\n")
            return

        deleted = self.assistant.delete_user_fact(fid)
        if deleted:
            self.console.print(f"[bold green]✓ Fact #{fid} removed from long-term memory.[/bold green]\n")
        else:
            self.console.print(f"[yellow]Fact #{fid} not found in memory.[/yellow]\n")

    def handle_facts_command(self, arg: str) -> None:
        """Display all stored facts in long-term memory."""
        facts = self.assistant.list_user_facts()
        if not facts:
            self.console.print("[yellow]No facts stored in long-term memory yet. Use /remember <fact> to add one.[/yellow]\n")
            return

        table = Table(title=f"🧠 Long-Term Memory Facts ({len(facts)} stored)", box=ROUNDED, header_style="bold cyan")
        table.add_column("ID", justify="center", style="bold yellow", width=6)
        table.add_column("Category", style="cyan", width=14)
        table.add_column("Fact / Preference", style="bold white")
        table.add_column("Recorded", style="dim", width=19)

        for f in facts:
            table.add_row(
                str(f["id"]),
                f.get("category", "general"),
                f["fact"],
                f.get("created_at", "")[:19].replace("T", " "),
            )

        self.console.print(table)
        self.console.print("[dim]Use [bold cyan]/remember <fact>[/bold cyan] to add or [bold cyan]/forget <id>[/bold cyan] to delete.[/dim]\n")

    def handle_export_command(self, arg: str) -> None:
        """Export conversation data, persona, or readable transcripts (Markdown / HTML)."""
        parts = arg.strip().split(maxsplit=2)
        subcmd = parts[0].lower() if parts else ""

        if subcmd in ("md", "markdown"):
            target_id = parts[1] if len(parts) > 1 and not parts[1].endswith(".md") else None
            out_file = parts[2] if len(parts) > 2 else (parts[1] if len(parts) > 1 and parts[1].endswith(".md") else None)
            with self.console.status("[bold cyan]Exporting Markdown transcript...[/bold cyan]"):
                try:
                    path = self.assistant.export_transcript_markdown(session_id=target_id, output_path=out_file)
                    size = os.path.getsize(path)
                except Exception as e:
                    self.console.print(f"[bold red]Markdown export failed:[/bold red] {e}\n")
                    return
            self.console.print(
                f"[bold green]✓ Markdown Transcript Exported:[/bold green] [bold cyan]{path}[/bold cyan] ({size:,} bytes)\n"
            )
            return

        elif subcmd in ("html", "web"):
            target_id = parts[1] if len(parts) > 1 and not parts[1].endswith(".html") else None
            out_file = parts[2] if len(parts) > 2 else (parts[1] if len(parts) > 1 and parts[1].endswith(".html") else None)
            with self.console.status("[bold cyan]Exporting HTML transcript...[/bold cyan]"):
                try:
                    path = self.assistant.export_transcript_html(session_id=target_id, output_path=out_file)
                    size = os.path.getsize(path)
                except Exception as e:
                    self.console.print(f"[bold red]HTML export failed:[/bold red] {e}\n")
                    return
            self.console.print(
                f"[bold green]✓ Standalone HTML Transcript Exported:[/bold green] [bold cyan]{path}[/bold cyan] ({size:,} bytes)\n"
                f"[dim]You can open this file in any web browser to view the dialogue transcript.[/dim]\n"
            )
            return

        # Default: Full backup export bundle (.zip)
        target_path = arg.strip() if arg.strip() and subcmd not in ("backup", "zip", "all") else (parts[1] if len(parts) > 1 else None)
        with self.console.status("[bold cyan]Creating backup bundle (conversations + personality + facts)...[/bold cyan]"):
            try:
                zip_path = self.assistant.export_backup(output_path=target_path)
                file_size = os.path.getsize(zip_path)
            except Exception as e:
                self.console.print(f"[bold red]Backup export failed:[/bold red] {e}\n")
                return

        grid = Table.grid(padding=1)
        grid.add_column(style="cyan", justify="left")
        grid.add_column(style="bold white", justify="left")

        grid.add_row("Backup Bundle:", zip_path)
        grid.add_row("Bundle Size:", f"{file_size:,} bytes ({file_size/1024:.1f} KB)")
        grid.add_row("Contents:", "conversations.db, conversations.json, personality.md, user_facts.json, manifest.json")
        grid.add_row("Active Persona:", self.assistant.config.personality_path)

        panel = Panel(
            grid,
            title="[bold green]📦 Backup Bundle Exported Successfully[/bold green]",
            border_style="green",
            box=ROUNDED,
        )
        self.console.print(panel)
        self.console.print(f"[dim]To restore this backup later, run: [bold cyan]/import {zip_path}[/bold cyan][/dim]\n")

    def handle_import_command(self, arg: str) -> None:
        """Restore conversation data and personality.md from a backup zip bundle."""
        parts = arg.strip().split()
        if not parts:
            self.console.print("[yellow]Usage: /import <backup_file.zip> [replace|merge][/yellow]\n")
            return

        file_path = parts[0]
        mode = parts[1].lower() if len(parts) > 1 else "replace"
        if mode not in ("replace", "merge"):
            mode = "replace"

        with self.console.status(f"[bold cyan]Restoring from backup '{file_path}' ({mode} mode)...[/bold cyan]"):
            try:
                res = self.assistant.import_backup(backup_path=file_path, mode=mode)
            except Exception as e:
                self.console.print(f"[bold red]Backup import failed:[/bold red] {e}\n")
                return

        grid = Table.grid(padding=1)
        grid.add_column(style="cyan", justify="left")
        grid.add_column(style="bold white", justify="left")

        grid.add_row("Backup Source:", res.get("backup_path", file_path))
        grid.add_row("Restore Mode:", mode.upper())
        grid.add_row("Sessions Restored:", str(res.get("sessions_restored", 0)))
        grid.add_row("Messages Restored:", str(res.get("messages_restored", 0)))
        grid.add_row("Personality Restored:", "Yes (live updated)" if res.get("persona_restored") else "No")
        grid.add_row("Current Active Session:", self.assistant.current_session.id)

        panel = Panel(
            grid,
            title="[bold green]🔄 Backup Restored Successfully[/bold green]",
            border_style="green",
            box=ROUNDED,
        )
        self.console.print(panel)
        self.console.print()

    def handle_compression_command(self, arg: str) -> None:
        """Toggle or inspect Headroom context compression pipeline."""
        cmd = arg.strip().lower()
        if cmd == "on":
            active = self.assistant.compression_pipeline.toggle(True)
            if active:
                self.console.print("[bold green]✓ Headroom Context Compression is now ENABLED.[/bold green]\n")
            else:
                self.console.print("[bold red]Cannot enable Headroom: package not installed.[/bold red]\n")
        elif cmd == "off":
            self.assistant.compression_pipeline.toggle(False)
            self.console.print("[yellow]Headroom Context Compression is now DISABLED.[/yellow]\n")
        else:
            # Show summary stats table
            summary = self.assistant.compression_pipeline.get_summary()
            status_text = "[bold green]ENABLED[/bold green]" if summary["enabled"] else "[yellow]DISABLED[/yellow]"

            table = Table(title="Headroom Compression Pipeline Telemetry", box=ROUNDED)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="bold white")

            table.add_row("Status", status_text)
            table.add_row("Total Passes Run", str(summary["total_compressions"]))
            table.add_row("Total Tokens Evaluated", str(summary["total_tokens_before"]))
            table.add_row("Total Tokens Sent", str(summary["total_tokens_after"]))
            table.add_row("Total Tokens Saved", f"[bold green]{summary['total_tokens_saved']}[/bold green]")
            table.add_row("Cumulative Savings %", f"[bold green]{summary['overall_savings_pct']}%[/bold green]")
            table.add_row("Protected System Persona", "Yes (personality.md byte-exact)")
            table.add_row("Protected Recent Turns", f"{summary['protect_recent']} turns")

            self.console.print(table)
            self.console.print("[dim]Use [bold cyan]/compress on[/bold cyan] or [bold cyan]/compress off[/bold cyan] to toggle.[/dim]\n")

    def handle_buffer_command(self) -> None:
        """Display current sliding-window buffer state."""
        stats = self.assistant.memory.get_stats()
        messages = self.assistant.memory.get_messages()

        table = Table(title=f"Message Buffer ({stats['active_messages']}/{stats['window_size']} active messages)", box=ROUNDED)
        table.add_column("#", justify="center", width=4)
        table.add_column("Role", width=12)
        table.add_column("Length", justify="right", width=10)
        table.add_column("Content Snippet", style="dim")

        for idx, msg in enumerate(messages, 1):
            role_style = "blue" if msg.role.value == "system" else ("green" if msg.role.value == "user" else "magenta")
            snippet = msg.content.replace("\n", " ")
            if len(snippet) > 80:
                snippet = snippet[:77] + "..."
            table.add_row(
                str(idx),
                f"[{role_style}]{msg.role.value.upper()}[/{role_style}]",
                f"{len(msg.content)} chars",
                snippet,
            )

        self.console.print(table)
        self.console.print(
            f"[dim]Approx context tokens: ~{stats['approx_tokens']} | Total chars: {stats['total_characters']} | Memory: {get_process_memory_mb()} MB[/dim]\n"
        )

    def print_telemetry_bar(
        self,
        stats: Optional[StreamStats],
        comp_stats: Optional[CompressionStats] = None,
    ) -> None:
        """Render performance, memory overhead, and Headroom compression statistics."""
        mem_mb = get_process_memory_mb()
        buf_stats = self.assistant.memory.get_stats()

        if stats:
            tps_display = f"{stats.tokens_per_second:.1f} t/s" if stats.tokens_per_second > 0 else f"{stats.client_tokens_per_second:.1f} t/s"
            eval_time_sec = f"{stats.eval_duration_ms / 1000.0:.2f}s" if stats.eval_duration_ms > 0 else f"{stats.total_duration_ms / 1000.0:.2f}s"

            telemetry = Text()
            telemetry.append("⚡ ", style="yellow")
            telemetry.append(f"Speed: {tps_display} ", style="bold green")
            telemetry.append("• ", style="dim")
            telemetry.append(f"TTFT: {stats.ttft_ms:.0f}ms ", style="cyan")
            telemetry.append("• ", style="dim")
            telemetry.append(f"Tokens: {stats.eval_count} ", style="white")
            telemetry.append("• ", style="dim")
            telemetry.append(f"Eval Time: {eval_time_sec} ", style="dim white")
            telemetry.append("• ", style="dim")
            telemetry.append(f"RAM: {mem_mb} MB ", style="magenta")
            telemetry.append("• ", style="dim")
            telemetry.append(f"Buffer: {buf_stats['active_messages']}/{buf_stats['window_size']} msgs", style="blue")

            self.console.print(telemetry)

        # Print Headroom Compression Stats if available
        if comp_stats and comp_stats.tokens_before > 0:
            comp_text = Text()
            comp_text.append("🗜️  Headroom: ", style="bold cyan")
            if comp_stats.tokens_saved > 0:
                comp_text.append(
                    f"{comp_stats.tokens_before} → {comp_stats.tokens_after} tokens ",
                    style="bold white",
                )
                comp_text.append(
                    f"(saved {comp_stats.tokens_saved}, {comp_stats.savings_pct}%) ",
                    style="bold green",
                )
            else:
                comp_text.append(
                    f"{comp_stats.tokens_before} tokens (0 saved / below threshold) ",
                    style="dim",
                )

            comp_text.append("• ", style="dim")
            comp_text.append(f"{comp_stats.duration_ms:.1f}ms", style="dim cyan")

            if comp_stats.transforms_applied:
                transforms_str = ", ".join(comp_stats.transforms_applied[:2])
                comp_text.append(" • ", style="dim")
                comp_text.append(f"[{transforms_str}]", style="dim yellow")

            self.console.print(comp_text)

        self.console.print()

    async def handle_summarize_command(self) -> None:
        """Generate and display summary and title for current session."""
        with self.console.status("[bold cyan]Generating session summary...[/bold cyan]"):
            try:
                title, summary = await self.assistant.summarize_session()
            except Exception as e:
                self.console.print(f"[bold red]Error generating summary:[/bold red] {e}\n")
                return

        panel = Panel(
            f"[bold white]{title}[/bold white]\n\n[cyan]{summary}[/cyan]",
            title=f"[bold green]📝 Session Summary ({self.assistant.current_session.id})[/bold green]",
            border_style="green",
            box=ROUNDED,
        )
        self.console.print(panel)
        self.console.print()

    def handle_sessions_command(self, arg: str) -> None:
        """List past archived sessions from SQLite."""
        limit = 20
        if arg:
            try:
                limit = int(arg.strip())
            except ValueError:
                limit = 20

        sessions = self.assistant.list_sessions(limit=limit)
        if not sessions:
            self.console.print("[yellow]No archived sessions found in SQLite.[/yellow]\n")
            return

        table = Table(title=f"Archived Sessions (most recent {len(sessions)})", box=ROUNDED, header_style="bold cyan")
        table.add_column("Active", justify="center", width=6)
        table.add_column("Session ID", style="bold yellow")
        table.add_column("Title", style="bold white")
        table.add_column("Messages", justify="right", width=8)
        table.add_column("Created", style="dim", width=19)
        table.add_column("Summary Snippet", style="dim cyan")

        for s in sessions:
            is_active = s.id == self.assistant.current_session.id
            active_str = "[bold green]▶[/bold green]" if is_active else ""
            summary_snippet = (s.summary or "-").replace("\n", " ")
            if len(summary_snippet) > 60:
                summary_snippet = summary_snippet[:57] + "..."
            created_short = s.created_at[:19].replace("T", " ")

            table.add_row(
                active_str,
                f"[green]{s.id}[/green]" if is_active else s.id,
                s.title,
                str(s.message_count),
                created_short,
                summary_snippet,
            )

        self.console.print(table)
        self.console.print("[dim]Use [bold cyan]/session[/bold cyan] to inspect the current session.[/dim]\n")

    def handle_session_info_command(self) -> None:
        """Show detailed metadata for current session."""
        sess = self.assistant.get_current_session()
        if not sess:
            self.console.print("[yellow]No active session found.[/yellow]\n")
            return

        grid = Table.grid(padding=1)
        grid.add_column(style="cyan", justify="left")
        grid.add_column(style="bold white", justify="left")

        grid.add_row("Session ID:", sess.id)
        grid.add_row("Title:", sess.title)
        grid.add_row("Model:", sess.model)
        grid.add_row("Mode:", sess.mode)
        grid.add_row("Total Messages:", str(sess.message_count))
        grid.add_row("Created At:", sess.created_at)
        grid.add_row("Updated At:", sess.updated_at)
        grid.add_row("Summary:", sess.summary or "[italic dim]Not yet generated (run /summarize)[/italic dim]")

        panel = Panel(
            grid,
            title="[bold cyan]Active Session Details[/bold cyan]",
            border_style="cyan",
            box=ROUNDED,
        )
        self.console.print(panel)
        self.console.print()

    def handle_new_session_command(self, arg: str) -> None:
        """Start a fresh session and clear context buffer."""
        title = arg.strip() if arg else "New Session"
        sess = self.assistant.new_session(title=title)
        self.console.print(
            f"[bold green]✓ Started new session:[/bold green] [bold cyan]{sess.id}[/bold cyan] ('{sess.title}')\n"
        )

    def handle_search_command(self, query: str) -> None:
        """Search past conversation messages using SQLite FTS5 lexical retrieval."""
        if not query.strip():
            self.console.print("[yellow]Usage: /search <exact term, date, code snippet, or phrase>[/yellow]\n")
            return

        with self.console.status(f"[bold cyan]Searching messages for '{query}'...[/bold cyan]"):
            results = self.assistant.search_keyword(query=query, limit=10)

        if not results:
            self.console.print(f"[yellow]No matches found for '{query}'.[/yellow]\n")
            return

        table = Table(
            title=f"Search Results for '{query}' ({len(results)} matches)",
            box=ROUNDED,
            header_style="bold cyan",
        )
        table.add_column("Session ID", style="cyan", width=16)
        table.add_column("Session Title", style="bold white", width=22)
        table.add_column("Role", justify="center", width=8)
        table.add_column("Timestamp", style="dim", width=19)
        table.add_column("Matched Snippet", style="white")

        for r in results:
            role_style = "green" if r.role.lower() == "user" else "magenta"
            time_short = r.timestamp[:19].replace("T", " ")
            clean_snippet = r.snippet.replace("[MATCH]", "[bold yellow]").replace("[/MATCH]", "[/bold yellow]")
            table.add_row(
                r.session_id,
                r.session_title,
                f"[{role_style}]{r.role.upper()}[/{role_style}]",
                time_short,
                clean_snippet,
            )

        self.console.print(table)
        self.console.print("[dim]Use [bold cyan]/transcript <session_id>[/bold cyan] to view full dialogue.[/dim]\n")

    async def handle_hybrid_command(self, query: str) -> None:
        """Execute Hybrid Search combining FTS5 lexical matching and sqlite-vec embeddings via RRF."""
        if not query.strip():
            self.console.print("[yellow]Usage: /hybrid <conceptual or keyword query>[/yellow]\n")
            return

        with self.console.status(f"[bold cyan]Performing hybrid RRF search for '{query}'...[/bold cyan]"):
            results = await self.assistant.search_hybrid(query=query, limit=10)

        if not results:
            self.console.print(f"[yellow]No hybrid matches found for '{query}'.[/yellow]\n")
            return

        table = Table(
            title=f"Hybrid Search Results for '{query}' (RRF Fusion, {len(results)} matches)",
            box=ROUNDED,
            header_style="bold cyan",
        )
        table.add_column("Session ID", style="cyan", width=16)
        table.add_column("Session Title", style="bold white", width=22)
        table.add_column("RRF Score", justify="right", style="bold green", width=11)
        table.add_column("Lex Rank", justify="center", width=10)
        table.add_column("Sem Sim", justify="right", width=10)
        table.add_column("Summary / Snippet", style="white")

        for r in results:
            lex_str = f"#{r.lexical_rank}" if r.lexical_rank is not None else "[dim]-[/dim]"
            sim_str = f"{r.cosine_similarity * 100:.1f}%" if r.cosine_similarity is not None else "[dim]-[/dim]"
            preview = ""
            if r.matched_snippets:
                clean_snip = r.matched_snippets[0].replace("[MATCH]", "[bold yellow]").replace("[/MATCH]", "[/bold yellow]")
                preview = clean_snip.replace("\n", " ")
            elif r.summary:
                preview = f"[dim italic]{r.summary}[/dim italic]"
            else:
                preview = "[dim](No summary or snippet)[/dim]"

            if len(preview) > 90:
                preview = preview[:87] + "..."

            table.add_row(
                r.session_id,
                r.title,
                f"{r.rrf_score:.5f}",
                lex_str,
                sim_str,
                preview,
            )

        self.console.print(table)
        self.console.print("[dim]Use [bold cyan]/transcript <session_id>[/bold cyan] to inspect full session dialogue.[/dim]\n")

    def handle_transcript_command(self, session_id: str) -> None:
        """Display full dialogue transcript for an identified session."""
        target_id = session_id.strip() or self.assistant.current_session.id
        transcript = self.assistant.load_session_transcript(session_id=target_id)
        if not transcript:
            self.console.print(f"[bold red]Session '{target_id}' not found.[/bold red]\n")
            return

        self.console.print(
            Panel(
                transcript.formatted_transcript,
                title=f"[bold cyan]Transcript: {transcript.title} ({transcript.session_id})[/bold cyan]",
                border_style="cyan",
                box=ROUNDED,
            )
        )
    def handle_maintenance_command(self) -> None:
        """Execute database maintenance (WAL checkpoint, integrity check, PRAGMA optimize, VACUUM)."""
        with self.console.status("[bold cyan]Running SQLite maintenance and vacuum...[/bold cyan]"):
            res = self.assistant.run_maintenance(vacuum=True)

        stats = res.get("storage_stats", {})
        grid = Table.grid(padding=1)
        grid.add_column(style="cyan", justify="left")
        grid.add_column(style="bold white", justify="left")

        grid.add_row("Database File:", stats.get("db_path", ""))
        grid.add_row("Integrity Check:", f"[bold green]{res.get('integrity', 'ok')}[/bold green]")
        grid.add_row("WAL Checkpoint:", f"log={res.get('checkpoint', {}).get('log', 0)}, checkpointed={res.get('checkpoint', {}).get('checkpointed', 0)}")
        grid.add_row("Size Before:", f"{res.get('total_bytes_before', 0):,} bytes")
        grid.add_row("Size After:", f"{res.get('total_bytes_after', 0):,} bytes")
        grid.add_row("Space Reclaimed:", f"[bold green]{res.get('bytes_reclaimed', 0):,} bytes[/bold green]")
        grid.add_row("Total Sessions:", str(stats.get("session_count", 0)))
        grid.add_row("Total Messages:", str(stats.get("message_count", 0)))

        panel = Panel(
            grid,
            title="[bold green]🛠️ Database Maintenance & VACUUM Complete[/bold green]",
            border_style="green",
            box=ROUNDED,
        )
        self.console.print(panel)
        self.console.print()

    def handle_storage_command(self) -> None:
        """Display database disk usage, pages, and record statistics."""
        stats = self.assistant.get_storage_stats()

        grid = Table.grid(padding=1)
        grid.add_column(style="cyan", justify="left")
        grid.add_column(style="bold white", justify="left")

        file_bytes = stats.get("file_size_bytes", 0)
        grid.add_row("Database Path:", stats.get("db_path", ""))
        grid.add_row("DB File Size:", f"{file_bytes:,} bytes ({file_bytes/1024:.1f} KB)")
        grid.add_row("WAL File Size:", f"{stats.get('wal_size_bytes', 0):,} bytes")
        grid.add_row("Page Size:", f"{stats.get('page_size', 0)} bytes")
        grid.add_row("Page Count:", f"{stats.get('page_count', 0)}")
        grid.add_row("Freelist Pages:", f"{stats.get('freelist_count', 0)} ({stats.get('reclaimable_bytes', 0):,} reclaimable bytes)")
        grid.add_row("Sessions Stored:", str(stats.get("session_count", 0)))
        grid.add_row("Messages Stored:", str(stats.get("message_count", 0)))
        if stats.get("vec_entry_count"):
            grid.add_row("Vector Embeddings:", str(stats.get("vec_entry_count", 0)))

        panel = Panel(
            grid,
            title="[bold cyan]📊 Storage & Database Statistics[/bold cyan]",
            border_style="cyan",
            box=ROUNDED,
        )
        self.console.print(panel)
        self.console.print()

    def handle_archive_command(self, arg: str) -> None:
        """Archive older sessions to an archive database file and vacuum main database."""
        days = 30
        if arg:
            try:
                days = int(arg.strip())
            except ValueError:
                days = 30

        with self.console.status(f"[bold cyan]Archiving sessions older than {days} days...[/bold cyan]"):
            res = self.assistant.archive_sessions(older_than_days=days)

        if res["archived_sessions"] == 0:
            self.console.print(f"[yellow]No sessions found older than {days} days to archive.[/yellow]\n")
            return

        self.console.print(
            f"[bold green]✓ Successfully archived {res['archived_sessions']} session(s) and {res['archived_messages']} message(s)![/bold green]\n"
            f"[cyan]Archive Database:[/cyan] {res['archive_db_path']}\n"
            f"[green]Space Reclaimed:[/green] {res['bytes_reclaimed']:,} bytes\n"
        )

    def handle_tools_command(self, arg: str) -> None:
        """Inspect or toggle 3-state agent tool permissions: OFF, ASK (Confirm), AUTO (Complete AI Control)."""
        parts = arg.strip().split()
        if not parts:
            perms = self.assistant.get_tool_permissions()
            table = Table(
                title=f"Agent Mode Tool Permissions (Mode: {self.assistant.mode.value.upper()})",
                box=ROUNDED,
                header_style="bold cyan",
            )
            table.add_column("Category", style="bold white", width=12)
            table.add_column("Status / Policy", justify="center", width=26)
            table.add_column("Tools Included", style="cyan", width=34)
            table.add_column("Description", style="dim")
            table.add_column("Cycle Command", style="yellow")

            meta = [
                ("terminal", "run_terminal_command", "Local shell execution, git, command-line inspection"),
                ("python", "python_interpreter", "Isolated Python REPL, math, data processing, algorithms"),
                ("web", "web_search, fetch_web_page", "Live web search and text content extraction"),
                ("files", "read_file, write_file, list_directory", "Workspace file system reading, writing, and listing"),
            ]

            has_auto = False
            for cat, tools, desc in meta:
                mode = perms.get(cat, "off")
                if mode == "auto":
                    status = "[bold yellow]⚡ AUTO (Complete AI Control)[/bold yellow]"
                    next_mode = "off"
                    has_auto = True
                elif mode == "ask":
                    status = "[bold cyan]🛡️  ASK (Confirm with User)[/bold cyan]"
                    next_mode = "auto"
                else:
                    status = "[bold red]🚫 OFF (Disabled)[/bold red]"
                    next_mode = "ask"

                toggle_cmd = f"/tools {cat} {next_mode}"
                table.add_row(cat.capitalize(), status, tools, desc, toggle_cmd)

            self.console.print(table)
            if has_auto:
                self.console.print(
                    "[bold yellow]⚠️  Notice: One or more tool categories are in Complete AI Control (AUTO) mode without human confirmation.[/bold yellow]"
                )
            self.console.print("[dim]Use [bold cyan]/tools <category> [off|ask|auto][/bold cyan] to change permissions.[/dim]\n")
            return

        cat = parts[0].lower()
        if cat not in ("terminal", "python", "web", "files"):
            self.console.print(f"[bold red]Unknown tool category '{cat}'.[/bold red] Choose from: terminal, python, web, files.\n")
            return

        action = parts[1].lower() if len(parts) > 1 else "toggle"
        if action in ("toggle",):
            new_mode = self.assistant.toggle_tool_permission(cat)
        else:
            new_mode = self.assistant.set_tool_permission(cat, action)

        if new_mode == "auto":
            self.console.print(f"[bold yellow]✓ {cat.capitalize()} access is now ENABLED (AUTO - Complete AI Control).[/bold yellow]")
            self.console.print(
                Panel(
                    f"[bold yellow]⚠️  WARNING: Complete AI Control enabled for '{cat.upper()}' tools![/bold yellow]\n"
                    "The AI can execute commands, run code, or modify files autonomously without human confirmation.\n"
                    f"[italic dim]If you prefer reviewing actions before execution, switch to ask mode: /tools {cat} ask[/italic dim]",
                    border_style="yellow",
                    box=ROUNDED,
                )
            )
            self.console.print()
        elif new_mode == "ask":
            self.console.print(
                f"[bold cyan]✓ {cat.capitalize()} access is now ENABLED (ASK - Human in the Loop confirmation).[/bold cyan]\n"
                "[dim]Finch.ai will ask for your confirmation before running each tool.[/dim]\n"
            )
        else:
            self.console.print(f"[bold red]✓ {cat.capitalize()} access is now DISABLED (OFF).[/bold red]\n")

    async def _cleanup_and_exit(self) -> None:
        """Run session exit hooks (fast background summarization) and cleanly close database."""
        if self.assistant.config.auto_summarize_on_exit:
            try:
                msgs = self.assistant.archive.get_messages(
                    self.assistant.current_session.id, include_system=False
                )
                if any(m.role.lower() == "user" for m in msgs):
                    with self.console.status("[bold cyan]Summarizing session before exit...[/bold cyan]"):
                        title, summary = await self.assistant.summarize_session()
                    self.console.print(
                        f"[bold green]✓ Session Archived:[/bold green] [bold white]{title}[/bold white]\n"
                        f"[dim cyan]{summary}[/dim cyan]\n"
                    )
            except Exception as e:
                self.console.print(f"[dim yellow]Notice: Exit summarization skipped: {e}[/dim yellow]")

        if getattr(self.assistant.config, "auto_adapt_persona", False):
            try:
                msgs = self.assistant.archive.get_messages(
                    self.assistant.current_session.id, include_system=False
                )
                if any(m.role.lower() == "user" for m in msgs):
                    with self.console.status("[bold cyan]Adapting personality from session...[/bold cyan]"):
                        adapted, msg = await self.assistant.adapt_persona()
                    if adapted:
                        self.console.print(f"[bold green]✦ Persona Evolved:[/bold green] {msg}\n")
            except Exception as e:
                self.console.print(f"[dim yellow]Notice: Persona adaptation skipped: {e}[/dim yellow]")

        self.assistant.close()
        self.console.print("[yellow]Assistant session closed. Goodbye![/yellow]")

    async def handle_input(self, text: str) -> bool:
        """Process user input. Return False if the session should exit."""
        cmd = text.strip()
        if not cmd:
            return True

        # Slash command dispatch
        if cmd.startswith("/"):
            parts = cmd.split(maxsplit=1)
            command = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""

            if command in ("/exit", "/quit"):
                return False

            elif command == "/help":
                self.print_help()
                return True

            elif command == "/summarize":
                await self.handle_summarize_command()
                return True

            elif command == "/sessions":
                self.handle_sessions_command(arg)
                return True

            elif command == "/session":
                self.handle_session_info_command()
                return True

            elif command == "/search":
                self.handle_search_command(arg)
                return True

            elif command in ("/hybrid", "/find"):
                await self.handle_hybrid_command(arg)
                return True

            elif command in ("/transcript", "/dialogue"):
                self.handle_transcript_command(arg)
                return True

            elif command in ("/new", "/newsession"):
                self.handle_new_session_command(arg)
                return True

            elif command == "/models":
                await self.handle_models_command()
                return True

            elif command in ("/use", "/model"):
                if not arg:
                    self.console.print("[yellow]Usage: /use <model_name>[/yellow]")
                    return True
                self.assistant.set_model(arg)
                self.console.print(f"[bold green]Switched active model to:[/bold green] {arg}\n")
                return True

            elif command == "/mode":
                if arg:
                    try:
                        new_mode = self.assistant.set_mode(arg)
                        desc = "conversational dialogue, single-hop memory" if new_mode == AssistantMode.CHAT else "autonomous multi-turn actions"
                        self.console.print(f"[bold green]Switched mode to:[/bold green] {new_mode.value.upper()} [dim]({desc})[/dim]\n")
                    except ValueError:
                        self.console.print(f"[bold red]Invalid mode '{arg}'. Choose 'chat' or 'agent'.[/bold red]\n")
                else:
                    toggled = self.assistant.toggle_mode()
                    desc = "conversational dialogue, single-hop memory" if toggled == AssistantMode.CHAT else "autonomous multi-turn actions"
                    self.console.print(f"[bold magenta]Toggled mode to:[/bold magenta] {toggled.value.upper()} [dim]({desc})[/dim]\n")
                return True

            elif command in ("/tools", "/tool", "/toggle"):
                self.handle_tools_command(arg)
                return True

            elif command == "/compress":
                self.handle_compression_command(arg)
                return True

            elif command == "/persona":
                await self.handle_persona_command(arg)
                return True

            elif command in ("/adapt", "/evolve"):
                await self.handle_persona_adapt_command()
                return True

            elif command in ("/wipe", "/reset"):
                self.handle_wipe_command(arg)
                return True

            elif command in ("/export", "/backup"):
                self.handle_export_command(arg)
                return True

            elif command in ("/import", "/restore"):
                self.handle_import_command(arg)
                return True

            elif command in ("/remember", "/recall"):
                self.handle_remember_command(arg)
                return True

            elif command == "/forget":
                self.handle_forget_command(arg)
                return True

            elif command in ("/facts", "/fact", "/memory"):
                self.handle_facts_command(arg)
                return True

            elif command == "/buffer":
                self.handle_buffer_command()
                return True

            elif command == "/clear":
                sub = arg.strip().lower()
                if sub in ("all", "db", "database", "everything"):
                    stats = self.assistant.wipe_conversation(all_sessions=True)
                    self.console.print(
                        f"[bold green]✓ Successfully wiped ALL conversation history from SQLite![/bold green]\n"
                        f"Cleared [bold white]{stats.get('sessions_wiped', 0)}[/bold white] sessions and "
                        f"[bold white]{stats.get('messages_wiped', 0)}[/bold white] messages.\n"
                        f"Started fresh session: [cyan]{self.assistant.current_session.id}[/cyan]\n"
                    )
                elif sub in ("history", "conversation", "conv", "session", "messages", "current"):
                    stats = self.assistant.wipe_conversation(all_sessions=False)
                    self.console.print(
                        f"[bold green]✓ Successfully wiped conversation messages for active session![/bold green]\n"
                        f"Removed [bold white]{stats.get('messages_wiped', 0)}[/bold white] message(s). In-memory buffer cleared.\n"
                    )
                else:
                    self.assistant.clear_history()
                    self.console.print(
                        "[bold green]Conversation history cleared (system prompt preserved).[/bold green]\n"
                        "[dim]Tip: To also wipe messages from the database, use [bold cyan]/wipe conversation[/bold cyan] or [bold cyan]/clear all[/bold cyan].[/dim]\n"
                    )
                return True

            elif command in ("/vacuum", "/maintenance"):
                self.handle_maintenance_command()
                return True

            elif command in ("/storage", "/db"):
                self.handle_storage_command()
                return True

            elif command == "/archive":
                self.handle_archive_command(arg)
                return True

            elif command == "/system":
                if arg:
                    self.assistant.memory.set_system_prompt(arg)
                    self.console.print(f"[bold green]System prompt updated directly in memory.[/bold green]\n")
                else:
                    prompt = self.assistant.memory.get_system_prompt() or "(None)"
                    self.console.print(f"[bold cyan]Current System Prompt:[/bold cyan]\n{prompt}\n")
                return True

            else:
                self.console.print(f"[yellow]Unknown command '{command}'. Type /help for available commands.[/yellow]\n")
                return True

        # Regular chat turn
        await self._stream_response(cmd)
        return True

    async def _stream_response(self, user_text: str) -> None:
        """Stream model response to console with live output and telemetry."""
        self.console.print(f"\n[bold green]You:[/bold green] {user_text}")
        self.console.print(f"[bold blue]Assistant ({self.assistant.active_model}):[/bold blue]")

        thinking_shown = False
        final_stats: Optional[StreamStats] = None
        final_comp_stats: Optional[CompressionStats] = None
        persona_was_updated = False

        try:
            async for chunk in self.assistant.chat_stream(user_text):
                if chunk.persona_updated:
                    persona_was_updated = True

                if chunk.compression_stats:
                    final_comp_stats = chunk.compression_stats

                # If model triggered autonomous tool execution (Phase 6)
                if chunk.tool_call_notice:
                    if thinking_shown:
                        self.console.print("\n")
                        thinking_shown = False
                    self.console.print(f"[cyan]⚡ {chunk.tool_call_notice}[/cyan]")
                    sys.stdout.flush()

                # If model is outputting thinking tokens (e.g. Gemma 4 / reasoning models)
                if chunk.thinking_delta:
                    if not thinking_shown:
                        self.console.print("[dim italic]Thinking...[/dim italic] ", end="")
                        thinking_shown = True
                    self.console.print(f"[dim]{chunk.thinking_delta}[/dim]", end="")
                    sys.stdout.flush()

                if chunk.delta:
                    if thinking_shown:
                        self.console.print("\n")
                        thinking_shown = False
                    self.console.print(chunk.delta, end="")
                    sys.stdout.flush()

                if chunk.is_done and chunk.stats:
                    final_stats = chunk.stats

            self.console.print()  # Final newline

            if persona_was_updated:
                self.console.print(
                    f"[bold green]✦ Dynamic Persona Update:[/bold green] Model updated its persona guidelines! "
                    f"Saved to [bold cyan]{self.assistant.config.personality_path}[/bold cyan] (backup created: .bak)\n"
                )

            self.print_telemetry_bar(final_stats, final_comp_stats)

        except Exception as e:
            self.console.print(f"\n[bold red]Error during generation:[/bold red] {e}\n")

    async def start(self) -> None:
        """Run the main interactive prompt loop."""
        self.print_welcome_banner()

        # Check Ollama connectivity
        is_healthy = await self.assistant.verify_provider_health()
        if not is_healthy:
            self.console.print(
                f"[bold red]Warning:[/bold red] Cannot connect to Ollama at {self.assistant.config.ollama_host}. "
                "Make sure Ollama is running (`ollama serve`).\n"
            )

        try:
            while True:
                try:
                    user_input = await self.prompt_session.prompt_async(
                        HTML("<ansicyan><b>➤</b></ansicyan> ")
                    )
                    keep_running = await self.handle_input(user_input)
                    if not keep_running:
                        break
                except KeyboardInterrupt:
                    self.console.print("\n[dim]Input cancelled. Press Ctrl+D or type /exit to quit.[/dim]")
                    continue
                except EOFError:
                    self.console.print("\n[yellow]Session interrupted.[/yellow]")
                    break
        finally:
            await self._cleanup_and_exit()
