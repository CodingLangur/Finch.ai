"""Interactive Rich CLI for streaming chat, telemetry, and buffer monitoring."""
import asyncio
import os
import sys
from typing import Optional
import psutil
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


class InteractiveCLI:
    """Rich interactive streaming terminal interface."""

    def __init__(self, assistant: AIAssistant):
        self.assistant = assistant
        self.console = Console()

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
        banner.add_row("Mode:", f"[magenta]{self.assistant.mode.value.upper()}[/magenta]")
        banner.add_row("Persona:", f"[yellow]{self.assistant.config.personality_path}[/yellow]")
        banner.add_row("Compression:", comp_status)
        banner.add_row("Sliding Window:", f"{self.assistant.memory.max_messages} messages (pinned system prompt)")
        banner.add_row("Process Memory:", f"{mem_mb} MB")

        panel = Panel(
            banner,
            title="[bold blue]🤖 Finch.ai Terminal[/bold blue]",
            subtitle="[dim]Type /help for commands, /exit to quit[/dim]",
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
        table.add_row("/mode [chatbot|agent]", "Toggle or set assistant mode")
        table.add_row("/compress [on|off|stats]", "Toggle or inspect Headroom context compression")
        table.add_row("/persona [reload|edit]", "View, reload, or manage personality.md")
        table.add_row("/buffer", "Inspect current sliding-window message buffer")
        table.add_row("/clear", "Clear message history (retains system prompt)")
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

    def handle_persona_command(self, arg: str) -> None:
        """View, reload, or get edit instructions for personality.md."""
        arg = arg.lower()
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
        else:
            # Display current persona markdown
            content = self.assistant.persona_manager.load_persona()
            panel = Panel(
                Markdown(content),
                title=f"[bold yellow]Active Persona ({self.assistant.config.personality_path})[/bold yellow]",
                border_style="yellow",
                box=ROUNDED,
            )
            self.console.print(panel)
            self.console.print("[dim]Use [bold cyan]/persona reload[/bold cyan] after editing the file.[/dim]\n")

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
        self.console.print()

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
                        self.console.print(f"[bold green]Switched mode to:[/bold green] {new_mode.value.upper()}\n")
                    except ValueError:
                        self.console.print(f"[bold red]Invalid mode '{arg}'. Choose 'chatbot' or 'agent'.[/bold red]\n")
                else:
                    toggled = self.assistant.toggle_mode()
                    self.console.print(f"[bold magenta]Toggled mode to:[/bold magenta] {toggled.value.upper()}\n")
                return True

            elif command == "/compress":
                self.handle_compression_command(arg)
                return True

            elif command == "/persona":
                self.handle_persona_command(arg)
                return True

            elif command == "/buffer":
                self.handle_buffer_command()
                return True

            elif command == "/clear":
                self.assistant.clear_history()
                self.console.print("[bold green]Conversation history cleared (system prompt preserved).[/bold green]\n")
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
                    loop = asyncio.get_running_loop()
                    user_input = await loop.run_in_executor(None, input, "➤ ")
                    keep_running = await self.handle_input(user_input)
                    if not keep_running:
                        break
                except (KeyboardInterrupt, EOFError):
                    self.console.print("\n[yellow]Session interrupted.[/yellow]")
                    break
        finally:
            await self._cleanup_and_exit()
