"""Finch to Hermes Agent Seed Migration CLI Utility.

Task 3.1: One-Time Seed Migration
- Exports active personality.md into ~/.hermes/SOUL.md
- Dumps user_facts into ~/.hermes/memories/USER.md formatted with § line delimiters

Usage:
    python -m finch.tools.export_hermes
    python -m finch.tools.export_hermes --hermes-dir ~/.hermes
    python -m finch.tools.export_hermes --dry-run
"""
import argparse
import os
import sys
from typing import Any, Dict, List, Optional

from finch.config import AppConfig
from finch.storage import SQLiteArchive
from finch.memory import PersonaManager


DEFAULT_FINCH_HERMES_TOOLS = [
    "finch_search_history",
    "finch_get_transcript",
    "finch_get_user_facts",
    "finch_remember_fact",
    "finch_run_subagent",
]


def configure_hermes_mcp(
    hermes_dir: Optional[str] = None,
    config_file: Optional[str] = None,
    command: str = "python",
    args: Optional[List[str]] = None,
    tools: Optional[List[str]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Register Finch MCP server configuration in Hermes config.yaml.
    
    Args:
        hermes_dir: Base directory for Hermes agent (default: ~/.hermes or $HERMES_DIR).
        config_file: Optional direct path to config.yaml.
        command: Command executable for starting MCP server (default: 'python').
        args: Command line arguments (default: ['-m', 'finch.mcp_server']).
        tools: List of exposed MCP tool names to include.
        dry_run: If True, simulates configuration without writing to disk.
        
    Returns:
        Dictionary with status and configuration file path.
    """
    import yaml

    base_dir = os.path.abspath(
        os.path.expanduser(hermes_dir or os.getenv("HERMES_DIR", "~/.hermes"))
    )
    target_config = os.path.abspath(
        os.path.expanduser(config_file or os.path.join(base_dir, "config.yaml"))
    )

    cmd_args = args if args is not None else ["-m", "finch.mcp_server"]
    included_tools = tools if tools is not None else list(DEFAULT_FINCH_HERMES_TOOLS)

    config_data: Dict[str, Any] = {}
    if os.path.exists(target_config):
        try:
            with open(target_config, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    config_data = loaded
        except Exception:
            config_data = {}

    if "mcp_servers" not in config_data or not isinstance(config_data["mcp_servers"], dict):
        config_data["mcp_servers"] = {}

    config_data["mcp_servers"]["finch"] = {
        "command": command,
        "args": cmd_args,
        "tools": {
            "include": included_tools,
        },
    }

    # Format cleanly matching Hermes specification
    yaml_lines = ["mcp_servers:"]
    for server_name, server_cfg in config_data["mcp_servers"].items():
        if server_name == "finch":
            yaml_lines.append(f"  {server_name}:")
            yaml_lines.append(f"    command: \"{server_cfg['command']}\"")
            args_formatted = ", ".join(f'"{a}"' for a in server_cfg["args"])
            yaml_lines.append(f"    args: [{args_formatted}]")
            yaml_lines.append("    tools:")
            yaml_lines.append("      include:")
            for t in server_cfg.get("tools", {}).get("include", []):
                yaml_lines.append(f"        - {t}")
        else:
            # Preserve other servers as standard YAML block
            sub_yaml = yaml.dump({server_name: server_cfg}, default_flow_style=False).strip()
            yaml_lines.extend("  " + line for line in sub_yaml.splitlines())

    # Preserve other top-level keys
    other_keys = {k: v for k, v in config_data.items() if k != "mcp_servers"}
    if other_keys:
        other_yaml = yaml.dump(other_keys, default_flow_style=False).strip()
        yaml_lines.append("")
        yaml_lines.append(other_yaml)

    content = "\n".join(yaml_lines) + "\n"

    if not dry_run:
        os.makedirs(os.path.dirname(target_config), exist_ok=True)
        with open(target_config, "w", encoding="utf-8") as f:
            f.write(content)

    return {
        "status": "success",
        "config_file": target_config,
        "server_name": "finch",
        "command": command,
        "args": cmd_args,
        "tools": included_tools,
        "dry_run": dry_run,
    }


def export_to_hermes(
    hermes_dir: Optional[str] = None,
    personality_path: Optional[str] = None,
    db_path: Optional[str] = None,
    soul_file: Optional[str] = None,
    user_file: Optional[str] = None,
    delimiter: str = "§",
    include_category: bool = False,
    append: bool = False,
    configure_mcp: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Export Finch active personality and long-term user facts to Hermes Agent directories.
    
    Args:
        hermes_dir: Base directory for Hermes agent (default: ~/.hermes or $HERMES_DIR).
        personality_path: Source personality.md path (default: Finch config personality_path).
        db_path: Source conversations.db path (default: Finch config db_path).
        soul_file: Custom destination path for SOUL.md.
        user_file: Custom destination path for USER.md.
        delimiter: Line delimiter symbol used for memory entries in USER.md (default: §).
        include_category: Whether to prefix entries with [category] tags.
        append: If True, appends facts to existing USER.md instead of overwriting.
        configure_mcp: If True, also registers Finch MCP server in config.yaml.
        dry_run: If True, performs read and formatting without writing files to disk.
        
    Returns:
        Dictionary with migration statistics and destination paths.
    """
    cfg = AppConfig()

    # 1. Resolve paths
    base_dir = os.path.abspath(
        os.path.expanduser(hermes_dir or os.getenv("HERMES_DIR", "~/.hermes"))
    )
    dest_soul = os.path.abspath(
        os.path.expanduser(soul_file or os.path.join(base_dir, "SOUL.md"))
    )
    dest_user = os.path.abspath(
        os.path.expanduser(user_file or os.path.join(base_dir, "memories", "USER.md"))
    )

    src_personality = os.path.abspath(
        os.path.expanduser(personality_path or cfg.personality_path)
    )
    src_db = os.path.abspath(
        os.path.expanduser(db_path or cfg.db_path)
    )

    # 2. Extract active personality
    persona_mgr = PersonaManager(file_path=src_personality)
    soul_content = persona_mgr.load_persona()
    if not soul_content.endswith("\n"):
        soul_content += "\n"

    # 3. Extract verified user facts from SQLite
    archive = SQLiteArchive(db_path=src_db)
    try:
        raw_facts = archive.list_facts(limit=10000)
    finally:
        archive.close()

    # 4. Format facts with § line delimiters for Hermes USER.md
    fact_lines: List[str] = []
    clean_delim = delimiter.strip() or "§"

    for f in raw_facts:
        fact_text = f.get("fact", "").strip()
        if not fact_text:
            continue
        # Strip any existing leading delimiter to avoid double delimiters
        if fact_text.startswith(clean_delim):
            fact_text = fact_text[len(clean_delim):].strip()

        cat = f.get("category", "").strip()
        if include_category and cat and cat.lower() != "general":
            fact_lines.append(f"{clean_delim} [{cat}] {fact_text}")
        else:
            fact_lines.append(f"{clean_delim} {fact_text}")

    user_content = "\n".join(fact_lines) + ("\n" if fact_lines else "")

    # 5. Write to destination paths (unless dry_run)
    if not dry_run:
        # Create directories
        os.makedirs(os.path.dirname(dest_soul), exist_ok=True)
        os.makedirs(os.path.dirname(dest_user), exist_ok=True)

        # Write SOUL.md
        with open(dest_soul, "w", encoding="utf-8") as f:
            f.write(soul_content)

        # Write USER.md
        mode = "a" if append and os.path.exists(dest_user) else "w"
        with open(dest_user, mode, encoding="utf-8") as f:
            f.write(user_content)

    mcp_result = None
    if configure_mcp:
        mcp_result = configure_hermes_mcp(
            hermes_dir=base_dir,
            dry_run=dry_run,
        )

    return {
        "status": "success",
        "hermes_dir": base_dir,
        "source_personality": src_personality,
        "source_db": src_db,
        "soul_file": dest_soul,
        "user_file": dest_user,
        "soul_bytes": len(soul_content.encode("utf-8")),
        "facts_count": len(fact_lines),
        "user_memories_bytes": len(user_content.encode("utf-8")),
        "mcp_config": mcp_result,
        "dry_run": dry_run,
        "append": append,
    }


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line options for Hermes seed migration."""
    parser = argparse.ArgumentParser(
        prog="python -m finch.tools.export_hermes",
        description="Migrate Finch foundational state (personality & user facts) into Hermes Agent native directories.",
    )
    parser.add_argument(
        "--hermes-dir",
        default=os.getenv("HERMES_DIR", "~/.hermes"),
        help="Hermes home directory (default: ~/.hermes or $HERMES_DIR)",
    )
    parser.add_argument(
        "--personality",
        dest="personality_path",
        default=None,
        help="Source personality markdown file (default: Finch personality_path)",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="Source Finch SQLite database path (default: Finch db_path)",
    )
    parser.add_argument(
        "--soul-file",
        default=None,
        help="Custom destination for SOUL.md (default: <hermes-dir>/SOUL.md)",
    )
    parser.add_argument(
        "--user-file",
        default=None,
        help="Custom destination for USER.md (default: <hermes-dir>/memories/USER.md)",
    )
    parser.add_argument(
        "--delimiter",
        default="§",
        help="Line delimiter prefix for user facts in USER.md (default: §)",
    )
    parser.add_argument(
        "--include-category",
        action="store_true",
        help="Include fact category tag (e.g. § [preference] ...) in USER.md",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append facts to existing USER.md instead of overwriting",
    )
    parser.add_argument(
        "--configure-mcp",
        action="store_true",
        help="Register Finch MCP server in Hermes config.yaml",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate migration without writing any files to disk",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress console summary output",
    )
    return parser.parse_args(args)


def main(args: Optional[List[str]] = None) -> int:
    """CLI entrypoint for Hermes seed migration."""
    parsed = parse_args(args)

    try:
        result = export_to_hermes(
            hermes_dir=parsed.hermes_dir,
            personality_path=parsed.personality_path,
            db_path=parsed.db_path,
            soul_file=parsed.soul_file,
            user_file=parsed.user_file,
            delimiter=parsed.delimiter,
            include_category=parsed.include_category,
            append=parsed.append,
            configure_mcp=parsed.configure_mcp,
            dry_run=parsed.dry_run,
        )
    except Exception as e:
        print(f"Error during Hermes migration: {e}", file=sys.stderr)
        return 1

    if not parsed.quiet:
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table

            console = Console()
            prefix = "[bold yellow][DRY RUN][/bold yellow] " if result["dry_run"] else ""

            table = Table(title=f"{prefix}Hermes Agent Seed Migration Summary", expand=False)
            table.add_column("Component", style="cyan", justify="left")
            table.add_column("Source", style="dim white", justify="left")
            table.add_column("Destination", style="bold green", justify="left")
            table.add_column("Details", style="yellow", justify="right")

            table.add_row(
                "Persona (SOUL.md)",
                result["source_personality"],
                result["soul_file"],
                f"{result['soul_bytes']} bytes",
            )
            table.add_row(
                "User Facts (USER.md)",
                result["source_db"],
                result["user_file"],
                f"{result['facts_count']} facts ({result['user_memories_bytes']} B)",
            )
            if result.get("mcp_config"):
                table.add_row(
                    "MCP Server Config",
                    "Finch FastMCP Service",
                    result["mcp_config"]["config_file"],
                    f"{len(result['mcp_config']['tools'])} tools registered",
                )

            status_note = (
                "[yellow]Dry run simulation complete. No files were written to disk.[/yellow]"
                if result["dry_run"]
                else "[bold green]✓ Migration complete! Foundational state successfully exported to Hermes native directories.[/bold green]"
            )

            console.print()
            console.print(table)
            console.print(Panel(status_note, border_style="green" if not result["dry_run"] else "yellow"))
            console.print()

        except ImportError:
            prefix = "[DRY RUN] " if result["dry_run"] else ""
            print(f"{prefix}Hermes Agent Seed Migration Complete:")
            print(f"  SOUL.md: {result['source_personality']} -> {result['soul_file']} ({result['soul_bytes']} bytes)")
            print(f"  USER.md: {result['source_db']} -> {result['user_file']} ({result['facts_count']} facts, {result['user_memories_bytes']} bytes)")
            if result.get("mcp_config"):
                print(f"  MCP Config: -> {result['mcp_config']['config_file']} ({len(result['mcp_config']['tools'])} tools)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
