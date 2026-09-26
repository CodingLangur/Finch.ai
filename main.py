"""Entry point for Finch.ai."""
import argparse
import asyncio
import sys

from ai_assistant.config import AppConfig
from ai_assistant.core.assistant import AIAssistant, AssistantMode
from ai_assistant.ui.cli import InteractiveCLI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finch.ai - Local & Cloud AI Assistant with Ollama & Gemini runtimes, sliding-window buffer, and telemetry."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Ollama model name (default: Gemma4-26000-ctx:latest)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Ollama host URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="Sliding-window history size in messages (default: 8)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["chat", "chatbot", "agent"],
        default=None,
        help="Initial mode (default: chat)",
    )
    parser.add_argument(
        "--no-compression",
        action="store_true",
        help="Disable Headroom context compression pipeline",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["ollama", "gemini"],
        default=None,
        help="LLM Provider to use (default: from .env or gemini/ollama)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Path to SQLite conversation database (default: conversations.db)",
    )
    parser.add_argument(
        "--no-auto-summarize",
        action="store_true",
        help="Disable automatic session summarization on exit",
    )
    parser.add_argument(
        "-q", "--query",
        type=str,
        default=None,
        help="Run a single query and exit (useful for automated testing/scripts)",
    )
    return parser.parse_args()


async def main_async() -> None:
    args = parse_args()

    # Build config overrides
    config = AppConfig()
    if args.provider:
        config.provider = args.provider
        if args.provider == "gemini" and not args.model:
            config.default_model = config.gemini_default_model
    if args.host:
        config.ollama_host = args.host
    if args.model:
        config.default_model = args.model
    if args.window_size:
        config.window_size = args.window_size
    if args.mode:
        config.default_mode = args.mode
    if args.no_compression:
        config.compression_enabled = False
    if args.db:
        config.db_path = args.db
    if args.no_auto_summarize:
        config.auto_summarize_on_exit = False

    assistant = AIAssistant(config=config)
    cli = InteractiveCLI(assistant=assistant)

    if args.query:
        # Non-interactive single query
        try:
            await cli.handle_input(args.query)
        finally:
            assistant.close()
    else:
        # Interactive chat loop
        await cli.start()


def main() -> None:
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\nSession aborted.")
        sys.exit(0)


if __name__ == "__main__":
    main()
