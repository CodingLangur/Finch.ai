"""Entry point for Finch.ai."""
import os
import sys

# Auto re-exec inside local .venv if run with global python without dependencies
_venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "bin", "python")
if os.path.exists(_venv_python) and sys.executable != _venv_python:
    try:
        import dotenv  # noqa: F401
    except ImportError:
        os.execv(_venv_python, [_venv_python] + sys.argv)

import argparse
import asyncio

from finch.config import AppConfig
from finch.core.assistant import AIAssistant, AssistantMode
from finch.ui.cli import InteractiveCLI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finch.ai - Local & Cloud AI Assistant with Ollama & Gemini runtimes, sliding-window buffer, and telemetry."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name to use for inference (e.g. llama3.2, gpt-4o-mini, gemini-2.5-flash)",
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
        default="chat",
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
        choices=["ollama", "gemini", "openai", "openai_compatible"],
        default=None,
        help="LLM Provider to use (default: from .env, ollama, gemini, or openai)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Custom base URL for OpenAI-compatible or Ollama endpoint",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API Key for Gemini or OpenAI-compatible provider",
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
        elif args.provider in ("openai", "openai_compatible") and not args.model:
            config.default_model = config.openai_default_model
    elif args.model and "gemini" in args.model.lower():
        config.provider = "gemini"
        config.default_model = args.model
    if args.base_url:
        if config.provider in ("openai", "openai_compatible"):
            config.openai_base_url = args.base_url
        else:
            config.ollama_host = args.base_url
    if args.api_key:
        if args.provider is None and (args.api_key.startswith("AQ.") or args.api_key.startswith("AIza")):
            config.provider = "gemini"
            if not args.model:
                config.default_model = config.gemini_default_model
        if config.provider == "gemini":
            config.gemini_api_key = args.api_key
        else:
            config.openai_api_key = args.api_key
    if args.host:
        config.ollama_host = args.host
    if args.model:
        config.default_model = args.model
    if args.window_size:
        config.window_size = args.window_size
    config.default_mode = args.mode or "chat"
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
