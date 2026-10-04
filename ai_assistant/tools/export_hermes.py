"""Hermes export re-export for ai_assistant namespace compatibility."""
import sys
from finch.tools.export_hermes import export_to_hermes, main

__all__ = ["export_to_hermes", "main"]

if __name__ == "__main__":
    sys.exit(main())
