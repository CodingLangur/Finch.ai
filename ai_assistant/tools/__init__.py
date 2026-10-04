"""External integration and migration tools for ai_assistant."""

__all__ = ["export_to_hermes", "configure_hermes_mcp"]


def __getattr__(name: str):
    if name in ("export_to_hermes", "configure_hermes_mcp"):
        from finch.tools.export_hermes import export_to_hermes, configure_hermes_mcp
        if name == "export_to_hermes":
            return export_to_hermes
        return configure_hermes_mcp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


