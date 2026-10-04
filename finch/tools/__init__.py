"""Finch external integration and migration tools."""

__all__ = ["export_to_hermes", "configure_hermes_mcp", "DEFAULT_FINCH_HERMES_TOOLS"]


def __getattr__(name: str):
    if name in ("export_to_hermes", "configure_hermes_mcp", "DEFAULT_FINCH_HERMES_TOOLS"):
        from .export_hermes import export_to_hermes, configure_hermes_mcp, DEFAULT_FINCH_HERMES_TOOLS
        if name == "export_to_hermes":
            return export_to_hermes
        elif name == "configure_hermes_mcp":
            return configure_hermes_mcp
        return DEFAULT_FINCH_HERMES_TOOLS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


