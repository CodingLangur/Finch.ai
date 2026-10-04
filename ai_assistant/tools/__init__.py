"""External integration and migration tools for ai_assistant."""

__all__ = ["export_to_hermes"]


def __getattr__(name: str):
    if name == "export_to_hermes":
        from finch.tools.export_hermes import export_to_hermes
        return export_to_hermes
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

