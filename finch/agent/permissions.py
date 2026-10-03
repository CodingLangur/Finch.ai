"""3-State Tool Permission Engine (OFF / ASK / AUTO) for Finch.ai."""
from enum import Enum
from typing import Any, Dict, Optional


class ToolPermissionMode(str):
    """3-state tool permission mode ('off', 'ask', 'auto') with boolean evaluation support.

    Evaluates to False if 'off', and True if 'ask' or 'auto', preserving backward-compatibility
    with binary boolean checks.
    """

    OFF = "off"
    ASK = "ask"
    AUTO = "auto"

    def __bool__(self) -> bool:
        return self.lower() not in ("off", "disabled", "false", "0")


def resolve_tool_permission_mode(flag: Any) -> ToolPermissionMode:
    """Resolve a raw configuration value or string into a validated ToolPermissionMode."""
    if isinstance(flag, ToolPermissionMode):
        return flag
    if isinstance(flag, str):
        cleaned = flag.strip().lower()
        if cleaned in ("auto", "always", "yes", "full"):
            return ToolPermissionMode("auto")
        elif cleaned in ("off", "disabled", "false", "no", "0"):
            return ToolPermissionMode("off")
        return ToolPermissionMode("ask")
    return ToolPermissionMode("ask") if flag else ToolPermissionMode("off")


class ToolPermissionEngine:
    """Manages granular 3-state execution permissions across tool categories."""

    CYCLE_ORDER = [ToolPermissionMode.ASK, ToolPermissionMode.AUTO, ToolPermissionMode.OFF]

    def __init__(self, initial_permissions: Optional[Dict[str, Any]] = None):
        self._permissions: Dict[str, ToolPermissionMode] = {
            "terminal": ToolPermissionMode.ASK,
            "python": ToolPermissionMode.ASK,
            "web": ToolPermissionMode.OFF,
            "files": ToolPermissionMode.ASK,
        }
        if initial_permissions:
            for cat, mode in initial_permissions.items():
                self._permissions[cat] = resolve_tool_permission_mode(mode)

    def get_permission(self, category: str) -> ToolPermissionMode:
        """Get the current permission mode for a category."""
        return self._permissions.get(category.lower(), ToolPermissionMode.OFF)

    def is_enabled(self, category: str) -> bool:
        """Return True if category is either ASK or AUTO (i.e. not OFF)."""
        return bool(self.get_permission(category))

    def set_permission(self, category: str, mode: Any) -> ToolPermissionMode:
        """Set permission mode for a category."""
        resolved = resolve_tool_permission_mode(mode)
        self._permissions[category.lower()] = resolved
        return resolved

    def toggle_permission(self, category: str) -> ToolPermissionMode:
        """Cycle permission mode: ASK -> AUTO -> OFF -> ASK."""
        cat = category.lower()
        curr = self.get_permission(cat)
        try:
            next_idx = (self.CYCLE_ORDER.index(curr) + 1) % len(self.CYCLE_ORDER)
            new_mode = self.CYCLE_ORDER[next_idx]
        except ValueError:
            new_mode = ToolPermissionMode.ASK
        self._permissions[cat] = new_mode
        return new_mode

    def get_all(self) -> Dict[str, ToolPermissionMode]:
        """Return a copy of all category permissions."""
        return dict(self._permissions)
