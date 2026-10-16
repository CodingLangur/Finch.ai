"""Dynamic Temporal Context Generator for Finch.ai runtime inference."""
import re
from datetime import datetime
from typing import Optional

TEMPORAL_HEADER = "[Temporal Context]"
TEMPORAL_PATTERN = re.compile(
    r"(?:<temporal_context>\s*)?\[Temporal Context\](?:\n(?:[ \t]*-[ \t]+[A-Za-z ]+:[^\n]*|[ \t]+[^\n]*))*(?:\s*</temporal_context>)?",
    re.MULTILINE,
)


def get_temporal_context(
    dt: Optional[datetime] = None,
    wrap_tags: bool = False,
) -> str:
    """Generate dynamic temporal context with local time, weekday, and timezone.

    Always evaluated at runtime per request to prevent caching stale timestamps.

    Args:
        dt: Optional datetime object. If None, current local time with system
            timezone is generated at call time.
        wrap_tags: If True, wraps block in XML `<temporal_context>` tags
            for additional AST / tag protector compatibility.

    Returns:
        Formatted [Temporal Context] block string.
    """
    if dt is None:
        dt = datetime.now().astimezone()
    elif dt.tzinfo is None:
        dt = dt.astimezone()

    weekday = dt.strftime("%A")
    formatted_date_time = dt.strftime("%Y-%m-%d %I:%M:%S %p")
    tz_name = dt.tzname() or ""
    utc_offset = dt.strftime("%z")

    # Format UTC offset (e.g. +0530 -> UTC+05:30, -0700 -> UTC-07:00)
    if len(utc_offset) == 5:
        utc_offset_str = f"UTC{utc_offset[:3]}:{utc_offset[3:]}"
    elif utc_offset:
        utc_offset_str = f"UTC{utc_offset}"
    else:
        utc_offset_str = "UTC"

    tz_display = f"{tz_name} ({utc_offset_str})" if tz_name else utc_offset_str

    lines = [
        TEMPORAL_HEADER,
        f"- Current Time: {weekday}, {formatted_date_time} {tz_display}",
        f"- Weekday: {weekday}",
        f"- Timezone: {tz_display}",
        f"- ISO Timestamp: {dt.isoformat()}",
    ]
    block = "\n".join(lines)

    if wrap_tags:
        return f"<temporal_context>\n{block}\n</temporal_context>"
    return block


def extract_temporal_context(text: str) -> Optional[str]:
    """Extract [Temporal Context] block from text if present."""
    match = TEMPORAL_PATTERN.search(text)
    if match:
        return match.group(0).strip()
    return None


def strip_temporal_context(text: str) -> str:
    """Remove [Temporal Context] block from text."""
    cleaned = TEMPORAL_PATTERN.sub("", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def inject_temporal_context(
    base_text: str,
    temporal_context: Optional[str] = None,
) -> str:
    """Append or replace dynamic [Temporal Context] block in a base prompt string.

    Args:
        base_text: Existing prompt text (e.g., base system prompt).
        temporal_context: Pre-computed temporal context string. If None,
            calls `get_temporal_context()` dynamically.

    Returns:
        Prompt text with updated runtime temporal context.
    """
    tc = temporal_context if temporal_context is not None else get_temporal_context()
    clean_base = strip_temporal_context(base_text)
    if not clean_base:
        return tc
    return f"{clean_base}\n\n{tc}"
