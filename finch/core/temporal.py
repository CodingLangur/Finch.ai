"""Dynamic Temporal Context Generator and Zero-Turn Memory Prefetch for Finch.ai."""
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

TEMPORAL_HEADER = "[Temporal Context]"
PREFETCH_HEADER = "[Implicit Memory Prefetch: Relevant Past Context]"

# Pattern matching the dynamic [Temporal Context] block inside system prompts
TEMPORAL_BLOCK_PATTERN = re.compile(
    r"(?:<temporal_context>\s*)?\[Temporal Context\](?:\n(?:[ \t]*-[ \t]+[A-Za-z ]+:[^\n]*|[ \t]+[^\n]*))*(?:\s*</temporal_context>)?",
    re.MULTILINE,
)

# Pattern matching implicit memory prefetch blocks
PREFETCH_BLOCK_PATTERN = re.compile(
    r"(?:<recalled_past_context>\s*)?\[Implicit Memory Prefetch: Relevant Past Context\].*?(?=(?:\n\n\[|\n\n#|\n\n---|\s*</recalled_past_context>)|\Z)(?:\s*</recalled_past_context>)?",
    re.DOTALL,
)

# Lightweight keyword / regex matcher to detect planning and scheduling prompts at ingress
TEMPORAL_TRIGGER_KEYWORDS = [
    r"schedule(?:s|d|ing)?",
    r"reschedule(?:s|d|ing)?",
    r"plan(?:s|ned|ning)?",
    r"calendar(?:s)?",
    r"agenda(?:s)?",
    r"appointment(?:s)?",
    r"meeting(?:s)?",
    r"sync(?:s)?",
    r"standup(?:s)?",
    r"reminder(?:s)?",
    r"remind(?:ing)?",
    r"deadline(?:s)?",
    r"due\s+date(?:s)?",
    r"timeline(?:s)?",
    r"roadmap(?:s)?",
    r"milestone(?:s)?",
    r"itinerar(?:y|ies)",
    r"availability",
    r"to-?do(?:s)?",
    r"today",
    r"tomorrow",
    r"yesterday",
    r"tonight",
    r"(?:this|next|last|upcoming)\s+(?:week|weekend|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday|sprint|term|quarter)",
    r"when\s+(?:is|are|was|were|did|do|will|can)\b",
    r"what\s+time\b",
    r"what\s+day\b",
    r"coming\s+up\b",
    r"upcoming\b",
]

TEMPORAL_PATTERN = re.compile(
    r"\b(?:" + "|".join(TEMPORAL_TRIGGER_KEYWORDS) + r")\b",
    re.IGNORECASE,
)
TEMPORAL_TRIGGER_PATTERN = TEMPORAL_PATTERN


def is_temporal_trigger(text: str) -> bool:
    """Check if incoming user prompt matches planning/scheduling temporal triggers."""
    if not text or not isinstance(text, str):
        return False
    return bool(TEMPORAL_PATTERN.search(text))


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

    # Compute upcoming 7 days to eliminate LLM arithmetic hallucinations for relative dates
    upcoming_days = [(dt + timedelta(days=i)).strftime("%A (%Y-%m-%d)") for i in range(1, 8)]
    upcoming_str = ", ".join(upcoming_days)

    lines = [
        TEMPORAL_HEADER,
        f"- Current Time: {weekday}, {formatted_date_time} {tz_display}",
        f"- Weekday: {weekday}",
        f"- Timezone: {tz_display}",
        f"- ISO Timestamp: {dt.isoformat()}",
        f"- Upcoming Week Days: {upcoming_str}",
        "- Reference Note: Use this current timestamp and weekday reference to resolve all relative dates (e.g., 'today', 'tomorrow', 'yesterday', 'next Wednesday', upcoming deadlines).",
    ]
    block = "\n".join(lines)

    if wrap_tags:
        return f"<temporal_context>\n{block}\n</temporal_context>"
    return block


def extract_temporal_context(text: str) -> Optional[str]:
    """Extract [Temporal Context] block from text if present."""
    match = TEMPORAL_BLOCK_PATTERN.search(text)
    if match:
        return match.group(0).strip()
    return None


def strip_temporal_context(text: str) -> str:
    """Remove [Temporal Context] block from text."""
    cleaned = TEMPORAL_BLOCK_PATTERN.sub("", text)
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


def format_prefetched_memories(
    past_sessions: List[Any],
    user_facts: Optional[List[Dict[str, Any]]] = None,
    max_tokens: int = 350,
    wrap_tags: bool = True,
) -> Optional[str]:
    """Format retrieved hybrid search sessions and user facts into a clean context block.

    Applies token guardrail limits (default 350 tokens, ~1400 chars) to prevent ambient
    memory from consuming excessive context space. Wraps results in <recalled_past_context>
    tags so LLMs cleanly recognize them as recalled past context.

    Args:
        past_sessions: List of HybridSearchResult objects or session dicts.
        user_facts: Optional list of relevant user facts.
        max_tokens: Maximum token threshold for context guardrail (default 350).
        wrap_tags: If True, wraps block in XML `<recalled_past_context>` tags.

    Returns:
        Cleanly formatted markdown memory block, or None if no memories provided.
    """
    if not past_sessions and not user_facts:
        return None

    lines = [
        PREFETCH_HEADER,
        "The user's prompt matches planning/scheduling triggers. The following relevant past memories and facts were retrieved from memory:",
    ]

    if past_sessions:
        lines.append("### Past Conversations & Schedules:")
        for idx, r in enumerate(past_sessions, 1):
            title = getattr(r, "title", None) or (r.get("title") if isinstance(r, dict) else "Untitled Session")
            snippets = getattr(r, "matched_snippets", None) or (r.get("matched_snippets") if isinstance(r, dict) else [])
            summary = getattr(r, "summary", None) or (r.get("summary") if isinstance(r, dict) else "")
            updated_at = getattr(r, "updated_at", None) or (r.get("updated_at") if isinstance(r, dict) else "")

            snippet_text = " ".join(snippets) if snippets else (summary or "")
            clean_snippet = snippet_text.replace("[MATCH]", "").replace("[/MATCH]", "").strip()
            if len(clean_snippet) > 200:
                clean_snippet = clean_snippet[:200] + "..."

            date_str = f" | Date: {updated_at[:10]}" if updated_at else ""
            lines.append(f"- Memory #{idx} (Session: \"{title}\"{date_str}):")
            lines.append(f"  \"{clean_snippet}\"")

    if user_facts:
        lines.append("### Relevant Stored Facts & Preferences:")
        for f in user_facts:
            fid = f.get("id", "?")
            cat = f.get("category", "general")
            fact = f.get("fact", "")
            lines.append(f"- [Fact #{fid}] ({cat}): {fact}")

    block = "\n".join(lines)

    # Enforce token guardrail limit (~4 chars per token)
    max_chars = max_tokens * 4
    if len(block) > max_chars:
        truncated = block[:max_chars].rsplit("\n", 1)[0]
        block = f"{truncated}\n... [remaining memory truncated to {max_tokens} tokens context limit]"

    if wrap_tags:
        return f"<recalled_past_context>\n{block}\n</recalled_past_context>"
    return block


def extract_prefetched_memories(text: str) -> Optional[str]:
    """Extract [Implicit Memory Prefetch...] block from text if present."""
    match = PREFETCH_BLOCK_PATTERN.search(text)
    if match:
        return match.group(0).strip()
    return None


def strip_prefetched_memories(text: str) -> str:
    """Remove [Implicit Memory Prefetch...] block from text."""
    cleaned = PREFETCH_BLOCK_PATTERN.sub("", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()
