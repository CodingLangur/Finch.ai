"""Session Summarizer using fast LLM prompts to generate titles and 2-sentence summaries."""
import re
from typing import List, Optional, Tuple

from .sqlite_archive import MessageRecord, SQLiteArchive
from ..providers.base import BaseLLMProvider


SUMMARIZER_SYSTEM_PROMPT = (
    "You are an expert conversation summarizer. Analyze the conversation and provide a concise title and a 2-sentence summary.\n"
    "Follow this exact format:\n"
    "TITLE: <title in 2 to 6 words>\n"
    "SUMMARY: <exactly 2 sentences summarizing the discussion and key conclusions>"
)


class SessionSummarizer:
    """Generates and persists session title and 2-sentence summary using LLM prompts."""

    def __init__(self, archive: SQLiteArchive, provider: BaseLLMProvider):
        self.archive = archive
        self.provider = provider

    def format_conversation_for_summary(
        self, messages: List[MessageRecord], max_chars_per_msg: int = 400
    ) -> str:
        """Format messages into a compact transcript for summarization."""
        lines: List[str] = []
        for msg in messages:
            if msg.role.lower() == "system":
                continue
            role_label = "User" if msg.role.lower() == "user" else "Assistant"
            content = msg.content.strip().replace("\r\n", " ").replace("\n", " ")
            if len(content) > max_chars_per_msg:
                content = content[:max_chars_per_msg] + "..."
            lines.append(f"{role_label}: {content}")
        return "\n".join(lines)

    def parse_summary_output(self, text: str) -> Tuple[str, str]:
        """Extract title and 2-sentence summary from the model response."""
        title = ""
        summary = ""

        # Check for TITLE: and SUMMARY: tags with optional markdown bold/hashes
        title_match = re.search(
            r"(?:^|\n)[*#_ ]*TITLE[*#_ ]*:\s*([^\n]+)",
            text,
            re.IGNORECASE,
        )
        if title_match:
            title = title_match.group(1).strip().strip('*"\' ')

        summary_match = re.search(
            r"(?:^|\n)[*#_ ]*SUMMARY[*#_ ]*:\s*(.+?)(?=\n[*#_ ]*[A-Z]+[*#_ ]*:|$)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if summary_match:
            summary = summary_match.group(1).strip().strip('*"\' ')

        # Fallback if pattern matching did not capture both cleanly
        if not title or not summary:
            lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
            for line in lines:
                if not title and re.match(r"^[*#_ ]*title[*#_ ]*:", line, flags=re.IGNORECASE):
                    cleaned = re.sub(r"^[*#_ ]*title[*#_ ]*:\s*", "", line, flags=re.IGNORECASE).strip('*"\' ')
                    if cleaned:
                        title = cleaned
                        continue
                if not summary and re.match(r"^[*#_ ]*summary[*#_ ]*:", line, flags=re.IGNORECASE):
                    cleaned = re.sub(r"^[*#_ ]*summary[*#_ ]*:\s*", "", line, flags=re.IGNORECASE).strip('*"\' ')
                    if cleaned:
                        summary = cleaned
                        continue

        # Final sanity checks
        if not title:
            title = "Chat Session"
        if not summary:
            summary = text.strip() if text.strip() else "Session concluded without summary."

        # Clean title length and formatting
        title = re.sub(r"^[*#_\"']+|[*#_\"']+$", "", title).strip()
        if len(title) > 60:
            title = title[:57] + "..."

        return title, summary

    async def summarize_session(
        self,
        session_id: str,
        model: str,
        options: Optional[dict] = None,
    ) -> Tuple[str, str]:
        """Generate title and 2-sentence summary for a session and save to SQLite."""
        messages = self.archive.get_messages(session_id, include_system=False)

        user_messages = [m for m in messages if m.role.lower() == "user"]
        if not user_messages:
            fallback_title = "Empty Session"
            fallback_summary = "No conversational turns took place in this session."
            self.archive.update_session_summary(
                session_id, title=fallback_title, summary=fallback_summary
            )
            return fallback_title, fallback_summary

        transcript = self.format_conversation_for_summary(messages)
        prompt_payload = [
            {"role": "system", "content": SUMMARIZER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Here is the chat session transcript:\n\n{transcript}\n\nGenerate the TITLE and 2-sentence SUMMARY now:",
            },
        ]

        llm_options = {"temperature": 0.2}
        if options:
            llm_options.update(options)

        accumulated: List[str] = []
        try:
            async for chunk in self.provider.stream_chat(
                messages=prompt_payload,
                model=model,
                options=llm_options,
            ):
                if chunk.delta:
                    accumulated.append(chunk.delta)

            raw_response = "".join(accumulated).strip()
            title, summary = self.parse_summary_output(raw_response)

        except Exception as e:
            # Graceful fallback on LLM failure (e.g. timeout, service offline)
            fallback_title = f"Session ({len(user_messages)} turns)"
            fallback_summary = f"Summary generation unavailable ({type(e).__name__})."
            self.archive.update_session_summary(
                session_id, title=fallback_title, summary=fallback_summary
            )
            return fallback_title, fallback_summary

        # Persist to database
        self.archive.update_session_summary(
            session_id,
            title=title,
            summary=summary,
            metadata={"summarized_by_model": model},
        )
        return title, summary
