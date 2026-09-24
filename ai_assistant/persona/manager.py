"""Dynamic Persona Manager for personality.md file-first injection and model-driven updates."""
import os
import re
import shutil
from pathlib import Path
from typing import Optional, Tuple

DEFAULT_PERSONA_TEMPLATE = """# Assistant Persona & Guidelines

## Core Identity
- **Name**: Antigravity Assistant
- **Role**: Intelligent, adaptable local AI companion and pair programmer.
- **Demeanor**: Direct, thoughtful, pragmatic, and courteous.

## Communication Style
- Provide clear, well-structured, and concise responses.
- Avoid unnecessary filler, fluff, or excessive pleasantries.
- Use GitHub Flavored Markdown (bullet points, bold text, code fences) for readability.
- When explaining technical concepts, emphasize code clarity, performance, and best practices.

## Operational Rules
1. Prioritize accuracy and efficiency over verbosity.
2. Preserve user context and adhere strictly to user instructions.
3. If the user asks you to adapt, modify, or change your personality, tone, behavior, or guidelines, you MUST acknowledge the change and emit the updated version of this entire document wrapped inside `<personality_update>...</personality_update>` tags so the system can save it to disk.
"""

TAG_PATTERN = re.compile(
    r"<personality_update>(.*?)</personality_update>",
    re.DOTALL | re.IGNORECASE
)


class PersonaManager:
    """Manages reading, injecting, parsing, and persisting the assistant's persona."""

    def __init__(self, file_path: str = "personality.md"):
        self.file_path = Path(file_path)
        self.backup_path = Path(f"{file_path}.bak")
        self._ensure_file_exists()

    def _ensure_file_exists(self) -> None:
        """Create default personality.md if not already present."""
        if not self.file_path.exists():
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            self.file_path.write_text(DEFAULT_PERSONA_TEMPLATE, encoding="utf-8")

    def load_persona(self) -> str:
        """Read the active persona markdown content from disk."""
        self._ensure_file_exists()
        return self.file_path.read_text(encoding="utf-8").strip()

    def save_persona(self, new_content: str, append: bool = False) -> str:
        """Validate, backup, and write new persona content to disk."""
        clean_content = new_content.strip()
        if not clean_content:
            raise ValueError("Persona content cannot be empty.")

        # Backup existing file
        if self.file_path.exists():
            shutil.copyfile(self.file_path, self.backup_path)

        if append and self.file_path.exists():
            existing = self.file_path.read_text(encoding="utf-8").rstrip()
            updated = f"{existing}\n\n{clean_content}\n"
        else:
            updated = f"{clean_content}\n"

        self.file_path.write_text(updated, encoding="utf-8")
        return updated.strip()

    @staticmethod
    def parse_update_tags(text: str) -> Optional[str]:
        """Extract content inside <personality_update>...</personality_update> if present."""
        match = TAG_PATTERN.search(text)
        if not match:
            return None

        content = match.group(1).strip()
        # Clean potential markdown code blocks surrounding the contents inside the tag
        if content.startswith("```markdown") and content.endswith("```"):
            content = content[len("```markdown"):-3].strip()
        elif content.startswith("```md") and content.endswith("```"):
            content = content[len("```md"):-3].strip()
        elif content.startswith("```") and content.endswith("```"):
            content = content[3:-3].strip()

        return content if content else None

    @staticmethod
    def strip_tags(text: str) -> str:
        """Remove <personality_update> tags from text for clean user-facing display."""
        cleaned = TAG_PATTERN.sub("", text)
        # Normalize leftover multiple blank lines
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def process_response(self, raw_response: str) -> Tuple[str, bool, Optional[str]]:
        """Process model response.

        Returns:
            Tuple of:
            - user_facing_text: Cleaned response with tags removed.
            - updated: True if a personality update was detected and applied.
            - updated_content: The new persona text if updated, else None.
        """
        extracted = self.parse_update_tags(raw_response)
        if not extracted:
            return raw_response, False, None

        # Persist the update
        saved = self.save_persona(extracted, append=False)
        clean_text = self.strip_tags(raw_response)
        return clean_text, True, saved

    def build_system_prompt(self) -> str:
        """Compose the full pinned system prompt with file content and dynamic protocol."""
        persona = self.load_persona()
        protocol_suffix = (
            "\n\n---\n"
            "### Dynamic Persona Protocol\n"
            "If the user requests you to change, update, or adapt your tone, personality, "
            "demeanor, or rules, acknowledge the request and emit the updated version of "
            "your persona instructions wrapped inside <personality_update>...</personality_update> "
            "tags so the system can update personality.md."
        )
        return f"{persona}{protocol_suffix}"
