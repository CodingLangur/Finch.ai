"""Unit tests for PersonaManager (offline, no model execution)."""
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from ai_assistant.persona.manager import PersonaManager


class TestPersonaManager(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.personality_file = os.path.join(self.test_dir, "test_personality.md")
        self.manager = PersonaManager(file_path=self.personality_file)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_default_file_creation(self):
        """Verify personality.md is automatically created if missing."""
        self.assertTrue(os.path.exists(self.personality_file))
        content = self.manager.load_persona()
        self.assertIn("Assistant Persona & Guidelines", content)

    def test_save_and_backup(self):
        """Verify saving content creates a .bak backup and updates file."""
        original = self.manager.load_persona()
        new_persona = "# Pirate Assistant\nAhoy, matey!"

        self.manager.save_persona(new_persona)

        # Active file must have new content
        self.assertEqual(self.manager.load_persona(), new_persona)

        # Backup file must exist with original content
        backup_file = self.personality_file + ".bak"
        self.assertTrue(os.path.exists(backup_file))
        backup_content = Path(backup_file).read_text(encoding="utf-8").strip()
        self.assertEqual(backup_content, original)

    def test_parse_update_tags(self):
        """Verify parsing <personality_update> tags in model responses."""
        raw_text = (
            "Understood! I will now adopt a sarcastic personality.\n\n"
            "<personality_update>\n"
            "# Sarcastic Persona\n"
            "- Always be sarcastic.\n"
            "</personality_update>\n\n"
            "Here's my answer."
        )

        extracted = PersonaManager.parse_update_tags(raw_text)
        self.assertIsNotNone(extracted)
        self.assertEqual(
            extracted,
            "# Sarcastic Persona\n- Always be sarcastic."
        )

    def test_parse_update_tags_with_code_block(self):
        """Verify parsing when model wraps the update inside ```markdown code block."""
        raw_text = (
            "<personality_update>\n"
            "```markdown\n"
            "# Code Persona\n"
            "- Write code only.\n"
            "```\n"
            "</personality_update>"
        )
        extracted = PersonaManager.parse_update_tags(raw_text)
        self.assertEqual(extracted, "# Code Persona\n- Write code only.")

    def test_strip_tags(self):
        """Verify that <personality_update> is stripped from user-facing text."""
        raw_text = (
            "Sure, I'll update my tone!\n\n"
            "<personality_update>\n# New Persona\n</personality_update>\n\n"
            "How can I help you next?"
        )
        cleaned = PersonaManager.strip_tags(raw_text)
        self.assertNotIn("<personality_update>", cleaned)
        self.assertNotIn("# New Persona", cleaned)
        self.assertIn("Sure, I'll update my tone!", cleaned)
        self.assertIn("How can I help you next?", cleaned)

    def test_process_response_flow(self):
        """Verify end-to-end response processing and file updating."""
        raw_text = (
            "I've updated my persona for you.\n"
            "<personality_update>\n"
            "# Fast Engineer\n"
            "- Direct, no pleasantries.\n"
            "</personality_update>"
        )
        clean_text, updated, new_content = self.manager.process_response(raw_text)

        self.assertTrue(updated)
        self.assertEqual(clean_text, "I've updated my persona for you.")
        self.assertIn("# Fast Engineer", self.manager.load_persona())

    def test_build_system_prompt(self):
        """Verify system prompt includes base persona plus dynamic protocol instructions."""
        prompt = self.manager.build_system_prompt()
        self.assertIn("Assistant Persona & Guidelines", prompt)
        self.assertIn("<personality_update>", prompt)


if __name__ == "__main__":
    unittest.main()
