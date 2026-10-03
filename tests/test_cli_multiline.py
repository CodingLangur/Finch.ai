"""Unit tests for prompt_toolkit multiline editing and keybindings."""
import unittest
from unittest.mock import MagicMock
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.keys import Keys

from finch.ui.cli import create_cli_keybindings, prompt_continuation, InteractiveCLI
from finch.core.assistant import AIAssistant
from finch.config import AppConfig


class DummyEvent:
    def __init__(self, buffer: Buffer):
        self.current_buffer = buffer
        self.app = MagicMock()


class TestCliMultiline(unittest.TestCase):
    def setUp(self):
        self.kb = create_cli_keybindings()

    def test_slash_command_submits_on_single_enter(self):
        buf = Buffer()
        buf.document = Document("/help", cursor_position=5)
        handled = False
        def mock_validate():
            nonlocal handled
            handled = True
        buf.validate_and_handle = mock_validate

        event = DummyEvent(buf)
        bindings = [b for b in self.kb.bindings if b.keys == (Keys.ControlM,)]
        self.assertTrue(len(bindings) > 0)
        bindings[0].call(event)

        self.assertTrue(handled, "Slash command should submit immediately on single Enter")

    def test_multiline_inserts_newline_on_first_enter(self):
        buf = Buffer()
        buf.document = Document("def foo():", cursor_position=10)
        handled = False
        def mock_validate():
            nonlocal handled
            handled = True
        buf.validate_and_handle = mock_validate

        event = DummyEvent(buf)
        bindings = [b for b in self.kb.bindings if b.keys == (Keys.ControlM,)]
        bindings[0].call(event)

        self.assertFalse(handled, "First Enter should not submit multiline code")
        self.assertEqual(buf.text, "def foo():\n")

    def test_double_enter_submits(self):
        buf = Buffer()
        # Text already ends with newline and cursor is at the end
        buf.document = Document("def foo():\n", cursor_position=11)
        handled = False
        def mock_validate():
            nonlocal handled
            handled = True
        buf.validate_and_handle = mock_validate

        event = DummyEvent(buf)
        bindings = [b for b in self.kb.bindings if b.keys == (Keys.ControlM,)]
        bindings[0].call(event)

        self.assertTrue(handled, "Double Enter should submit buffer")
        self.assertEqual(buf.text, "def foo():", "Trailing newline should be stripped on submit")

    def test_alt_enter_submits_immediately(self):
        buf = Buffer()
        buf.document = Document("def foo():\n    return 42", cursor_position=24)
        handled = False
        def mock_validate():
            nonlocal handled
            handled = True
        buf.validate_and_handle = mock_validate

        event = DummyEvent(buf)
        alt_bindings = [
            b for b in self.kb.bindings 
            if b.keys == (Keys.Escape, Keys.ControlM)
        ]
        self.assertTrue(len(alt_bindings) > 0)
        alt_bindings[0].call(event)

        self.assertTrue(handled, "Alt+Enter should submit multiline buffer immediately")

    def test_prompt_continuation_rendered(self):
        res = prompt_continuation(width=4, line_number=1, is_soft_wrap=False)
        self.assertIsNotNone(res)

    def test_interactive_cli_prompt_session_initialized(self):
        config = AppConfig(db_path=":memory:")
        assistant = AIAssistant(config=config)
        cli = InteractiveCLI(assistant=assistant)

        self.assertIsNotNone(cli.prompt_session)
        self.assertTrue(cli.prompt_session.multiline)
        self.assertIsNotNone(cli.key_bindings)
        assistant.close()


if __name__ == "__main__":
    unittest.main()
