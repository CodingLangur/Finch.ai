"""Tests for the SlidingWindowBuffer and message models."""
import unittest
from ai_assistant.memory.message import ChatMessage, Role
from ai_assistant.memory.sliding_window import SlidingWindowBuffer


class TestSlidingWindowBuffer(unittest.TestCase):
    def test_system_prompt_pinning(self):
        """Verify that the system prompt is always preserved at index 0."""
        buffer = SlidingWindowBuffer(max_messages=4, system_prompt="System instructions")

        # Fill buffer beyond capacity
        for i in range(10):
            buffer.add_user_message(f"User message {i}")
            buffer.add_assistant_message(f"Assistant reply {i}")

        messages = buffer.get_messages()

        # System message must still be at index 0
        self.assertEqual(messages[0].role, Role.SYSTEM)
        self.assertEqual(messages[0].content, "System instructions")

        # Active conversation messages must not exceed max_messages
        conv_messages = [m for m in messages if m.role != Role.SYSTEM]
        self.assertEqual(len(conv_messages), 4)

    def test_sliding_window_eviction(self):
        """Verify that the oldest messages are popped when exceeding max_messages."""
        buffer = SlidingWindowBuffer(max_messages=3)

        buffer.add_user_message("msg 1")
        buffer.add_assistant_message("msg 2")
        buffer.add_user_message("msg 3")
        self.assertEqual(len(buffer.get_messages()), 3)

        # Adding 4th message should evict 'msg 1'
        buffer.add_assistant_message("msg 4")
        contents = [m.content for m in buffer.get_messages()]
        self.assertEqual(contents, ["msg 2", "msg 3", "msg 4"])

    def test_clear_buffer(self):
        """Verify clear() clears conversation messages while retaining system prompt."""
        buffer = SlidingWindowBuffer(max_messages=4, system_prompt="Pinned Prompt")
        buffer.add_user_message("Hello")
        buffer.add_assistant_message("Hi")

        buffer.clear(keep_system=True)
        messages = buffer.get_messages()
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].content, "Pinned Prompt")

        buffer.clear(keep_system=False)
        self.assertEqual(len(buffer.get_messages()), 0)

    def test_to_provider_payload(self):
        """Verify dictionary serialization for LLM API payloads."""
        buffer = SlidingWindowBuffer(max_messages=4, system_prompt="System")
        buffer.add_user_message("User")
        buffer.add_assistant_message("Bot")

        payload = buffer.to_provider_payload()
        self.assertEqual(
            payload,
            [
                {"role": "system", "content": "System"},
                {"role": "user", "content": "User"},
                {"role": "assistant", "content": "Bot"},
            ],
        )

    def test_stats(self):
        """Verify buffer stats reporting."""
        buffer = SlidingWindowBuffer(max_messages=6, system_prompt="Test System")
        buffer.add_user_message("Hello world")
        buffer.add_assistant_message("Greetings")

        stats = buffer.get_stats()
        self.assertEqual(stats["window_size"], 6)
        self.assertEqual(stats["active_messages"], 2)
        self.assertEqual(stats["total_messages"], 3)
        self.assertEqual(stats["user_messages"], 1)
        self.assertEqual(stats["assistant_messages"], 1)
        self.assertTrue(stats["has_system_prompt"])
        self.assertGreater(stats["total_characters"], 0)


if __name__ == "__main__":
    unittest.main()
