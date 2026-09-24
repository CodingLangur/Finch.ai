"""Unit tests for Headroom Context Compression Pipeline (offline, no model execution)."""
import unittest
from ai_assistant.compression.pipeline import ContextCompressionPipeline, HEADROOM_AVAILABLE


class TestCompressionPipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = ContextCompressionPipeline(
            enabled=True,
            compress_user_messages=True,
            compress_system_messages=False,
            protect_recent=0,
            min_tokens_to_compress=50,
        )

    def test_headroom_installed(self):
        """Verify headroom package is available."""
        self.assertTrue(HEADROOM_AVAILABLE, "headroom-ai should be installed in the environment.")

    def test_compression_disabled(self):
        """Verify pipeline bypass when disabled."""
        self.pipeline.toggle(False)
        messages = [{"role": "user", "content": "Sample user query"}]
        result_msgs, stats = self.pipeline.compress_messages(messages)

        self.assertEqual(messages, result_msgs)
        self.assertIsNone(stats)

    def test_system_prompt_untouched(self):
        """Verify system prompt is not altered when compress_system_messages=False."""
        system_content = "# Assistant Persona & Guidelines\nDo not change this."
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": '{"items": [1, 2, 3]} ' * 25},
        ]
        result_msgs, stats = self.pipeline.compress_messages(messages)

        # System message must match exactly
        self.assertEqual(result_msgs[0]["role"], "system")
        self.assertEqual(result_msgs[0]["content"], system_content)

    def test_repetitive_data_compression(self):
        """Verify Headroom compresses repetitive structured JSON / logs."""
        messages = [
            {"role": "system", "content": "You are a concise assistant."},
            {
                "role": "user",
                "content": "Analyze this data:\n" + '{"id": 101, "status": "active", "tags": ["prod", "us-east"]}\n' * 30,
            },
        ]
        result_msgs, stats = self.pipeline.compress_messages(messages)

        self.assertIsNotNone(stats)
        self.assertGreater(stats.tokens_before, 0)
        self.assertGreater(stats.tokens_saved, 0)
        self.assertLess(stats.tokens_after, stats.tokens_before)
        self.assertGreater(stats.savings_pct, 0.0)

    def test_cumulative_session_stats(self):
        """Verify cumulative tokens saved are accumulated across requests."""
        messages = [
            {"role": "user", "content": '{"k": "v", "arr": [1,2,3,4,5]} ' * 20}
        ]
        self.pipeline.compress_messages(messages)
        self.pipeline.compress_messages(messages)

        summary = self.pipeline.get_summary()
        self.assertEqual(summary["total_compressions"], 2)
        self.assertGreater(summary["total_tokens_saved"], 0)
        self.assertGreater(summary["overall_savings_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
