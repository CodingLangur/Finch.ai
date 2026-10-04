"""Unit tests for Finch to Hermes Agent Seed Migration (finch.tools.export_hermes)."""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from finch.storage import SQLiteArchive
from finch.tools.export_hermes import export_to_hermes, main


class TestExportHermes(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.hermes_dir = os.path.join(self.tmp_dir, ".hermes")
        self.db_path = os.path.join(self.tmp_dir, "test_conversations.db")
        self.personality_path = os.path.join(self.tmp_dir, "test_personality.md")

        # Create dummy personality
        self.sample_persona = (
            "# Hermes Custom Soul\n\n"
            "## Core Identity\n"
            "- **Name**: Finch-Hermes\n"
            "- **Style**: Pragmatic, direct, and concise.\n"
        )
        with open(self.personality_path, "w", encoding="utf-8") as f:
            f.write(self.sample_persona)

        # Create test archive and populate sample facts
        self.archive = SQLiteArchive(db_path=self.db_path)
        self.archive.add_fact(fact="Prefers Vim keybindings in editor", category="preference")
        self.archive.add_fact(fact="Targeting GATE CSE examination", category="goal")
        self.archive.add_fact(fact="Primary language is Python and Rust", category="tech_stack")

    def tearDown(self):
        self.archive.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_export_creates_soul_and_user_memories(self):
        """Test default export creates SOUL.md and memories/USER.md with § line delimiters."""
        res = export_to_hermes(
            hermes_dir=self.hermes_dir,
            personality_path=self.personality_path,
            db_path=self.db_path,
        )

        self.assertEqual(res["status"], "success")
        self.assertEqual(res["facts_count"], 3)
        self.assertFalse(res["dry_run"])

        soul_file = os.path.join(self.hermes_dir, "SOUL.md")
        user_file = os.path.join(self.hermes_dir, "memories", "USER.md")

        self.assertTrue(os.path.exists(soul_file))
        self.assertTrue(os.path.exists(user_file))

        # Check SOUL.md
        with open(soul_file, "r", encoding="utf-8") as f:
            soul_content = f.read()
        self.assertIn("Finch-Hermes", soul_content)
        self.assertIn("Pragmatic, direct, and concise", soul_content)

        # Check USER.md
        with open(user_file, "r", encoding="utf-8") as f:
            user_lines = [line.strip() for line in f if line.strip()]

        self.assertEqual(len(user_lines), 3)
        for line in user_lines:
            self.assertTrue(line.startswith("§ "), f"Expected line to start with '§ ', got: {line}")

        self.assertIn("§ Prefers Vim keybindings in editor", user_lines)
        self.assertIn("§ Targeting GATE CSE examination", user_lines)
        self.assertIn("§ Primary language is Python and Rust", user_lines)

    def test_export_dry_run_creates_no_files(self):
        """Test that --dry-run reads and returns metrics without creating any files on disk."""
        res = export_to_hermes(
            hermes_dir=self.hermes_dir,
            personality_path=self.personality_path,
            db_path=self.db_path,
            dry_run=True,
        )

        self.assertEqual(res["status"], "success")
        self.assertTrue(res["dry_run"])
        self.assertEqual(res["facts_count"], 3)

        soul_file = os.path.join(self.hermes_dir, "SOUL.md")
        user_file = os.path.join(self.hermes_dir, "memories", "USER.md")

        self.assertFalse(os.path.exists(self.hermes_dir))
        self.assertFalse(os.path.exists(soul_file))
        self.assertFalse(os.path.exists(user_file))

    def test_export_with_include_category(self):
        """Test formatting facts with category prefixes (e.g. § [preference] ...)."""
        export_to_hermes(
            hermes_dir=self.hermes_dir,
            personality_path=self.personality_path,
            db_path=self.db_path,
            include_category=True,
        )

        user_file = os.path.join(self.hermes_dir, "memories", "USER.md")
        with open(user_file, "r", encoding="utf-8") as f:
            user_lines = [line.strip() for line in f if line.strip()]

        self.assertIn("§ [preference] Prefers Vim keybindings in editor", user_lines)
        self.assertIn("§ [goal] Targeting GATE CSE examination", user_lines)
        self.assertIn("§ [tech_stack] Primary language is Python and Rust", user_lines)

    def test_export_append_mode(self):
        """Test appending additional facts to an existing USER.md without overwriting."""
        memories_dir = os.path.join(self.hermes_dir, "memories")
        os.makedirs(memories_dir, exist_ok=True)
        user_file = os.path.join(memories_dir, "USER.md")

        with open(user_file, "w", encoding="utf-8") as f:
            f.write("§ Initial pre-existing Hermes memory\n")

        export_to_hermes(
            hermes_dir=self.hermes_dir,
            personality_path=self.personality_path,
            db_path=self.db_path,
            append=True,
        )

        with open(user_file, "r", encoding="utf-8") as f:
            user_lines = [line.strip() for line in f if line.strip()]

        self.assertEqual(len(user_lines), 4)
        self.assertEqual(user_lines[0], "§ Initial pre-existing Hermes memory")
        self.assertIn("§ Prefers Vim keybindings in editor", user_lines)

    def test_cli_main_execution(self):
        """Test invoking CLI main() function directly with arguments."""
        exit_code = main([
            "--hermes-dir", self.hermes_dir,
            "--personality", self.personality_path,
            "--db-path", self.db_path,
            "--quiet",
        ])
        self.assertEqual(exit_code, 0)

        soul_file = os.path.join(self.hermes_dir, "SOUL.md")
        user_file = os.path.join(self.hermes_dir, "memories", "USER.md")
        self.assertTrue(os.path.exists(soul_file))
        self.assertTrue(os.path.exists(user_file))

    def test_cli_subprocess_module_execution(self):
        """Test invoking python -m finch.tools.export_hermes via subprocess."""
        cmd = [
            sys.executable,
            "-m",
            "finch.tools.export_hermes",
            "--hermes-dir", self.hermes_dir,
            "--personality", self.personality_path,
            "--db-path", self.db_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, f"CLI execution failed: {proc.stderr}")
        self.assertIn("Hermes Agent Seed Migration Summary", proc.stdout)

        soul_file = os.path.join(self.hermes_dir, "SOUL.md")
        self.assertTrue(os.path.exists(soul_file))

    def test_configure_hermes_mcp_creates_config_yaml(self):
        """Test configure_hermes_mcp registers Finch MCP server in config.yaml."""
        import yaml
        from finch.tools.export_hermes import configure_hermes_mcp, DEFAULT_FINCH_HERMES_TOOLS

        res = configure_hermes_mcp(hermes_dir=self.hermes_dir)
        self.assertEqual(res["status"], "success")

        config_path = os.path.join(self.hermes_dir, "config.yaml")
        self.assertTrue(os.path.exists(config_path))

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.assertIn("mcp_servers", cfg)
        self.assertIn("finch", cfg["mcp_servers"])
        finch_cfg = cfg["mcp_servers"]["finch"]
        self.assertEqual(finch_cfg["command"], "python")
        self.assertEqual(finch_cfg["args"], ["-m", "finch.mcp_server"])
        self.assertEqual(finch_cfg["tools"]["include"], DEFAULT_FINCH_HERMES_TOOLS)

    def test_configure_hermes_mcp_merges_with_existing(self):
        """Test configure_hermes_mcp preserves other configuration keys."""
        import yaml
        from finch.tools.export_hermes import configure_hermes_mcp

        os.makedirs(self.hermes_dir, exist_ok=True)
        config_path = os.path.join(self.hermes_dir, "config.yaml")
        existing_data = {
            "model": "claude-3-5-sonnet",
            "mcp_servers": {
                "filesystem": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                }
            },
        }
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(existing_data, f)

        configure_hermes_mcp(hermes_dir=self.hermes_dir)

        with open(config_path, "r", encoding="utf-8") as f:
            updated = yaml.safe_load(f)

        self.assertEqual(updated["model"], "claude-3-5-sonnet")
        self.assertIn("filesystem", updated["mcp_servers"])
        self.assertIn("finch", updated["mcp_servers"])

    def test_export_with_configure_mcp(self):
        """Test export_to_hermes with configure_mcp=True creates SOUL.md, USER.md, and config.yaml."""
        res = export_to_hermes(
            hermes_dir=self.hermes_dir,
            personality_path=self.personality_path,
            db_path=self.db_path,
            configure_mcp=True,
        )

        self.assertIsNotNone(res["mcp_config"])
        self.assertTrue(os.path.exists(os.path.join(self.hermes_dir, "SOUL.md")))
        self.assertTrue(os.path.exists(os.path.join(self.hermes_dir, "memories", "USER.md")))
        self.assertTrue(os.path.exists(os.path.join(self.hermes_dir, "config.yaml")))

    def test_cli_subprocess_with_configure_mcp(self):
        """Test invoking python -m finch.tools.export_hermes --configure-mcp."""
        cmd = [
            sys.executable,
            "-m",
            "finch.tools.export_hermes",
            "--hermes-dir", self.hermes_dir,
            "--personality", self.personality_path,
            "--db-path", self.db_path,
            "--configure-mcp",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, f"CLI execution failed: {proc.stderr}")
        self.assertIn("MCP Server Config", proc.stdout)
        self.assertTrue(os.path.exists(os.path.join(self.hermes_dir, "config.yaml")))

    def test_ai_assistant_namespace_compatibility(self):
        """Test that ai_assistant.tools.export_hermes exposes the same functionality."""
        from ai_assistant.tools import export_to_hermes as ai_export, configure_hermes_mcp as ai_cfg
        from ai_assistant.tools.export_hermes import export_to_hermes as ai_export_module, configure_hermes_mcp as ai_cfg_module
        from finch.tools.export_hermes import configure_hermes_mcp

        self.assertIs(ai_export, export_to_hermes)
        self.assertIs(ai_export_module, export_to_hermes)
        self.assertIs(ai_cfg, configure_hermes_mcp)
        self.assertIs(ai_cfg_module, configure_hermes_mcp)



if __name__ == "__main__":
    unittest.main()
