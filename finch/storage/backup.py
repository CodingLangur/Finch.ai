"""Backup and restore manager for bundling conversation data and personality.md."""
import json
import os
import shutil
import sqlite3
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .sqlite_archive import SQLiteArchive


def utc_now_compact() -> str:
    """Return compact UTC timestamp string for filenames."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def export_backup_bundle(
    archive: SQLiteArchive,
    personality_path: str,
    output_path: Optional[str] = None,
    backup_dir: str = "backups",
) -> str:
    """Export conversation data and personality.md into a portable zip backup bundle.

    Returns:
        Absolute path to the created backup zip file.
    """
    if not output_path:
        os.makedirs(backup_dir, exist_ok=True)
        filename = f"backup_{utc_now_compact()}.zip"
        target_zip = os.path.abspath(os.path.join(backup_dir, filename))
    else:
        target_zip = os.path.abspath(output_path)
        parent = os.path.dirname(target_zip)
        if parent:
            os.makedirs(parent, exist_ok=True)

    # 1. Export conversation data and long-term user facts
    conversations_data = archive.export_all_conversations()
    session_count = len(conversations_data)
    message_count = sum(len(c.get("messages", [])) for c in conversations_data)
    facts_data = archive.list_facts() if hasattr(archive, "list_facts") else []

    # 2. Read personality markdown
    persona_file = Path(personality_path)
    if persona_file.exists():
        persona_content = persona_file.read_text(encoding="utf-8")
    else:
        persona_content = "# Assistant Persona\n"

    # 3. Create manifest
    manifest = {
        "format_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sessions_count": session_count,
        "messages_count": message_count,
        "facts_count": len(facts_data),
        "persona_bytes": len(persona_content.encode("utf-8")),
        "db_filename": "conversations.db",
        "persona_filename": "personality.md",
        "json_filename": "conversations.json",
        "facts_filename": "user_facts.json",
    }

    # 4. Pack into zip archive
    temp_dir = tempfile.mkdtemp(prefix="assistant_export_")
    try:
        # Write manifest
        manifest_path = os.path.join(temp_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        # Write personality
        persona_out_path = os.path.join(temp_dir, "personality.md")
        with open(persona_out_path, "w", encoding="utf-8") as f:
            f.write(persona_content)

        # Write conversations.json
        conv_json_path = os.path.join(temp_dir, "conversations.json")
        with open(conv_json_path, "w", encoding="utf-8") as f:
            json.dump(conversations_data, f, indent=2)

        # Write user_facts.json
        facts_json_path = os.path.join(temp_dir, "user_facts.json")
        with open(facts_json_path, "w", encoding="utf-8") as f:
            json.dump(facts_data, f, indent=2)

        # Create a clean SQLite snapshot using SQLite backup API if available
        db_snap_path = os.path.join(temp_dir, "conversations.db")
        try:
            with archive._lock:
                src_conn = archive._get_connection()
                try:
                    src_conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                except Exception:
                    pass
                dest_conn = sqlite3.connect(db_snap_path)
                try:
                    src_conn.backup(dest_conn)
                finally:
                    dest_conn.close()
        except Exception:
            pass

        # Package zip
        with zipfile.ZipFile(target_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(manifest_path, arcname="manifest.json")
            zf.write(persona_out_path, arcname="personality.md")
            zf.write(conv_json_path, arcname="conversations.json")
            zf.write(facts_json_path, arcname="user_facts.json")
            if os.path.exists(db_snap_path):
                zf.write(db_snap_path, arcname="conversations.db")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return target_zip


def import_backup_bundle(
    archive: SQLiteArchive,
    personality_path: str,
    backup_path: str,
    mode: str = "replace",
) -> Dict[str, Any]:
    """Import conversation data and personality.md from a backup bundle zip.

    Args:
        archive: The active SQLiteArchive instance.
        personality_path: Target path for restoring personality.md.
        backup_path: Path to the .zip backup bundle.
        mode: 'replace' to wipe existing conversations, or 'merge' to combine.

    Returns:
        Summary dict of restoration operations.
    """
    clean_path = os.path.abspath(backup_path)
    if not os.path.exists(clean_path):
        raise FileNotFoundError(f"Backup file not found at: {clean_path}")

    if not zipfile.is_zipfile(clean_path):
        raise ValueError(f"File is not a valid zip archive: {clean_path}")

    temp_dir = tempfile.mkdtemp(prefix="assistant_import_")
    try:
        with zipfile.ZipFile(clean_path, "r") as zf:
            zf.extractall(temp_dir)

        # 1. Read and validate manifest
        manifest_file = os.path.join(temp_dir, "manifest.json")
        manifest_data = {}
        if os.path.exists(manifest_file):
            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)

        # 2. Restore personality.md
        persona_src = os.path.join(temp_dir, "personality.md")
        persona_restored = False
        if os.path.exists(persona_src):
            target_p = Path(personality_path)
            if target_p.exists():
                bak_p = Path(f"{personality_path}.bak")
                shutil.copyfile(target_p, bak_p)
            else:
                target_p.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(persona_src, target_p)
            persona_restored = True

        # 3. Restore conversations
        conv_json_path = os.path.join(temp_dir, "conversations.json")
        sessions_restored = 0
        messages_restored = 0

        if os.path.exists(conv_json_path):
            with open(conv_json_path, "r", encoding="utf-8") as f:
                conv_data = json.load(f)
            stats = archive.import_conversations(conv_data, mode=mode)
            sessions_restored = stats.get("sessions_imported", 0)
            messages_restored = stats.get("messages_imported", 0)

        # 4. Restore user facts
        facts_json_path = os.path.join(temp_dir, "user_facts.json")
        facts_restored = 0
        if os.path.exists(facts_json_path) and hasattr(archive, "import_facts"):
            if mode == "replace" and hasattr(archive, "wipe_all_facts"):
                archive.wipe_all_facts()
            with open(facts_json_path, "r", encoding="utf-8") as f:
                facts_data = json.load(f)
            facts_restored = archive.import_facts(facts_data)

        return {
            "success": True,
            "backup_path": clean_path,
            "persona_restored": persona_restored,
            "sessions_restored": sessions_restored,
            "messages_restored": messages_restored,
            "facts_restored": facts_restored,
            "mode": mode,
            "manifest": manifest_data,
        }

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
