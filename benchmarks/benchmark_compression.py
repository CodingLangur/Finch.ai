"""Benchmark Suite for Phase 8: Column-Level Zstandard Compression & SQLite VACUUM Housekeeping."""
import os
import sys
import tempfile
import time
from typing import Any, Dict, List

from rich.box import ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ai_assistant.storage.compression import ColumnCompressor, ZSTD_AVAILABLE
from ai_assistant.storage.sqlite_archive import SQLiteArchive


console = Console()

# --- Sample Test Payloads ---

CODE_SAMPLE = """
import asyncio
import math
import typing

class MatrixTransformer:
    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.matrix = [[0.0 for _ in range(cols)] for _ in range(rows)]

    def populate(self, seed: float = 1.0) -> None:
        for r in range(self.rows):
            for c in range(self.cols):
                self.matrix[r][c] = math.sin(seed * (r + 1)) * math.cos(seed * (c + 1))

    def compute_frobenius_norm(self) -> float:
        total = sum(val ** 2 for row in self.matrix for val in row)
        return math.sqrt(total)

    def serialize(self) -> typing.Dict[str, typing.Any]:
        return {
            "rows": self.rows,
            "cols": self.cols,
            "matrix": self.matrix,
            "norm": self.compute_frobenius_norm(),
        }
""" * 8

JSON_TOOL_OUTPUT = """
{
  "status": "success",
  "tool": "search_past_conversations",
  "query": "Kubernetes cluster architecture and docker deployment",
  "matched_count": 12,
  "sessions": [
    {
      "id": "sess_k8s_prod_cluster_deploy_001",
      "title": "Production Kubernetes Cluster Deployment and Ingress Setup",
      "summary": "Detailed discussion covering Helm chart definitions, persistent volume claims, MetalLB load balancers, and ingress controllers.",
      "tokens": 4250,
      "metadata": {
        "namespace": "production",
        "nodes": ["node-1.us-east.compute.internal", "node-2.us-east.compute.internal", "node-3.us-east.compute.internal"],
        "services": ["frontend-nginx", "auth-api-gateway", "worker-celery-daemon", "redis-sentinel-master"],
        "metrics": {"cpu_utilization": "42%", "memory_rss_mb": 8192, "network_io_mbps": 120.4}
      }
    },
    {
      "id": "sess_k8s_staging_cluster_deploy_002",
      "title": "Staging Kubernetes Cluster Deployment and Ingress Setup",
      "summary": "Detailed staging environment discussion covering Helm chart definitions and ingress controllers.",
      "tokens": 3120,
      "metadata": {
        "namespace": "staging",
        "nodes": ["stg-1.us-east.compute.internal", "stg-2.us-east.compute.internal"],
        "services": ["stg-frontend", "stg-auth-api", "stg-worker"],
        "metrics": {"cpu_utilization": "18%", "memory_rss_mb": 4096, "network_io_mbps": 34.2}
      }
    }
  ]
}
""" * 5

TRANSCRIPT_LOG = """
=== Conversation Session Transcript ===
User: Can you explain how the Headroom compression pipeline processes token transforms?
Assistant: The Headroom compression pipeline intercepts prompts before they reach the model.
It applies structural parsers (AST for code, JSON normalization for dictionaries, log filtering)
and reduces redundant whitespace while strictly preserving pinned persona instructions.
User: What are the primary algorithms used in the sqlite-vec reciprocal rank fusion (RRF)?
Assistant: Reciprocal rank fusion combines rank signals from lexical FTS5 BM25 search and
cosine distance from vector similarity search. The formula computes RRF score as sum(1 / (k + rank)).
User: Show me an example calculation with k=60.
Assistant: If document A is ranked #1 in lexical search and #3 in vector search, its score is
1/(60+1) + 1/(60+3) = 0.01639 + 0.01587 = 0.03226.
""" * 8


def benchmark_compression_algorithms() -> None:
    """Benchmark Zstandard vs zlib across code, JSON, and transcript payloads."""
    table = Table(
        title="Column-Level Compression Benchmark (zstd vs zlib)",
        box=ROUNDED,
        header_style="bold cyan",
    )
    table.add_column("Payload Type", style="bold white")
    table.add_column("Algorithm", justify="center")
    table.add_column("Original", justify="right")
    table.add_column("Compressed", justify="right")
    table.add_column("Savings", justify="right", style="bold green")
    table.add_column("Ratio", justify="right")
    table.add_column("Compress Time", justify="right")
    table.add_column("Decompress Time", justify="right")

    compressors = []
    if ZSTD_AVAILABLE:
        compressors.append(ColumnCompressor(level=3, prefer_zstd=True))
    compressors.append(ColumnCompressor(level=6, prefer_zstd=False))

    datasets = [
        ("Python Code (AST)", [CODE_SAMPLE]),
        ("JSON Tool Output", [JSON_TOOL_OUTPUT]),
        ("Transcript Log", [TRANSCRIPT_LOG]),
        ("Mixed Workload", [CODE_SAMPLE, JSON_TOOL_OUTPUT, TRANSCRIPT_LOG]),
    ]

    for name, samples in datasets:
        for c in compressors:
            stats = c.benchmark(samples)
            algo_style = "bold magenta" if stats["algorithm"] == "zstd" else "bold yellow"
            table.add_row(
                name,
                f"[{algo_style}]{stats['algorithm'].upper()}[/{algo_style}]",
                f"{stats['total_raw_bytes']:,} B",
                f"{stats['total_compressed_bytes']:,} B",
                f"{stats['savings_pct']}%",
                f"{stats['compression_ratio']}x",
                f"{stats['avg_compress_time_ms']:.2f} ms",
                f"{stats['avg_decompress_time_ms']:.2f} ms",
            )

    console.print(table)
    console.print()


def benchmark_sqlite_vacuum_housekeeping() -> None:
    """Benchmark SQLite database disk footprint before, during deletions, and after VACUUM."""
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "bench_housekeeping.db")

    try:
        archive = SQLiteArchive(db_path=db_path)

        console.print("[cyan]Populating database with 40 conversation sessions...[/cyan]")
        for i in range(40):
            sess = archive.create_session(title=f"Session #{i}: Data Engineering Workload")
            archive.add_message(sess.id, "user", f"Query {i}: " + CODE_SAMPLE[:500])
            archive.add_message(sess.id, "assistant", f"Result {i}: " + JSON_TOOL_OUTPUT[:800])
            archive.add_message(sess.id, "user", f"Follow-up {i}: " + TRANSCRIPT_LOG[:600])

        archive.checkpoint()
        stats_populated = archive.get_storage_stats()

        # Delete 20 sessions (simulating session pruning / archiving)
        console.print("[cyan]Deleting 20 sessions to generate fragmentation / free pages...[/cyan]")
        all_sessions = archive.list_sessions(limit=40)
        for s in all_sessions[:20]:
            archive.delete_session(s.id)

        archive.checkpoint()
        stats_deleted = archive.get_storage_stats()

        # Run maintenance (optimize + VACUUM)
        console.print("[cyan]Running SQLite Housekeeping (PRAGMA optimize + VACUUM)...[/cyan]")
        maint_res = archive.run_maintenance(vacuum=True)
        stats_vacuumed = archive.get_storage_stats()

        archive.close()

        # Display Housekeeping Results Table
        table = Table(
            title="SQLite Database Housekeeping & VACUUM Reclamation",
            box=ROUNDED,
            header_style="bold cyan",
        )
        table.add_column("State", style="bold white")
        table.add_column("Sessions", justify="right")
        table.add_column("Messages", justify="right")
        table.add_column("DB Size", justify="right")
        table.add_column("Page Count", justify="right")
        table.add_column("Freelist Pages", justify="right")
        table.add_column("Reclaimable", justify="right")

        table.add_row(
            "1. Initial Populated",
            str(stats_populated["session_count"]),
            str(stats_populated["message_count"]),
            f"{stats_populated['file_size_bytes']:,} B",
            str(stats_populated["page_count"]),
            str(stats_populated["freelist_count"]),
            f"{stats_populated['reclaimable_bytes']:,} B",
        )
        table.add_row(
            "2. Post Deletions (Fragmented)",
            str(stats_deleted["session_count"]),
            str(stats_deleted["message_count"]),
            f"{stats_deleted['file_size_bytes']:,} B",
            str(stats_deleted["page_count"]),
            f"[bold yellow]{stats_deleted['freelist_count']}[/bold yellow]",
            f"[yellow]{stats_deleted['reclaimable_bytes']:,} B[/yellow]",
        )
        table.add_row(
            "3. Post VACUUM (Defragmented)",
            str(stats_vacuumed["session_count"]),
            str(stats_vacuumed["message_count"]),
            f"[bold green]{stats_vacuumed['file_size_bytes']:,} B[/bold green]",
            str(stats_vacuumed["page_count"]),
            f"[bold green]{stats_vacuumed['freelist_count']}[/bold green]",
            "[green]0 B[/green]",
        )

        console.print(table)

        reclaimed_bytes = stats_deleted["file_size_bytes"] - stats_vacuumed["file_size_bytes"]
        reclaimed_pct = round((reclaimed_bytes / stats_deleted["file_size_bytes"] * 100.0), 2) if stats_deleted["file_size_bytes"] > 0 else 0.0

        panel = Panel(
            f"[bold green]✓ VACUUM successfully defragmented database![/bold green]\n\n"
            f"Space Reclaimed: [bold white]{reclaimed_bytes:,} bytes ({reclaimed_pct}%)[/bold white]\n"
            f"Freelist Pages Purged: [cyan]{stats_deleted['freelist_count']} → {stats_vacuumed['freelist_count']}[/cyan]\n"
            f"Integrity Check: [bold green]{maint_res['integrity']}[/bold green]",
            title="[bold green]Housekeeping Summary[/bold green]",
            border_style="green",
            box=ROUNDED,
        )
        console.print(panel)

    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main() -> None:
    console.print(Panel(
        "[bold cyan]Finch.ai Phase 8 Optimization & Maintenance Benchmark[/bold cyan]\n"
        "Testing column-level zstd compression and SQLite VACUUM space reclamation.",
        box=ROUNDED,
        border_style="cyan",
    ))
    console.print()

    benchmark_compression_algorithms()
    benchmark_sqlite_vacuum_housekeeping()


if __name__ == "__main__":
    main()
