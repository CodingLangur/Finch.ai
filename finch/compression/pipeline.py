"""Context Compression Pipeline utilizing Headroom."""
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from headroom import compress as headroom_compress, CompressConfig, CompressResult
    HEADROOM_AVAILABLE = True
except ImportError:
    HEADROOM_AVAILABLE = False
    headroom_compress = None
    CompressConfig = None
    CompressResult = None


@dataclass
class CompressionStats:
    """Telemetry captured from a single Headroom compression pass."""
    tokens_before: int = 0
    tokens_after: int = 0
    tokens_saved: int = 0
    compression_ratio: float = 0.0
    duration_ms: float = 0.0
    transforms_applied: List[str] = field(default_factory=list)

    @property
    def savings_pct(self) -> float:
        """Percentage of tokens saved."""
        if self.tokens_before <= 0:
            return 0.0
        return round((self.tokens_saved / self.tokens_before) * 100.0, 1)


class ContextCompressionPipeline:
    """Manages context compression using Headroom before LLM dispatch."""

    def __init__(
        self,
        enabled: bool = True,
        compress_user_messages: bool = True,
        compress_system_messages: bool = False,
        protect_recent: int = 2,
        min_tokens_to_compress: int = 100,
    ):
        self.enabled = enabled and HEADROOM_AVAILABLE
        self.compress_user_messages = compress_user_messages
        self.compress_system_messages = compress_system_messages
        self.protect_recent = protect_recent
        self.min_tokens_to_compress = min_tokens_to_compress

        # Cumulative session statistics
        self.total_tokens_saved: int = 0
        self.total_tokens_before: int = 0
        self.total_tokens_after: int = 0
        self.total_compressions: int = 0
        self.last_stats: Optional[CompressionStats] = None

    def toggle(self, enable: Optional[bool] = None) -> bool:
        """Toggle or set compression pipeline state."""
        if not HEADROOM_AVAILABLE:
            self.enabled = False
            return False

        if enable is None:
            self.enabled = not self.enabled
        else:
            self.enabled = enable
        return self.enabled

    def compress_messages(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4o",
    ) -> Tuple[List[Dict[str, str]], Optional[CompressionStats]]:
        """Compress input message list using Headroom.

        Args:
            messages: List of message dictionaries containing 'role' and 'content'.
            model: Model identifier for tokenizer alignment.

        Returns:
            Tuple of (compressed_messages, stats). If compression is skipped or fails,
            returns original messages and None.
        """
        if not self.enabled or not HEADROOM_AVAILABLE or not messages:
            return messages, None

        # Build Headroom compression configuration
        # System prompt (personality.md) is safeguarded when compress_system_messages=False
        cfg = CompressConfig(
            compress_user_messages=self.compress_user_messages,
            compress_system_messages=self.compress_system_messages,
            protect_recent=self.protect_recent,
            min_tokens_to_compress=self.min_tokens_to_compress,
            kompress_model="disabled",  # Use fast algorithmic / AST / JSON compression
        )

        start_time = time.perf_counter()
        try:
            result = headroom_compress(
                messages=messages,
                model=model,
                config=cfg,
            )
            duration_ms = (time.perf_counter() - start_time) * 1000.0

            stats = CompressionStats(
                tokens_before=result.tokens_before,
                tokens_after=result.tokens_after,
                tokens_saved=result.tokens_saved,
                compression_ratio=result.compression_ratio,
                duration_ms=round(duration_ms, 2),
                transforms_applied=result.transforms_applied or [],
            )

            # Update session cumulative tracking
            self.total_tokens_saved += stats.tokens_saved
            self.total_tokens_before += stats.tokens_before
            self.total_tokens_after += stats.tokens_after
            self.total_compressions += 1
            self.last_stats = stats

            return result.messages, stats

        except Exception as e:
            logger.warning(f"Headroom compression failed, falling back to uncompressed: {e}")
            return messages, None

    def get_summary(self) -> Dict[str, Any]:
        """Return cumulative compression metrics for the active session."""
        overall_ratio = 0.0
        if self.total_tokens_before > 0:
            overall_ratio = round((self.total_tokens_saved / self.total_tokens_before) * 100.0, 1)

        return {
            "enabled": self.enabled,
            "headroom_installed": HEADROOM_AVAILABLE,
            "total_compressions": self.total_compressions,
            "total_tokens_saved": self.total_tokens_saved,
            "total_tokens_before": self.total_tokens_before,
            "total_tokens_after": self.total_tokens_after,
            "overall_savings_pct": overall_ratio,
            "protect_recent": self.protect_recent,
            "compress_user_messages": self.compress_user_messages,
            "compress_system_messages": self.compress_system_messages,
        }
