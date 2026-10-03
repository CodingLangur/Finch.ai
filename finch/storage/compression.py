"""Column-level compression strategies using Zstandard (zstd) and zlib fallback."""
import base64
import time
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    zstd = None
    ZSTD_AVAILABLE = False

import zlib


# Magic prefix for text-encoded compressed payloads
ZSTD_TEXT_PREFIX = "__ZSTD__:"
ZLIB_TEXT_PREFIX = "__ZLIB__:"


class ColumnCompressor:
    """Manages column-level data compression for large message bodies and tool outputs.
    
    Prefers Zstandard (zstd) for high compression ratios and sub-millisecond decompression speed.
    Falls back gracefully to standard-library zlib if zstandard is not installed.
    """

    def __init__(
        self,
        level: int = 3,
        min_size_bytes: int = 512,
        prefer_zstd: bool = True,
    ):
        """
        Args:
            level: Compression level (default: 3 for zstd, 6 for zlib).
            min_size_bytes: Minimum payload size in bytes to trigger compression.
            prefer_zstd: Use zstd if available, otherwise zlib.
        """
        self.level = level
        self.min_size_bytes = min_size_bytes
        self.use_zstd = prefer_zstd and ZSTD_AVAILABLE

        if self.use_zstd:
            self._cctx = zstd.ZstdCompressor(level=self.level)
            self._dctx = zstd.ZstdDecompressor()
        else:
            self._cctx = None
            self._dctx = None

    @property
    def algorithm(self) -> str:
        """Return the active compression algorithm name."""
        return "zstd" if self.use_zstd else "zlib"

    def is_compressed(self, data: Union[str, bytes]) -> bool:
        """Check if data contains our compression header."""
        if isinstance(data, str):
            return data.startswith(ZSTD_TEXT_PREFIX) or data.startswith(ZLIB_TEXT_PREFIX)
        elif isinstance(data, (bytes, bytearray)):
            # Check Zstandard magic frame header 0x28B52FFD or zlib headers (0x78)
            if len(data) >= 4 and data[:4] == b"\x28\xb5\x2f\xfd":
                return True
            if len(data) >= 2 and data[0] == 0x78:
                return True
            if data.startswith(ZSTD_TEXT_PREFIX.encode("ascii")) or data.startswith(ZLIB_TEXT_PREFIX.encode("ascii")):
                return True
        return False

    def compress_bytes(self, raw_bytes: bytes) -> Tuple[bytes, bool]:
        """Compress raw bytes. Returns (result_bytes, was_compressed)."""
        if len(raw_bytes) < self.min_size_bytes:
            return raw_bytes, False

        if self.use_zstd:
            compressed = self._cctx.compress(raw_bytes)
        else:
            compressed = zlib.compress(raw_bytes, level=min(self.level, 9))

        # Only use compressed if it actually saves space
        if len(compressed) < len(raw_bytes):
            return compressed, True
        return raw_bytes, False

    def decompress_bytes(self, data: bytes) -> bytes:
        """Decompress bytes if compressed, otherwise return as-is."""
        if not self.is_compressed(data):
            return data

        # Check if text-prefixed binary
        if data.startswith(ZSTD_TEXT_PREFIX.encode("ascii")):
            b64_payload = data[len(ZSTD_TEXT_PREFIX):]
            compressed = base64.b64decode(b64_payload)
            if self.use_zstd:
                return self._dctx.decompress(compressed)
            return zlib.decompress(compressed)

        if data.startswith(ZLIB_TEXT_PREFIX.encode("ascii")):
            b64_payload = data[len(ZLIB_TEXT_PREFIX):]
            compressed = base64.b64decode(b64_payload)
            return zlib.decompress(compressed)

        # Raw frame bytes
        if len(data) >= 4 and data[:4] == b"\x28\xb5\x2f\xfd":
            if self.use_zstd:
                return self._dctx.decompress(data)
            raise RuntimeError("Cannot decompress zstd frame: zstandard library not available")

        if len(data) >= 2 and data[0] == 0x78:
            return zlib.decompress(data)

        return data

    def compress_text(self, text: str) -> Tuple[str, bool]:
        """Compress text to prefixed base64 string. Returns (result_text, was_compressed)."""
        raw_bytes = text.encode("utf-8")
        if len(raw_bytes) < self.min_size_bytes:
            return text, False

        if self.use_zstd:
            compressed = self._cctx.compress(raw_bytes)
            prefix = ZSTD_TEXT_PREFIX
        else:
            compressed = zlib.compress(raw_bytes, level=min(self.level, 9))
            prefix = ZLIB_TEXT_PREFIX

        # Encoded text length check (base64 adds ~33% overhead over raw compressed bytes)
        b64_str = base64.b64encode(compressed).decode("ascii")
        result_str = prefix + b64_str

        if len(result_str) < len(text):
            return result_str, True
        return text, False

    def decompress_text(self, text: str) -> str:
        """Decompress prefixed text if compressed, otherwise return unmodified."""
        if not isinstance(text, str):
            return text

        if text.startswith(ZSTD_TEXT_PREFIX):
            b64_payload = text[len(ZSTD_TEXT_PREFIX):]
            compressed = base64.b64decode(b64_payload)
            if self.use_zstd:
                raw_bytes = self._dctx.decompress(compressed)
            else:
                # If zstd not installed but compressed with zstd, try zlib or raise informative error
                try:
                    raw_bytes = zlib.decompress(compressed)
                except Exception as e:
                    raise RuntimeError("Payload was compressed with zstandard; install 'zstandard' to decompress.") from e
            return raw_bytes.decode("utf-8", errors="replace")

        if text.startswith(ZLIB_TEXT_PREFIX):
            b64_payload = text[len(ZLIB_TEXT_PREFIX):]
            compressed = base64.b64decode(b64_payload)
            raw_bytes = zlib.decompress(compressed)
            return raw_bytes.decode("utf-8", errors="replace")

        return text

    def benchmark(self, samples: List[str]) -> Dict[str, Any]:
        """Run benchmark metrics over a list of sample text payloads."""
        total_raw_bytes = sum(len(s.encode("utf-8")) for s in samples)
        total_comp_bytes = 0
        comp_time_total = 0.0
        decomp_time_total = 0.0
        compressed_count = 0

        for s in samples:
            t0 = time.perf_counter()
            comp_res, was_comp = self.compress_text(s)
            t1 = time.perf_counter()
            comp_time_total += (t1 - t0)

            if was_comp:
                compressed_count += 1
                total_comp_bytes += len(comp_res.encode("utf-8"))
                t2 = time.perf_counter()
                decomp_res = self.decompress_text(comp_res)
                t3 = time.perf_counter()
                decomp_time_total += (t3 - t2)
                assert decomp_res == s, "Decompression verification failed!"
            else:
                total_comp_bytes += len(s.encode("utf-8"))

        savings_bytes = max(0, total_raw_bytes - total_comp_bytes)
        savings_pct = round((savings_bytes / total_raw_bytes * 100.0), 2) if total_raw_bytes > 0 else 0.0
        compression_ratio = round(total_raw_bytes / total_comp_bytes, 2) if total_comp_bytes > 0 else 1.0

        return {
            "algorithm": self.algorithm,
            "sample_count": len(samples),
            "compressed_count": compressed_count,
            "total_raw_bytes": total_raw_bytes,
            "total_compressed_bytes": total_comp_bytes,
            "savings_bytes": savings_bytes,
            "savings_pct": savings_pct,
            "compression_ratio": compression_ratio,
            "avg_compress_time_ms": round((comp_time_total / len(samples) * 1000.0), 3) if samples else 0.0,
            "avg_decompress_time_ms": round((decomp_time_total / max(1, compressed_count) * 1000.0), 3) if compressed_count else 0.0,
        }


# Global singleton instance
default_column_compressor = ColumnCompressor()
