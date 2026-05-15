"""压缩或未压缩 UTF-8 JSON 的检查点读写（分词器 / 对话等）。"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, BinaryIO, Literal, Optional

Compression = Literal["gzip", "zstd"]

_ZSTD_MAGIC = bytes((0x28, 0xB5, 0x2F, 0xFD))


def infer_compression_from_path(path: str | Path) -> Optional[Compression]:
    s = str(path).lower()
    if s.endswith(".gz") or s.endswith(".json.gz"):
        return "gzip"
    if s.endswith(".zst") or s.endswith(".json.zst"):
        return "zstd"
    return None


def decompress_json_bytes(raw: bytes) -> bytes:
    if len(raw) >= 2 and raw[0] == 0x1F and raw[1] == 0x8B:
        return gzip.decompress(raw)
    if len(raw) >= 4 and raw[:4] == _ZSTD_MAGIC:
        try:
            import zstandard  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "检测到 zstd 压缩数据，需要安装: pip install zstandard"
            ) from e
        return zstandard.ZstdDecompressor().decompress(raw)
    return raw


def open_json_read_stream(path: str | Path) -> BinaryIO:
    """
    以流方式打开 JSON 字节流（不解压整个文件到内存后再 parse）。
    支持明文、``.gz``、``.zst``（需 zstandard）。
    """
    p = Path(path)
    name = p.name.lower()
    if name.endswith(".gz") or str(p).lower().endswith(".json.gz"):
        return gzip.open(p, "rb")
    if name.endswith(".zst") or str(p).lower().endswith(".json.zst"):
        try:
            import zstandard  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError("读取 .zst 需要: pip install zstandard") from e
        return zstandard.open(p, "rb")
    return p.open("rb")


def read_json_document(path: str | Path) -> Any:
    """读取 JSON；支持明文 UTF-8、gzip（后缀 .gz 或 gzip 魔数）、zstd（后缀 .zst 或 zstd 魔数）。"""
    p = Path(path)
    raw = p.read_bytes()
    payload = decompress_json_bytes(raw)
    return json.loads(payload.decode("utf-8"))


def compress_payload(data: bytes, compression: Compression, *, gzip_level: int = 9) -> bytes:
    if compression == "gzip":
        return gzip.compress(data, compresslevel=gzip_level)
    if compression == "zstd":
        try:
            import zstandard  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "写入 zstd 需要安装: pip install zstandard"
            ) from e
        return zstandard.ZstdCompressor(level=19).compress(data)
    raise ValueError(f"未知 compression: {compression!r}")


def write_json_document(
    path: str | Path,
    data: Any,
    *,
    compact: bool = True,
    compression: Optional[Compression] = None,
) -> None:
    """写入 JSON；compression 为 None 时按路径后缀推断。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    enc_compression = compression if compression is not None else infer_compression_from_path(p)
    if compact:
        text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    else:
        text = json.dumps(data, ensure_ascii=False, indent=2)
    payload = text.encode("utf-8")
    if enc_compression is not None:
        payload = compress_payload(payload, enc_compression)
    p.write_bytes(payload)


__all__ = [
    "Compression",
    "compress_payload",
    "decompress_json_bytes",
    "infer_compression_from_path",
    "open_json_read_stream",
    "read_json_document",
    "write_json_document",
]
