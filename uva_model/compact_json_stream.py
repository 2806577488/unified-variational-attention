"""
单遍流式紧凑 JSON 重写（依赖 ijson）：不把整份文件解析成一棵 Python dict 树。

印记 / token_meta 等按「小对象」缓冲后写出，峰值内存为常数级字段数，不随全库 token 线性增长。
"""

from __future__ import annotations

import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Tuple

from .checkpoint_json import open_json_read_stream

try:
    import ijson  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    ijson = None  # type: ignore[assignment]

Event = Tuple[str, str, Any]

_IMPRINT_ITEM = re.compile(r"^word_imprints\.tokens\.[^.]+\.item$")
_TOKEN_META_OBJ = re.compile(r"^word_imprints\.token_meta\.[^.]+$")
_ASSOC_INNER = re.compile(r"^preference_state\.association_value\.[^.]+$")

_WI_ZERO_KEYS = frozenset(
    {
        "seq",
        "cold_scan_cursor",
        "merged_imprints_total",
        "associated_activated_total",
        "cold_probe_total",
    }
)


def _norm(v: Any) -> Any:
    if isinstance(v, Decimal):
        return float(v)
    return v


def _emit_scalar(out, event: str, value: Any) -> None:
    v = _norm(value)
    if event == "number":
        if isinstance(v, bool):
            out.write(b"true" if v else b"false")
        elif isinstance(v, int) and not isinstance(v, bool):
            out.write(str(int(v)).encode("utf-8"))
        else:
            out.write(repr(float(v)).encode("utf-8"))
    elif event == "string":
        out.write(json.dumps(str(v), ensure_ascii=False).encode("utf-8"))
    elif event == "boolean":
        out.write(b"true" if v else b"false")
    elif event == "null":
        out.write(b"null")
    else:
        out.write(json.dumps(v, ensure_ascii=False).encode("utf-8"))


def _skip_value(it: Iterator[Event], first: Event) -> None:
    _, event, _ = first
    if event in ("number", "string", "boolean", "null"):
        return
    if event == "start_map":
        d = 1
        while d:
            _, e, _ = next(it)
            if e == "start_map":
                d += 1
            elif e == "end_map":
                d -= 1
        return
    if event == "start_array":
        d = 1
        while d:
            _, e, _ = next(it)
            if e == "start_array":
                d += 1
            elif e == "end_array":
                d -= 1


def _optimize_imprint(d: Dict[str, Any]) -> None:
    if not str(d.get("context_before", "") or ""):
        d.pop("context_before", None)
    if not str(d.get("context_after", "") or ""):
        d.pop("context_after", None)
    try:
        oc = int(d.get("occurrence_count", 1))
    except (TypeError, ValueError):
        oc = 1
    if oc == 1:
        d.pop("occurrence_count", None)


def _optimize_meta(d: Dict[str, Any]) -> None:
    if d.get("last_access_seq", 0) == 0:
        d.pop("last_access_seq", None)
    if d.get("last_trigger_seq", 0) == 0:
        d.pop("last_trigger_seq", None)
    if d.get("trigger_success_count", 0) == 0:
        d.pop("trigger_success_count", None)
    if d.get("is_dormant", False) is False:
        d.pop("is_dormant", None)


def _read_flat_map(it: Iterator[Event], obj_prefix: str) -> Dict[str, Any]:
    d: Dict[str, Any] = {}
    while True:
        ev = next(it)
        p, e, v = ev
        if p == obj_prefix and e == "end_map":
            return d
        if p == obj_prefix and e == "map_key":
            key = str(v)
            ev2 = next(it)
            p2, e2, v2 = ev2
            if e2 not in ("number", "string", "boolean", "null"):
                raise ValueError(
                    f"流式 optimize 仅支持扁平对象: {obj_prefix!r} 键 {key!r} 得到 {ev2!r}"
                )
            d[key] = _norm(v2)
        else:
            raise ValueError(f"read_flat_map 未预期事件: {ev!r}")


def _is_zero_number(ev: Event) -> bool:
    _, e, v = ev
    if e != "number":
        return False
    try:
        return float(_norm(v)) == 0.0
    except (TypeError, ValueError):
        return False


def _strip_matches(nxt: Event, key: str, default: Any) -> bool:
    np, ne, nv = nxt
    if np != key:
        return False
    if ne == "number":
        try:
            if isinstance(default, bool):
                return False
            if isinstance(default, int):
                return int(nv) == int(default)
            return float(nv) == float(default)
        except (TypeError, ValueError):
            return False
    if ne == "string":
        return str(nv) == str(default)
    if ne == "boolean":
        return bool(nv) == bool(default)
    return False


def _write_value(
    it: Iterator[Event],
    out,
    first: Event,
    *,
    tokenizer_strip: Optional[Mapping[str, Any]],
    optimize_dialogue: bool,
) -> None:
    p, e, v = first
    if e in ("number", "string", "boolean", "null"):
        _emit_scalar(out, e, v)
        return
    if e == "start_map":
        if optimize_dialogue and _IMPRINT_ITEM.match(p):
            d = _read_flat_map(it, p)
            _optimize_imprint(d)
            out.write(
                json.dumps(d, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            )
            return
        if optimize_dialogue and _TOKEN_META_OBJ.match(p):
            d = _read_flat_map(it, p)
            _optimize_meta(d)
            out.write(
                json.dumps(d, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            )
            return
        out.write(b"{")
        first_key = True
        while True:
            ev = next(it)
            pp, ee, vv = ev
            if pp == p and ee == "end_map":
                out.write(b"}")
                return
            if pp != p or ee != "map_key":
                raise ValueError(f"map 内期望 map_key: {ev!r}")
            key = str(vv)
            nxt = next(it)
            if tokenizer_strip is not None and p == "" and key in tokenizer_strip:
                if _strip_matches(nxt, key, tokenizer_strip[key]):
                    _skip_value(it, nxt)
                    continue
            if optimize_dialogue and p == "word_imprints" and key in _WI_ZERO_KEYS:
                if _is_zero_number(nxt):
                    _skip_value(it, nxt)
                    continue
            if optimize_dialogue and p in (
                "preference_state.branch_bias",
                "preference_state.token_value",
            ):
                if _is_zero_number(nxt):
                    _skip_value(it, nxt)
                    continue
            if optimize_dialogue and _ASSOC_INNER.match(p):
                if _is_zero_number(nxt):
                    _skip_value(it, nxt)
                    continue
            if not first_key:
                out.write(b",")
            first_key = False
            out.write(json.dumps(key, ensure_ascii=False).encode("utf-8") + b":")
            _write_value(
                it,
                out,
                nxt,
                tokenizer_strip=tokenizer_strip,
                optimize_dialogue=optimize_dialogue,
            )
        return
    if e == "start_array":
        out.write(b"[")
        first_el = True
        while True:
            ev = next(it)
            pp, ee, vv = ev
            if pp == p and ee == "end_array":
                out.write(b"]")
                return
            if not first_el:
                out.write(b",")
            first_el = False
            _write_value(
                it,
                out,
                ev,
                tokenizer_strip=tokenizer_strip,
                optimize_dialogue=optimize_dialogue,
            )
        return
    raise ValueError(f"_write_value 未处理: {first!r}")


def stream_compact_checkpoint(
    src: Path,
    dst: Path,
    *,
    tokenizer_strip: Optional[Mapping[str, Any]] = None,
    optimize_dialogue: bool = False,
) -> Tuple[int, int]:
    if ijson is None:
        raise ImportError("流式模式需要: pip install ijson")
    in_size = int(src.stat().st_size)
    src_f = open_json_read_stream(src)
    try:
        it = iter(ijson.parse(src_f))
        ev0 = next(it)
        if ev0 != ("", "start_map", None):
            raise ValueError("期望 JSON 根为对象")
        dst.parent.mkdir(parents=True, exist_ok=True)
        out_f = dst.open("wb")
        try:
            out_f.write(b"{")
            first_key = True
            while True:
                ev = next(it)
                p, e, v = ev
                if p == "" and e == "end_map":
                    out_f.write(b"}")
                    break
                if p != "" or e != "map_key":
                    raise ValueError(f"根级期望 map_key: {ev!r}")
                key = str(v)
                nxt = next(it)
                if tokenizer_strip is not None and key in tokenizer_strip:
                    if _strip_matches(nxt, key, tokenizer_strip[key]):
                        _skip_value(it, nxt)
                        continue
                if not first_key:
                    out_f.write(b",")
                first_key = False
                out_f.write(json.dumps(key, ensure_ascii=False).encode("utf-8") + b":")
                _write_value(
                    it,
                    out_f,
                    nxt,
                    tokenizer_strip=tokenizer_strip,
                    optimize_dialogue=optimize_dialogue,
                )
            out_size = out_f.tell()
        finally:
            out_f.close()
    finally:
        src_f.close()
    return in_size, out_size


def stream_available() -> bool:
    return ijson is not None


__all__ = ["stream_compact_checkpoint", "stream_available"]
