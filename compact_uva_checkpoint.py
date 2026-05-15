#!/usr/bin/env python3
"""
离线缩小 UVA 分词器 / 对话 checkpoint 的 JSON 体积。

1) 紧凑序列化：单行、无缩进（默认始终启用）。
2) 可选 --optimize：去掉与默认值等价、加载时会回填的字段（见下方）。
3) 默认 --stream：用 ijson 单遍流式读写，不把整份 JSON 解析成一棵大 dict（省内存）。
   若未安装 ijson，将自动退回整文件解析；可用 --no-stream 强制旧行为。

只读写你指定的路径；默认禁止输出路径 == 输入路径。

示例::

  pip install ijson
  python compact_uva_checkpoint.py ^
    --tokenizer-in a.json --tokenizer-out a.small.json ^
    --dialogue-in b.dialogue.json --dialogue-out b.small.dialogue.json ^
    --optimize
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Tuple, Union

# 与 PrecisionTokenizer.from_dict 中超参数默认值保持一致。
_TOKENIZER_STRIP_IF_EQUAL: Mapping[str, Union[int, float]] = {
    "alpha": 1.1,
    "beta": 0.3,
    "pi_min": 0.2,
    "sigma0": 0.8,
    "decay": 0.25,
    "boundary_threshold": 0.8,
    "surprise_threshold": 1.6,
    "dt": 0.8,
    "R_max": 1.0,
    "rho": 0.12,
    "lambda_deplete": 0.04,
    "tau_m": 8.0,
    "theta_F": 1.2,
    "R_crit": 0.35,
    "auto_rest_threshold": 0.2,
    "auto_resume_threshold": 0.5,
    "auto_rest_steps": 8,
    "R_base": 0.5,
    "R_max_cap": 3.0,
    "tau_grow": 400.0,
    "eta_learn": 0.04,
    "lambda_grow": 0.002,
    "F_ema_beta": 0.04,
}

_PREFERENCE_STATE_FORMAT = "preference_state_v1"

try:
    from uva_model.word_imprints import WORD_STATE_MEMORY_FORMAT
except ImportError:  # pragma: no cover
    WORD_STATE_MEMORY_FORMAT = "word_state_memory_v1"

JsonDict = Dict[str, Any]


def _optimize_tokenizer_dict(data: JsonDict) -> None:
    for key, default in _TOKENIZER_STRIP_IF_EQUAL.items():
        if key not in data:
            continue
        val = data[key]
        if key == "auto_rest_steps":
            try:
                ok = int(val) == int(default)
            except (TypeError, ValueError):
                ok = False
        else:
            try:
                ok = float(val) == float(default)
            except (TypeError, ValueError):
                ok = False
        if ok:
            del data[key]


def _optimize_word_imprints(wi: MutableMapping[str, Any]) -> None:
    if wi.get("format") != WORD_STATE_MEMORY_FORMAT:
        return
    for zk in (
        "seq",
        "cold_scan_cursor",
        "merged_imprints_total",
        "associated_activated_total",
        "cold_probe_total",
    ):
        if wi.get(zk, None) == 0:
            wi.pop(zk, None)
    tokens = wi.get("tokens")
    if isinstance(tokens, dict):
        for _tok, lst in tokens.items():
            if not isinstance(lst, list):
                continue
            for item in lst:
                if not isinstance(item, dict):
                    continue
                if not str(item.get("context_before", "") or ""):
                    item.pop("context_before", None)
                if not str(item.get("context_after", "") or ""):
                    item.pop("context_after", None)
                oc = item.get("occurrence_count", 1)
                try:
                    oc_int = int(oc)
                except (TypeError, ValueError):
                    oc_int = 1
                if oc_int == 1:
                    item.pop("occurrence_count", None)
    meta = wi.get("token_meta")
    if isinstance(meta, dict):
        for _tok, m in meta.items():
            if not isinstance(m, dict):
                continue
            if m.get("last_access_seq", 0) == 0:
                m.pop("last_access_seq", None)
            if m.get("last_trigger_seq", 0) == 0:
                m.pop("last_trigger_seq", None)
            if m.get("trigger_success_count", 0) == 0:
                m.pop("trigger_success_count", None)
            if m.get("is_dormant", False) is False:
                m.pop("is_dormant", None)


def _float_or_zero(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _optimize_preference_state(ps: MutableMapping[str, Any]) -> None:
    if str(ps.get("format", "")) != _PREFERENCE_STATE_FORMAT:
        return
    for top_key in ("branch_bias", "token_value"):
        d = ps.get(top_key)
        if not isinstance(d, dict):
            continue
        dead = [k for k, v in d.items() if _float_or_zero(v) == 0.0]
        for k in dead:
            del d[k]
    av = ps.get("association_value")
    if isinstance(av, dict):
        for tr in list(av.keys()):
            inner = av.get(tr)
            if not isinstance(inner, dict):
                continue
            dead_inner = [a for a, v in inner.items() if _float_or_zero(v) == 0.0]
            for a in dead_inner:
                del inner[a]
            if not inner:
                del av[tr]


def _optimize_dialogue_dict(data: JsonDict) -> None:
    wi = data.get("word_imprints")
    if isinstance(wi, dict):
        _optimize_word_imprints(wi)
    ps = data.get("preference_state")
    if isinstance(ps, dict):
        _optimize_preference_state(ps)


def optimize_payload(kind: str, data: JsonDict, *, optimize: bool) -> JsonDict:
    out = copy.deepcopy(data)
    if not optimize:
        return out
    if kind == "tokenizer":
        _optimize_tokenizer_dict(out)
    elif kind == "dialogue":
        _optimize_dialogue_dict(out)
    return out


def _rewrite_buffered(
    src: Path,
    dst: Path,
    *,
    kind: str,
    optimize: bool,
) -> Tuple[int, int]:
    raw = src.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("顶层必须是 JSON 对象")
    data = optimize_payload(kind, data, optimize=optimize)
    out = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    encoded = out.encode("utf-8")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(encoded)
    return len(raw), len(encoded)


def _rewrite_stream(
    src: Path,
    dst: Path,
    *,
    kind: str,
    optimize: bool,
) -> Tuple[int, int]:
    from uva_model.compact_json_stream import stream_available, stream_compact_checkpoint

    if not stream_available():
        raise RuntimeError("ijson 未安装，无法流式处理。请 pip install ijson 或使用 --no-stream")
    strip = dict(_TOKENIZER_STRIP_IF_EQUAL) if (optimize and kind == "tokenizer") else None
    opt_dlg = optimize and kind == "dialogue"
    return stream_compact_checkpoint(
        src,
        dst,
        tokenizer_strip=strip,
        optimize_dialogue=opt_dlg,
    )


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="将分词器 / 对话 JSON 重写为紧凑 UTF-8 JSON；可选语义精简；默认流式（ijson）。"
    )
    p.add_argument("--tokenizer-in", type=str, default="", help="输入分词器 .json")
    p.add_argument("--tokenizer-out", type=str, default="", help="输出分词器 .json")
    p.add_argument("--dialogue-in", type=str, default="", help="输入对话 .dialogue.json")
    p.add_argument("--dialogue-out", type=str, default="", help="输出对话 JSON")
    p.add_argument(
        "--optimize",
        action="store_true",
        help="启用语义精简（分词器默认超参键、印记与 preference 零项等）。",
    )
    p.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="使用 ijson 流式单遍处理（默认开）。关闭: --no-stream",
    )
    p.add_argument(
        "--in-place",
        action="store_true",
        help="允许输出路径与输入路径相同（缓冲模式会先整读入内存）。",
    )
    args = p.parse_args(argv)

    tin = args.tokenizer_in.strip()
    tout = args.tokenizer_out.strip()
    din = args.dialogue_in.strip()
    dout = args.dialogue_out.strip()

    pairs: list[tuple[str, str, str]] = []
    if tin or tout:
        if not tin or not tout:
            p.error("处理分词器须同时提供 --tokenizer-in 与 --tokenizer-out")
        pairs.append(("tokenizer", tin, tout))
    if din or dout:
        if not din or not dout:
            p.error("处理对话须同时提供 --dialogue-in 与 --dialogue-out")
        pairs.append(("dialogue", din, dout))

    if not pairs:
        p.error("至少提供一对分词器或对话的输入/输出路径")

    opt_note = " +optimize" if args.optimize else ""
    for kind, ins, outs in pairs:
        src = Path(ins).resolve()
        dst = Path(outs).resolve()
        if not src.is_file():
            print(f"[错误] {kind}: 输入不存在: {src}", file=sys.stderr)
            return 2
        if not args.in_place and src == dst:
            print(
                f"[错误] {kind}: 输出与输入为同一文件，拒绝写入以免覆盖原件。"
                f" 请改用新文件名，或显式加 --in-place。",
                file=sys.stderr,
            )
            return 3
        use_stream = bool(args.stream)
        stream_note = ""
        try:
            if use_stream:
                try:
                    before, after = _rewrite_stream(
                        src, dst, kind=kind, optimize=args.optimize
                    )
                    stream_note = " stream"
                except (ImportError, RuntimeError) as e:
                    print(f"[警告] {kind}: {e}；改用整文件缓冲模式。", file=sys.stderr)
                    before, after = _rewrite_buffered(
                        src, dst, kind=kind, optimize=args.optimize
                    )
                    stream_note = " buffered"
            else:
                before, after = _rewrite_buffered(
                    src, dst, kind=kind, optimize=args.optimize
                )
                stream_note = " buffered"
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as e:
            print(f"[错误] {kind}: {e}", file=sys.stderr)
            return 4
        pct = (1.0 - after / max(before, 1)) * 100.0
        print(
            f"[完成] {kind}{opt_note}{stream_note}: {src} -> {dst} | "
            f"{before} -> {after} 字节（约减小 {pct:.1f}%）"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
