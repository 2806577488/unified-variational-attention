"""
对话预热可用的「规范语料」JSONL 流式读入。

设计参照「数据治理分层」思路中的 **输出侧契约**（UTF-8、一行一条、正文 + 可选元数据），
**不包含**完整采集/萃取/去重/Airflow 等离线流水线——仅实现训练进程侧的 IO 与轻量解析。

行格式示例::

    {"text": "一段正文", "meta": {"src": "wiki", "id": "..."}}

亦接受 ``content`` 作为正文字段别名；支持 ``.jsonl.gz``。
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from io import TextIOWrapper
from pathlib import Path
from typing import BinaryIO, Dict, IO, Iterator, List, Union


def _open_text_lines(path: Path) -> IO[str]:
    p = Path(path)
    if p.suffix.lower() == ".gz" or str(p).lower().endswith(".jsonl.gz"):
        raw: BinaryIO = gzip.open(p, "rb")
        return TextIOWrapper(raw, encoding="utf-8", newline="")
    return p.open("r", encoding="utf-8", newline="")


def iter_corpus_jsonl_objects(
    path: Union[str, Path],
    *,
    skip_bad_lines: bool = False,
) -> Iterator[Dict[str, object]]:
    """逐行解析为 dict；空行跳过。"""
    pp = Path(path)
    with _open_text_lines(pp) as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                if skip_bad_lines:
                    continue
                raise ValueError(f"无效 JSON：{pp}:{lineno}") from None
            if not isinstance(obj, dict):
                if skip_bad_lines:
                    continue
                raise TypeError(f"每行必须是 JSON 对象：{pp}:{lineno}")
            yield obj


def _row_text(row: Dict[str, object], *, strip_text: bool) -> str | None:
    for key in ("text", "content"):
        v = row.get(key)
        if not isinstance(v, str):
            continue
        if strip_text:
            s = v.strip()
            if s:
                return s
        elif v.strip():
            return v
    return None


@dataclass(frozen=True)
class CorpusJsonlRecord:
    text: str
    meta: Dict[str, object]
    raw: Dict[str, object]


def iter_corpus_jsonl_records(
    path: Union[str, Path],
    *,
    strip_text: bool = True,
    skip_bad_lines: bool = False,
    strict: bool = False,
) -> Iterator[CorpusJsonlRecord]:
    """产出正文 + meta（meta 缺省为空 dict）。"""
    for row in iter_corpus_jsonl_objects(path, skip_bad_lines=skip_bad_lines):
        text = _row_text(row, strip_text=strip_text)
        if text is None:
            if strict:
                raise ValueError(f"缺少 text/content：{path}")
            continue
        meta_raw = row.get("meta")
        meta = dict(meta_raw) if isinstance(meta_raw, dict) else {}
        yield CorpusJsonlRecord(text=text, meta=meta, raw=dict(row))


def iter_corpus_jsonl_training_episodes(
    paths: List[Union[str, Path]],
    *,
    episode_size: int = 4,
    min_episode_turns: int = 1,
    skip_bad_lines: bool = False,
    strict: bool = False,
    strip_text: bool = True,
) -> Iterator[List[str]]:
    """
    多文件顺序拼接后，按固定窗口切成「多轮」episode（每元素一条用户侧字符串）。

    常数内存：只保留当前窗口缓冲。
    """
    episode_size = max(1, int(episode_size))
    min_episode_turns = max(1, int(min_episode_turns))
    expanded = [Path(p).expanduser() for p in paths if str(p).strip()]
    if not expanded:
        raise ValueError("至少需要一条 JSONL 路径")
    cur: List[str] = []
    for p in expanded:
        if not p.is_file():
            raise FileNotFoundError(f"JSONL 不存在：{p}")
        for row in iter_corpus_jsonl_records(
            p,
            strip_text=strip_text,
            skip_bad_lines=skip_bad_lines,
            strict=strict,
        ):
            cur.append(row.text)
            if len(cur) >= episode_size:
                yield list(cur)
                cur = []
    if len(cur) >= min_episode_turns:
        yield list(cur)


def load_corpus_jsonl_episodes_list(
    paths: List[Union[str, Path]],
    *,
    episode_size: int = 4,
    min_episode_turns: int = 1,
    skip_bad_lines: bool = False,
) -> List[List[str]]:
    """一次性装入内存（仅小数据或调试）。"""
    return list(
        iter_corpus_jsonl_training_episodes(
            paths,
            episode_size=episode_size,
            min_episode_turns=min_episode_turns,
            skip_bad_lines=skip_bad_lines,
        )
    )
