from __future__ import annotations

import argparse
import json
import os
import queue
import random
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

try:
    import torch
except Exception:  # pragma: no cover - 可选依赖
    torch = None

from uva_model import (
    ArithmeticPredictiveCoder,
    ArithmeticProblem,
    CognitiveDialogueAgent,
    CurriculumA,
    DialogueTurn,
    Evaluator,
    InternalMonologue,
    PrecisionTokenizer,
    UnifiedVariationalAttentionModel,
)
from uva_model.corpus_jsonl import (
    iter_corpus_jsonl_training_episodes,
    load_corpus_jsonl_episodes_list,
)


def _dialogue_feedback_reward(cmd: str) -> Optional[float]:
    low = cmd.strip().lower()
    if low == "good":
        return 1.0
    if low == "bad":
        return -1.0
    if low == "meh":
        return 0.0
    return None


def fmt(values: List[float]) -> str:
    return "[" + ", ".join(f"{x:.4f}" for x in values) + "]"


def default_dialogue_model_path(tokenizer_dest: str) -> str:
    """与分词器同目录、同主文件名，扩展名为 .dialogue.json。"""
    p = Path(tokenizer_dest)
    return str(p.with_name(f"{p.stem}.dialogue.json"))


def resolve_dialogue_model_load_path(
    tokenizer_model: str,
    dialogue_model_arg: str,
    *,
    dialogue_fresh: bool = False,
) -> str:
    """
    决定启动时是否加载对话状态：
    - --dialogue-fresh：不加载；
    - 已传 --dialogue-model：用该路径（文件须存在）；
    - 否则若 default_dialogue_model_path(tokenizer_model) 存在则自动加载，否则初值。
    """
    if dialogue_fresh:
        return ""
    if dialogue_model_arg.strip():
        return dialogue_model_arg.strip()
    auto = default_dialogue_model_path(tokenizer_model)
    if Path(auto).is_file():
        return auto
    return ""


def load_dialogue_training_jsonl(path: str) -> List[List[str]]:
    """读取 JSONL 对话预热数据，每行形如 {"turns": [...]}。"""
    episodes: List[List[str]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"第 {line_no} 行不是 JSON 对象")
            turns_raw = item.get("turns")
            if not isinstance(turns_raw, list):
                raise ValueError(f"第 {line_no} 行缺少 turns 列表")
            turns = [str(x).strip() for x in turns_raw if str(x).strip()]
            if turns:
                episodes.append(turns)
    return episodes


def load_dialogue_training_corpus_jsonl(
    paths: List[str],
    *,
    episode_size: int = 4,
    min_episode_turns: int = 1,
    skip_bad_lines: bool = False,
) -> List[List[str]]:
    """
    规范语料 JSONL（``text`` + 可选 ``meta``）一次性载入；仅适合中小规模。

    大规模请用 :func:`iter_dialogue_training_corpus_episodes` + ``run_dialogue_training(..., episode_iter_factory=...)`` 边读边训。
    """
    expanded = [Path(p).expanduser() for p in paths if str(p).strip()]
    if not expanded:
        raise ValueError("语料 JSONL 需要至少一个路径")
    return load_corpus_jsonl_episodes_list(
        expanded,
        episode_size=episode_size,
        min_episode_turns=min_episode_turns,
        skip_bad_lines=skip_bad_lines,
    )


def load_dialogue_training_haid_jsonl(
    paths: List[str],
    *,
    episode_size: int = 4,
    min_episode_turns: int = 1,
    skip_bad_lines: bool = False,
) -> List[List[str]]:
    """弃用别名：请改用 :func:`load_dialogue_training_corpus_jsonl`。"""
    return load_dialogue_training_corpus_jsonl(
        paths,
        episode_size=episode_size,
        min_episode_turns=min_episode_turns,
        skip_bad_lines=skip_bad_lines,
    )


def iter_dialogue_training_corpus_episodes(
    paths: List[str],
    *,
    episode_size: int = 4,
    min_episode_turns: int = 1,
    skip_bad_lines: bool = False,
) -> Iterable[List[str]]:
    """按行流式读语料 JSONL，边读边产出 episode（常数内存）。"""
    expanded = [Path(p).expanduser() for p in paths if str(p).strip()]
    if not expanded:
        raise ValueError("语料 JSONL 需要至少一个路径")
    return iter_corpus_jsonl_training_episodes(
        expanded,
        episode_size=episode_size,
        min_episode_turns=min_episode_turns,
        skip_bad_lines=skip_bad_lines,
    )


def _iter_take(it: Iterable[List[str]], limit: int) -> Iterable[List[str]]:
    """截取前 ``limit`` 个 episode（用于语料流式下的 max-episodes）。"""
    n = 0
    for ep in it:
        yield ep
        n += 1
        if n >= limit:
            break


def clean_qq_dialogue_line(text: str) -> str:
    """
    QQ 真实对话的最小清洗：
    - 去首尾空白
    - 去明显链接
    - 去纯标点/纯数字
    - 去过长异常行
    """
    s = str(text).strip()
    if not s:
        return ""
    if len(s) > 280:
        return ""
    low = s.lower()
    if low.startswith(("http://", "https://", "www.")):
        return ""
    system_hints = (
        "撤回了一条消息",
        "文件已过期",
        "加入本群",
        "退出了群聊",
        "分享的文件",
        "邀请你加入",
        "拍了拍",
    )
    if any(hint in s for hint in system_hints):
        return ""
    if re.fullmatch(r"[0-9\s]+", s):
        return ""
    if not any(ch.isalnum() for ch in s):
        return ""
    return s


def is_qq_dialogue_boundary_line(text: str) -> bool:
    """空行或常见聊天记录元信息行：只作为边界，不作为训练内容。"""
    s = str(text).strip()
    if not s:
        return True
    return bool(
        re.fullmatch(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?", s)
        or re.fullmatch(r"\d{1,2}:\d{2}(?::\d{2})?", s)
        or re.fullmatch(r"\[\d{1,2}:\d{2}(?::\d{2})?\]", s)
        or re.fullmatch(
            r".{1,32}\s+\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?",
            s,
        )
        or re.fullmatch(r".{1,32}\s+\d{1,2}:\d{2}(?::\d{2})?", s)
    )


def build_dialogue_episodes_from_lines(
    lines: List[str],
    *,
    episode_size: int = 4,
    min_episode_turns: int = 2,
) -> List[List[str]]:
    """把原始逐行对话切成 episode：优先尊重空行/元信息边界，段内再做最小清洗与固定窗口分块。"""
    episode_size = max(1, int(episode_size))
    min_episode_turns = max(1, int(min_episode_turns))
    episodes: List[List[str]] = []

    def _flush_segment(segment: List[str]) -> None:
        cleaned = [s for s in (clean_qq_dialogue_line(x) for x in segment) if s]
        if not cleaned:
            return
        cur: List[str] = []
        for line in cleaned:
            cur.append(line)
            if len(cur) >= episode_size:
                episodes.append(list(cur))
                cur = []
        if len(cur) >= min_episode_turns:
            episodes.append(list(cur))

    segment: List[str] = []
    for raw in lines:
        if is_qq_dialogue_boundary_line(raw):
            _flush_segment(segment)
            segment = []
            continue
        segment.append(raw)
    _flush_segment(segment)
    return episodes


def load_dialogue_training_raw_text(
    path: str,
    *,
    episode_size: int = 4,
    min_episode_turns: int = 2,
) -> List[List[str]]:
    with Path(path).open(encoding="utf-8") as f:
        lines = [line.rstrip("\r\n") for line in f]
    return build_dialogue_episodes_from_lines(
        lines,
        episode_size=episode_size,
        min_episode_turns=min_episode_turns,
    )


def _clean_chunked_dialogue_text(text: str) -> str:
    s = str(text).strip()
    if not s:
        return ""
    if s.startswith("[图片:") or s.startswith("[表情"):
        return ""
    # [回复消息]@某人 你好 -> 你好
    s = re.sub(r"^\[回复消息\](?:@\S+\s*)?", "", s).strip()
    if not s:
        return ""
    # 纯表情/纯标点刷屏
    if not any(ch.isalnum() for ch in s):
        return ""
    return clean_qq_dialogue_line(s)


def _chunk_path_within_manifest(base_dir: Path, relative_path: str) -> Path | None:
    """解析 manifest 内 chunk 路径；拒绝逃出 base_dir 的 relativePath（含 ..）。"""
    rel = str(relative_path).strip()
    if not rel or Path(rel).is_absolute():
        return None
    base_resolved = base_dir.resolve()
    chunk_path = (base_dir / rel).resolve()
    try:
        inside = chunk_path.is_relative_to(base_resolved)
    except AttributeError:
        try:
            chunk_path.relative_to(base_resolved)
            inside = True
        except ValueError:
            inside = False
    if not inside or not chunk_path.is_file():
        return None
    return chunk_path


def load_dialogue_training_chunked_jsonl(
    path: str,
    *,
    episode_size: int = 8,
    min_episode_turns: int = 2,
    gap_seconds: int = 180,
) -> List[List[str]]:
    """
    加载结构化 QQ chunk_*.jsonl：
    - 保留 text/reply
    - 过滤 system / recalled / 图片占位 / 表情刷屏
    - 按时间间隔与最大窗口切 episode
    """
    episode_size = max(1, int(episode_size))
    min_episode_turns = max(1, int(min_episode_turns))
    gap_seconds = max(1, int(gap_seconds))
    episodes: List[List[str]] = []
    cur: List[str] = []
    last_ts: int | None = None

    def _flush() -> None:
        nonlocal cur
        if len(cur) >= min_episode_turns:
            episodes.append(list(cur))
        cur = []

    with Path(path).open(encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"第 {line_no} 行不是 JSON 对象")
            if bool(item.get("system")) or bool(item.get("recalled")):
                _flush()
                last_ts = None
                continue
            msg_type = str(item.get("type", ""))
            if msg_type not in {"text", "reply"}:
                continue
            content = item.get("content", {})
            if not isinstance(content, dict):
                continue
            text = _clean_chunked_dialogue_text(str(content.get("text", "")))
            if not text:
                continue
            ts = int(item.get("timestamp", 0))
            if last_ts is not None and ts > 0 and (ts - last_ts) > gap_seconds * 1000:
                _flush()
            cur.append(text)
            if len(cur) >= episode_size:
                _flush()
            last_ts = ts if ts > 0 else last_ts
    _flush()
    return episodes


def _load_chunked_jsonl_worker(args: tuple[str, int, int, int]) -> List[List[str]]:
    path, episode_size, min_episode_turns, gap_seconds = args
    return load_dialogue_training_chunked_jsonl(
        path,
        episode_size=episode_size,
        min_episode_turns=min_episode_turns,
        gap_seconds=gap_seconds,
    )


def _collect_chunked_root_jobs(
    root: str,
    *,
    episode_size: int,
    min_episode_turns: int,
    gap_seconds: int,
) -> List[tuple[str, int, int, int]]:
    root_path = Path(root)
    chunk_jobs: List[tuple[str, int, int, int]] = []
    for manifest in sorted(root_path.glob("**/manifest.json")):
        with manifest.open(encoding="utf-8") as f:
            meta = json.load(f)
        if not isinstance(meta, dict):
            continue
        metadata = meta.get("metadata", {})
        if isinstance(metadata, dict):
            if str(metadata.get("format", "")).strip() not in {"chunked-jsonl", ""}:
                continue
        chunked = meta.get("chunked", {})
        if not isinstance(chunked, dict):
            continue
        chunks = chunked.get("chunks", [])
        if not isinstance(chunks, list):
            continue
        base_dir = manifest.parent
        for item in chunks:
            if not isinstance(item, dict):
                continue
            rel = str(item.get("relativePath", "")).strip()
            chunk_path = _chunk_path_within_manifest(base_dir, rel)
            if chunk_path is None:
                continue
            chunk_jobs.append(
                (
                    str(chunk_path),
                    int(episode_size),
                    int(min_episode_turns),
                    int(gap_seconds),
                )
            )
    return chunk_jobs


def load_dialogue_training_chunked_root(
    root: str,
    *,
    episode_size: int = 8,
    min_episode_turns: int = 2,
    gap_seconds: int = 180,
    workers: int = 1,
    prefetch_chunks: int = 0,
) -> List[List[str]]:
    """
    递归扫描 exportsmessage 根目录下所有 chunked-jsonl 导出目录，
    按 manifest.json 中声明的 chunk 顺序汇总为训练 episodes。
    """
    chunk_jobs = _collect_chunked_root_jobs(
        root,
        episode_size=episode_size,
        min_episode_turns=min_episode_turns,
        gap_seconds=gap_seconds,
    )
    episodes: List[List[str]] = []
    for chunk_episodes in _iter_chunk_job_results(
        chunk_jobs,
        workers=workers,
        prefetch_chunks=prefetch_chunks,
    ):
        episodes.extend(chunk_episodes)
    return episodes


def _iter_chunk_job_results(
    chunk_jobs: List[tuple[str, int, int, int]],
    *,
    workers: int,
    prefetch_chunks: int,
):
    worker_count = max(1, int(workers))
    if worker_count <= 1 or len(chunk_jobs) <= 1:
        for job in chunk_jobs:
            yield _load_chunked_jsonl_worker(job)
        return

    prefetch_limit = max(worker_count, int(prefetch_chunks) if prefetch_chunks > 0 else worker_count * 2)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        job_iter = iter(chunk_jobs)
        pending = deque()

        def _fill_pending() -> None:
            while len(pending) < prefetch_limit:
                try:
                    job = next(job_iter)
                except StopIteration:
                    break
                pending.append(executor.submit(_load_chunked_jsonl_worker, job))

        _fill_pending()
        while pending:
            future = pending.popleft()
            yield future.result()
            _fill_pending()


def _preflight_chunk_jobs_episode_count(
    chunk_jobs: List[tuple[str, int, int, int]],
    *,
    workers: int,
    prefetch_chunks: int,
) -> int:
    """
    按与训练相同的切块逻辑顺序遍历全部 chunk，累计 episode 条数。
    仅在 chunked-root + slow-final 且无 max_episodes 时调用，用于单次启动内定位「全局末尾」。
    """
    total = 0
    for chunk_episodes in _iter_chunk_job_results(
        chunk_jobs,
        workers=workers,
        prefetch_chunks=prefetch_chunks,
    ):
        total += len(chunk_episodes)
    return total


def run_dialogue_training_chunked_root_streaming(
    agent: CognitiveDialogueAgent,
    root: str,
    *,
    episode_size: int = 8,
    min_episode_turns: int = 2,
    gap_seconds: int = 180,
    epochs: int = 1,
    between_turn_ticks: int = 0,
    post_episode_ticks: int = 0,
    slow_final_episodes: int = 0,
    between_turn_ticks_slow: int = 1,
    post_episode_ticks_slow: int = 3,
    shuffle: bool = False,
    seed: int = 42,
    progress_every: int = 0,
    live_progress: bool = False,
    device_label: str = "cpu",
    batch_tokenizer_updates: bool = False,
    workers: int = 1,
    prefetch_chunks: int = 0,
    max_episodes: int = 0,
    training_fast_path: bool = False,
    skip_internal_efe: bool = False,
    output_fn: Callable[..., None] = print,
) -> dict[str, int]:
    chunk_jobs = _collect_chunked_root_jobs(
        root,
        episode_size=episode_size,
        min_episode_turns=min_episode_turns,
        gap_seconds=gap_seconds,
    )
    rng = random.Random(seed)
    epochs = max(1, int(epochs))
    remaining_limit = max(0, int(max_episodes))
    merged = {
        "episodes": 0,
        "turns": 0,
        "internal_ticks": 0,
        "internal_emits": 0,
        "imprint_tokens": 0,
        "new_imprint_tokens": 0,
        "merged_imprints": 0,
        "active_tokens": 0,
        "dormant_tokens": 0,
        "associated_activated_tokens": 0,
        "cold_probe_tokens": 0,
        "associative_probes": 0,
        "elapsed_sec": 0.0,
        "device": device_label,
        "train_path": "train_fast" if training_fast_path else "full_turn",
        "internal_efe": "skip" if skip_internal_efe else "full",
    }
    start = time.perf_counter()
    eff_slow_final = max(0, int(slow_final_episodes))
    schedule_total_full: Optional[int] = None
    if remaining_limit > 0:
        schedule_total_full = remaining_limit
    elif eff_slow_final > 0:
        output_fn(
            "[对话预热] 末尾蓄水：正在预扫描 chunked 源以统计 episode 总数（单次进程内生效；"
            "等于先把全部 chunk 解析一遍以建树，随后再正式训练一遍）。",
            flush=True,
        )
        pre_t0 = time.perf_counter()
        per_epoch_eps = _preflight_chunk_jobs_episode_count(
            chunk_jobs,
            workers=workers,
            prefetch_chunks=prefetch_chunks,
        )
        schedule_total_full = max(0, per_epoch_eps) * epochs
        output_fn(
            f"[对话预热] 预扫描完成：每 epoch 约 {per_epoch_eps} 条 episode，"
            f"调度总轮数上限={schedule_total_full}（epochs={epochs}），"
            f"耗时 {time.perf_counter() - pre_t0:.1f}s。",
            flush=True,
        )
    for _ in range(epochs):
        current_jobs = list(chunk_jobs)
        if shuffle and current_jobs:
            rng.shuffle(current_jobs)
        for chunk_episodes in _iter_chunk_job_results(
            current_jobs,
            workers=workers,
            prefetch_chunks=prefetch_chunks,
        ):
            if remaining_limit > 0:
                left = remaining_limit - int(merged["episodes"])
                if left <= 0:
                    break
                chunk_episodes = chunk_episodes[:left]
            if not chunk_episodes:
                continue
            eff_slow = eff_slow_final
            global_total_opt: Optional[int] = None
            if eff_slow > 0:
                global_total_opt = schedule_total_full
            metrics = run_dialogue_training(
                agent,
                chunk_episodes,
                epochs=1,
                between_turn_ticks=between_turn_ticks,
                post_episode_ticks=post_episode_ticks,
                slow_final_episodes=eff_slow,
                between_turn_ticks_slow=between_turn_ticks_slow,
                post_episode_ticks_slow=post_episode_ticks_slow,
                episode_global_base=int(merged["episodes"]),
                episode_global_total=global_total_opt,
                shuffle=False,
                seed=seed,
                progress_every=progress_every,
                live_progress=live_progress,
                device_label=device_label,
                batch_tokenizer_updates=batch_tokenizer_updates,
                training_fast_path=training_fast_path,
                skip_internal_efe=skip_internal_efe,
                output_fn=output_fn,
            )
            for key in ("episodes", "turns", "internal_ticks", "internal_emits"):
                merged[key] = int(merged[key]) + int(metrics[key])
            merged["imprint_tokens"] = int(metrics["imprint_tokens"])
            merged["new_imprint_tokens"] += int(metrics["new_imprint_tokens"])
            merged["merged_imprints"] = int(merged["merged_imprints"]) + int(metrics["merged_imprints"])
            merged["active_tokens"] = int(metrics["active_tokens"])
            merged["dormant_tokens"] = int(metrics["dormant_tokens"])
            merged["associated_activated_tokens"] = int(merged["associated_activated_tokens"]) + int(metrics["associated_activated_tokens"])
            merged["cold_probe_tokens"] = int(merged["cold_probe_tokens"]) + int(metrics["cold_probe_tokens"])
            merged["associative_probes"] = int(merged["associative_probes"]) + int(metrics["associative_probes"])
            if remaining_limit > 0 and int(merged["episodes"]) >= remaining_limit:
                break
        if remaining_limit > 0 and int(merged["episodes"]) >= remaining_limit:
            break
    merged["elapsed_sec"] = max(0.0, time.perf_counter() - start)
    final_mem_stats = agent._word_memory.stats()  # noqa: SLF001
    merged["imprint_tokens"] = int(final_mem_stats["total_tokens"])
    merged["active_tokens"] = int(final_mem_stats["active_tokens"])
    merged["dormant_tokens"] = int(final_mem_stats["dormant_tokens"])
    merged["associated_activated_tokens"] = int(final_mem_stats["associated_activated_tokens"])
    merged["cold_probe_tokens"] = int(final_mem_stats["cold_probe_tokens"])
    return merged


def resolve_dialogue_train_device(requested: str) -> tuple[str, str]:
    req = str(requested or "auto").strip().lower() or "auto"
    if req == "auto":
        if torch is not None and torch.cuda.is_available():
            return "cuda", "PyTorch CUDA"
        if torch is not None:
            return "cpu", "PyTorch CPU"
        return "cpu", "纯 Python / 无 PyTorch"
    if req == "cuda":
        if torch is not None and torch.cuda.is_available():
            return "cuda", "PyTorch CUDA"
        if torch is not None:
            return "cpu", "请求 CUDA，但当前不可用，已回退 CPU"
        return "cpu", "请求 CUDA，但未安装 PyTorch，已回退 CPU"
    return "cpu", "PyTorch CPU" if torch is not None else "纯 Python / 无 PyTorch"


def _format_counter(counter: Counter[str], *, limit: int = 4) -> str:
    if not counter:
        return "—"
    parts = [f"{k}:{v}" for k, v in counter.most_common(limit)]
    return ", ".join(parts)


def _format_sigma(sigma: Dict[str, float]) -> str:
    if not sigma:
        return "—"
    return ", ".join(f"{k}={v:.2f}" for k, v in sigma.items())


def effective_dialogue_train_ticks(
    *,
    global_one_based_episode_index: int,
    schedule_total_episodes: int,
    slow_final_episodes: int,
    between_turn_ticks: int,
    post_episode_ticks: int,
    between_turn_ticks_slow: int,
    post_episode_ticks_slow: int,
) -> Tuple[int, int]:
    """
    按「全局第几条 episode（从 1 起）」决定在预热中使用的回合间 / episode 末内心 tick 次数。
    ``slow_final_episodes > 0`` 时，全局最后若干条 episode 使用 ``*_slow`` 配置（蓄水期）。
    """
    sf = max(0, int(slow_final_episodes))
    total = max(0, int(schedule_total_episodes))
    idx = int(global_one_based_episode_index)
    if sf <= 0 or total <= 0:
        return between_turn_ticks, post_episode_ticks
    start_slow_at = total - sf + 1
    if idx >= start_slow_at:
        return between_turn_ticks_slow, post_episode_ticks_slow
    return between_turn_ticks, post_episode_ticks


def run_dialogue_training(
    agent: CognitiveDialogueAgent,
    episodes: Optional[List[List[str]]] = None,
    *,
    episode_iter_factory: Optional[Callable[[], Iterable[List[str]]]] = None,
    epochs: int = 1,
    between_turn_ticks: int = 0,
    post_episode_ticks: int = 0,
    shuffle: bool = False,
    seed: int = 42,
    progress_every: int = 0,
    live_progress: bool = False,
    device_label: str = "cpu",
    batch_tokenizer_updates: bool = False,
    training_fast_path: bool = False,
    skip_internal_efe: bool = False,
    slow_final_episodes: int = 0,
    between_turn_ticks_slow: int = 1,
    post_episode_ticks_slow: int = 3,
    episode_global_base: int = 0,
    episode_global_total: Optional[int] = None,
    output_fn: Callable[..., None] = print,
) -> dict[str, int]:
    """
    离线对话预热：把 episode 中的 turns 当作用户输入批量跑 `agent.turn()`，
    并在回合间/episode 末尾穿插 `internal_tick()`，快速积累印记、sigma 与在线表面统计。

    - 列表模式：传入 ``episodes``；
    - **边读边训**：传入 ``episode_iter_factory``，每 epoch 调用一次工厂得到新的迭代器（大语料 JSONL）。
      流式模式下不支持 ``shuffle``；若未提供 ``episode_global_total`` 则关闭 ``slow_final_episodes``。
    """
    streaming = episode_iter_factory is not None
    if episode_iter_factory is None:
        if episodes is None:
            raise TypeError("run_dialogue_training 需要 episodes 或 episode_iter_factory")
    elif episodes is not None:
        raise TypeError("run_dialogue_training 不能同时传入 episodes 与 episode_iter_factory")
    if streaming and shuffle:
        raise ValueError("episode_iter_factory 流式训练不支持 shuffle=True")

    rng = random.Random(seed)
    epochs = max(1, int(epochs))
    between_turn_ticks = max(0, int(between_turn_ticks))
    post_episode_ticks = max(0, int(post_episode_ticks))
    between_turn_ticks_slow = max(0, int(between_turn_ticks_slow))
    post_episode_ticks_slow = max(0, int(post_episode_ticks_slow))
    slow_final_episodes = max(0, int(slow_final_episodes))
    episode_global_base = max(0, int(episode_global_base))
    progress_every = max(0, int(progress_every))

    if streaming:
        base_episodes: List[List[str]] = []
        total_episodes: Optional[int] = None
        total_turns: Optional[int] = None
        if slow_final_episodes > 0 and episode_global_total is None:
            output_fn(
                "[训练] 流式数据源未提供 episode_global_total，已忽略 slow_final_episodes。",
                flush=True,
            )
            slow_final_episodes = 0
        schedule_total_episodes = (
            max(1, int(episode_global_total))
            if episode_global_total is not None
            else 10**12
        )
    else:
        assert episodes is not None
        base_episodes = [list(ep) for ep in episodes if ep]
        total_episodes = len(base_episodes) * epochs
        total_turns = sum(len(ep) for ep in base_episodes) * epochs
        schedule_total_episodes = (
            max(1, int(episode_global_total))
            if episode_global_total is not None
            else total_episodes
        )
    metrics = {
        "episodes": 0,
        "turns": 0,
        "internal_ticks": 0,
        "internal_emits": 0,
        "imprint_tokens": 0,
        "new_imprint_tokens": 0,
        "merged_imprints": 0,
        "active_tokens": 0,
        "dormant_tokens": 0,
        "associated_activated_tokens": 0,
        "cold_probe_tokens": 0,
        "associative_probes": 0,
        "elapsed_sec": 0.0,
        "device": device_label,
        "train_path": "train_fast" if training_fast_path else "full_turn",
        "internal_efe": "skip" if skip_internal_efe else "full",
    }
    train_start = time.perf_counter()
    window_start = train_start
    initial_mem_stats = agent._word_memory.stats()  # noqa: SLF001
    initial_total_tokens = int(initial_mem_stats["total_tokens"])
    agent._word_memory.progress_new_vocab_snapshot()  # noqa: SLF001 — 清零进度窗口，避免沿用会话内计数
    window_chars = int(agent.tokenizer.total_chars)
    window_branches: Counter[str] = Counter()
    window_focus: Counter[str] = Counter()
    sigma_now = agent.sigma_state()
    resource_now = agent.tokenizer.resource_state()
    need_learning_summary = progress_every > 0
    learn_flag = bool(getattr(agent, "_learn_tokenizer_from_user", True))
    fast_train_batch_size = 32
    if batch_tokenizer_updates:
        setattr(agent, "_learn_tokenizer_from_user", False)

    try:
        turn_fn = agent.train_step if training_fast_path else agent.turn
        turn_batch_fn = agent.train_step_batch if training_fast_path else None
        internal_tick_fn = (
            agent.internal_tick_train_fast if skip_internal_efe else agent.internal_tick
        )
        internal_tick_many_fn = (
            agent.internal_tick_train_fast_many if skip_internal_efe else None
        )
        sigma_state_fn = agent.sigma_state
        ingest_lines = agent.tokenizer.ingest_interaction_lines
        for _ in range(epochs):
            if streaming:
                assert episode_iter_factory is not None
                episode_source: Iterable[List[str]] = episode_iter_factory()
            elif shuffle:
                shuffled = list(base_episodes)
                rng.shuffle(shuffled)
                episode_source = shuffled
            else:
                episode_source = base_episodes
            for turns in episode_source:
                metrics["episodes"] += 1
                global_one_based = episode_global_base + metrics["episodes"]
                bt_use, pt_use = effective_dialogue_train_ticks(
                    global_one_based_episode_index=global_one_based,
                    schedule_total_episodes=schedule_total_episodes,
                    slow_final_episodes=slow_final_episodes,
                    between_turn_ticks=between_turn_ticks,
                    post_episode_ticks=post_episode_ticks,
                    between_turn_ticks_slow=between_turn_ticks_slow,
                    post_episode_ticks_slow=post_episode_ticks_slow,
                )
                batch_ranges = (
                    range(0, len(turns), fast_train_batch_size)
                    if training_fast_path
                    else range(0, len(turns), 1)
                )
                for batch_start in batch_ranges:
                    batch_turns = turns[batch_start : batch_start + (fast_train_batch_size if training_fast_path else 1)]
                    if training_fast_path and turn_batch_fn is not None:
                        batch_results = turn_batch_fn(batch_turns)
                    else:
                        batch_results = [turn_fn(batch_turns[0])]
                    for rel_idx, turn in enumerate(batch_results):
                        idx = batch_start + rel_idx
                        metrics["turns"] += 1
                        if need_learning_summary:
                            branch = str(turn["output_branch"] if isinstance(turn, dict) else turn.output_branch)
                            window_branches[branch] += 1
                            focus_tok = str(turn["conflict_focus_token"] if isinstance(turn, dict) else turn.conflict_focus_token)
                            if focus_tok:
                                window_focus[focus_tok] += 1
                            sigma_now = sigma_state_fn()
                            resource_now = (
                                turn["resource_snapshot"] if isinstance(turn, dict) else turn.resource_snapshot
                            )
                        if idx + 1 < len(turns):
                            if internal_tick_many_fn is not None and bt_use > 0:
                                metrics["internal_ticks"] += bt_use
                                mono = internal_tick_many_fn(bt_use)
                                if mono is not None:
                                    metrics["internal_emits"] += 1
                                    if str(mono.output_branch) == "associative_probe":
                                        metrics["associative_probes"] += 1
                            else:
                                for _ in range(bt_use):
                                    metrics["internal_ticks"] += 1
                                    mono = internal_tick_fn()
                                    if mono is not None:
                                        metrics["internal_emits"] += 1
                                        if str(mono.output_branch) == "associative_probe":
                                            metrics["associative_probes"] += 1
                if internal_tick_many_fn is not None and pt_use > 0:
                    metrics["internal_ticks"] += pt_use
                    mono = internal_tick_many_fn(pt_use)
                    if mono is not None:
                        metrics["internal_emits"] += 1
                        if str(mono.output_branch) == "associative_probe":
                            metrics["associative_probes"] += 1
                else:
                    for _ in range(pt_use):
                        metrics["internal_ticks"] += 1
                        mono = internal_tick_fn()
                        if mono is not None:
                            metrics["internal_emits"] += 1
                            if str(mono.output_branch) == "associative_probe":
                                metrics["associative_probes"] += 1
                if batch_tokenizer_updates:
                    ingest_lines(turns)
                if live_progress:
                    elapsed = max(1e-6, time.perf_counter() - train_start)
                    ep_disp = (
                        f"{metrics['episodes']}/{total_episodes}"
                        if total_episodes is not None
                        else str(metrics["episodes"])
                    )
                    turn_disp = (
                        f"{metrics['turns']}/{total_turns}"
                        if total_turns is not None
                        else str(metrics["turns"])
                    )
                    output_fn(
                        f"\r[训练] {ep_disp} 轮, {turn_disp} 条, "
                        f"{metrics['turns'] / elapsed:.1f} 条/秒, 设备={device_label}",
                        end="",
                        flush=True,
                    )
                if need_learning_summary and (metrics["episodes"] % progress_every == 0):
                    new_vocab_n, new_vocab_samples = agent._word_memory.progress_new_vocab_snapshot()  # noqa: SLF001
                    mem_stats = agent._word_memory.stats()  # noqa: SLF001
                    total_tokens = int(mem_stats["total_tokens"])
                    now = time.perf_counter()
                    elapsed_total = max(1e-6, now - train_start)
                    elapsed_window = max(1e-6, now - window_start)
                    chars_delta = int(agent.tokenizer.total_chars) - window_chars
                    output_fn("")
                    ep_disp2 = (
                        f"{metrics['episodes']}/{total_episodes}"
                        if total_episodes is not None
                        else str(metrics["episodes"])
                    )
                    turn_disp2 = (
                        f"{metrics['turns']}/{total_turns}"
                        if total_turns is not None
                        else str(metrics["turns"])
                    )
                    output_fn(
                        f"[训练] 轮次={ep_disp2}, 条数={turn_disp2}, "
                        f"设备={device_label}, "
                        f"条/秒={metrics['turns'] / elapsed_total:.2f}, "
                        f"窗口条/秒={max(0, sum(window_branches.values())) / elapsed_window:.2f}"
                    )
                    output_fn(
                        f"[learn] 新增印记词={new_vocab_n}"
                        f"{'（' + '、'.join(new_vocab_samples) + '）' if new_vocab_samples else ''}, "
                        f"总印记词={total_tokens}, 分词字符+={chars_delta}, "
                        f"合并印记={int(mem_stats['merged_imprints']) - int(initial_mem_stats['merged_imprints'])}, "
                        f"活跃词={int(mem_stats['active_tokens'])}, "
                        f"休眠词={int(mem_stats['dormant_tokens'])}, "
                        f"关联激活词={int(mem_stats['associated_activated_tokens']) - int(initial_mem_stats['associated_activated_tokens'])}, "
                        f"冷词补扫数={int(mem_stats['cold_probe_tokens']) - int(initial_mem_stats['cold_probe_tokens'])}, "
                        f"关联探针={int(metrics['associative_probes'])}, "
                        f"分支={_format_counter(window_branches)}, "
                        f"焦点={_format_counter(window_focus)}, "
                        f"sigma={_format_sigma(sigma_now)}, "
                        f"R={float(resource_now.get('R', 0.0)):.3f}, "
                        f"R_max={float(resource_now.get('R_max', 0.0)):.3f}, "
                        f"m={float(resource_now.get('m', 0.0)):.3f}, "
                        f"F_ema={float(resource_now.get('F_ema', 0.0)):.3f}"
                    )
                    window_start = now
                    window_chars = int(agent.tokenizer.total_chars)
                    window_branches = Counter()
                    window_focus = Counter()
    finally:
        setattr(agent, "_learn_tokenizer_from_user", learn_flag)

    if live_progress:
        output_fn("")
    final_mem_stats = agent._word_memory.stats()  # noqa: SLF001
    metrics["imprint_tokens"] = int(final_mem_stats["total_tokens"])
    metrics["new_imprint_tokens"] = int(final_mem_stats["total_tokens"]) - initial_total_tokens
    metrics["merged_imprints"] = int(final_mem_stats["merged_imprints"]) - int(initial_mem_stats["merged_imprints"])
    metrics["active_tokens"] = int(final_mem_stats["active_tokens"])
    metrics["dormant_tokens"] = int(final_mem_stats["dormant_tokens"])
    metrics["associated_activated_tokens"] = int(final_mem_stats["associated_activated_tokens"]) - int(initial_mem_stats["associated_activated_tokens"])
    metrics["cold_probe_tokens"] = int(final_mem_stats["cold_probe_tokens"]) - int(initial_mem_stats["cold_probe_tokens"])
    metrics["elapsed_sec"] = max(0.0, time.perf_counter() - train_start)
    return metrics


def build_model(dt: float) -> UnifiedVariationalAttentionModel:
    return UnifiedVariationalAttentionModel(
        observation=[1.0, 0.2, -0.3],
        mu=[0.15, 0.05, -0.1],
        pi=[1.3, 1.1, 0.9],
        pi_min=0.5,
        c_max=2.4,
        alpha=1.1,
        beta=0.6,
        gamma_phi=0.8,
        lambda_c=1.2,
        sigma0=0.3,
        tau_pi=1.0,
        dt=dt,
        task_input=[0.2, 0.5, 0.3],
        delta_s=[0.7, 0.4, 0.6],
    )


def run_interactive_calculator(
    coder: ArithmeticPredictiveCoder,
    input_fn=input,
    output_fn=print,
) -> None:
    output_fn("进入交互式计算器模式（输入 exit 退出）")
    while True:
        expr = input_fn("请输入表达式（如 35+17）: ").strip()
        if expr.lower() in {"exit", "quit", "q"}:
            output_fn("已退出交互式计算器模式")
            return
        if not expr:
            continue
        try:
            result = coder.solve_expression(expr)
            output_fn(f"答案: {result.answer}（步数={result.steps}）")
        except ValueError as exc:
            output_fn(f"输入错误: {exc}")


def run_interactive_tokenizer(
    tokenizer: PrecisionTokenizer,
    input_fn=input,
    output_fn=print,
) -> None:
    output_fn("进入交互式分词模式（输入 exit 退出，输入 rest 触发主动恢复）")
    while True:
        text = input_fn("请输入待分词文本: ").strip()
        if text.lower() in {"exit", "quit", "q"}:
            output_fn("已退出交互式分词模式")
            return
        if text.lower() == "rest":
            tokenizer.idle(steps=60)
            state = tokenizer.resource_state()
            output_fn(f"已主动恢复：R={state['R']:.3f}, m={state['m']:.3f}")
            continue
        if not text:
            continue
        trace = tokenizer.trace_tokenize(text)
        output_fn(f"分词结果: {' | '.join(trace['tokens'])}")
        output_fn(f"边界索引: {trace['boundary_indices']}")
        state = trace["resource_state"]
        output_fn(f"当前资源状态: R={state['R']:.3f}, m={state['m']:.3f}")


def _print_internal_monologue_block(
    agent: CognitiveDialogueAgent,
    mono: InternalMonologue,
    output_fn: Callable[..., None],
    *,
    leading_newline: bool = True,
) -> None:
    assoc_note = ""
    if mono.output_branch == "associative_probe" and mono.association_pick:
        assoc_note = (
            f", 联想={mono.association_trigger!r}→{mono.association_pick!r}"
        )
    prefix = "\n[内心] 模型: " if leading_newline else "[内心] 模型: "
    output_fn(
        f"{prefix}{mono.reply}\n"
        f"（trigger={mono.trigger_token!r}, tension={mono.tension:.3f}, "
        f"EFE={mono.best_efe:.3f}, 输出={mono.output_branch}, "
        f"ε_imprint={mono.semantic_surprise_max:.3f}{assoc_note}）"
    )
    rs = agent.tokenizer.resource_state()
    output_fn(
        f"资源: R={rs.get('R', 0):.3f}, R_max={rs.get('R_max', 0):.3f}, "
        f"m={rs.get('m', 0):.3f}, F_ema={rs.get('F_ema', 0):.3f}"
    )


def _print_external_turn_block(
    agent: CognitiveDialogueAgent,
    turn: DialogueTurn,
    output_fn: Callable[..., None],
) -> None:
    output_fn(f"模型: {turn.reply}")
    assoc_note = ""
    if turn.output_branch == "associative_probe" and turn.association_pick:
        assoc_note = f"，联想={turn.association_trigger!r}→{turn.association_pick!r}"
    output_fn(
        f"（u_curiosity={turn.u_curiosity:.3f}, u_task={turn.u_task:.3f}, π_statement={turn.pi_statement:.3f}, "
        f"ε_social={turn.epsilon_social_in:.3f}, ε_secondary={turn.epsilon_secondary:.3f}, 重规划={turn.replan_count}, "
        f"输出={turn.output_branch}, ε_imprint={turn.semantic_surprise_max:.3f}, 焦点={turn.conflict_focus_token or '—'}"
        f"{assoc_note}）"
    )
    rs = turn.resource_snapshot
    output_fn(
        f"资源: R={rs.get('R', 0):.3f}, R_max={rs.get('R_max', 0):.3f}, "
        f"m={rs.get('m', 0):.3f}, F_ema={rs.get('F_ema', 0):.3f}"
    )


def _emit_user_line_with_internal_arbitration(
    agent: CognitiveDialogueAgent,
    line: str,
    output_fn: Callable[..., None],
    *,
    inner_sole_reply_when_leading: bool = False,
) -> Union[DialogueTurn, InternalMonologue]:
    """每轮先 internal_tick（若有），再 ``turn(line)``；默认两段均输出，顺序由 EFE 仲裁。

    若 ``inner_sole_reply_when_leading`` 且仲裁为「内心先」，则只对用户展示内心独白
    （仍执行完整 ``turn`` 以更新印记与资源，但不打印对外草稿）。
    """
    mono_pre = agent.internal_tick()
    turn = agent.turn(line)
    if mono_pre is not None:
        ticket = agent.social_arbitration_ticket(float(turn.epsilon_social_in))
        g_ext_net = agent.arbitration_external_net_cost(turn)
        g_int = float(mono_pre.best_efe)
        ext_first = agent.arbitration_external_first(mono_pre, turn)
        output_fn(
            f"[仲裁] 内心EFE(4a)={g_int:.4f}, 外部EFE(全)={float(turn.best_efe):.4f}, "
            f"社会入场券={ticket:.4f}, 外部净成本={g_ext_net:.4f} → "
            f"{'外部先' if ext_first else '内心先'}"
        )
        if inner_sole_reply_when_leading and (not ext_first):
            _print_internal_monologue_block(agent, mono_pre, output_fn, leading_newline=True)
            return mono_pre
        if ext_first:
            _print_external_turn_block(agent, turn, output_fn)
            _print_internal_monologue_block(agent, mono_pre, output_fn, leading_newline=True)
        else:
            _print_internal_monologue_block(agent, mono_pre, output_fn, leading_newline=True)
            _print_external_turn_block(agent, turn, output_fn)
    else:
        _print_external_turn_block(agent, turn, output_fn)
    return turn


# 空闲内心主循环的 queue 超时须为正值；过小会忙等 CPU。
_MIN_DIALOGUE_INTERNAL_IDLE_SEC = 0.25


def _dialogue_idle_queue_timeout(
    *,
    internal_idle_sec: float,
    inner_speak_not_before: float,
) -> tuple[float, bool]:
    """计算 ``queue.get(timeout=...)`` 用的等待秒数，以及超时后是否允许空闲输出 [内心]。

    ``inner_speak_not_before`` 为 ``time.monotonic()`` 门槛：任意模型输出后会推后，
    实现「回应后再静默若干秒才把空闲内心说出来」。
    """
    idle = max(_MIN_DIALOGUE_INTERNAL_IDLE_SEC, float(internal_idle_sec))
    now = time.monotonic()
    if now >= float(inner_speak_not_before):
        return idle, True
    remain = float(inner_speak_not_before) - now
    return max(_MIN_DIALOGUE_INTERNAL_IDLE_SEC, min(idle, remain)), False


def _run_interactive_dialogue_idle_internal(
    agent: CognitiveDialogueAgent,
    output_fn: Callable[..., None],
    *,
    internal_idle_sec: float,
    post_exchange_inner_idle_sec: float,
    inner_sole_reply_when_leading: bool,
    save_tokenizer_path: str,
    save_dialogue_path: str,
    dialogue_save_compact: bool,
) -> None:
    """
    stdin 在后台线程读取；主循环在 idle 超时内跑 internal_tick，不依赖你发言。
    任意模型输出（对外轮或空闲 [内心]）后，再经 ``post_exchange_inner_idle_sec`` 秒静默
    才允许下一轮空闲 [内心] 发声。分词器仅在主线程使用。
    """
    evt_q: queue.Queue[tuple[str, str]] = queue.Queue()
    stop = threading.Event()

    def _stdin_reader() -> None:
        while not stop.is_set():
            try:
                line = sys.stdin.readline()
            except Exception:
                evt_q.put(("eof", ""))
                return
            if line == "":
                evt_q.put(("eof", ""))
                return
            evt_q.put(("line", line.rstrip("\r\n")))

    threading.Thread(target=_stdin_reader, daemon=True).start()

    last_fb: Optional[Union[DialogueTurn, InternalMonologue]] = None
    need_prompt = True
    inner_speak_not_before = 0.0
    post_ex = max(0.0, float(post_exchange_inner_idle_sec))
    while True:
        if need_prompt:
            output_fn("你: ", end="", flush=True)
            need_prompt = False
        t_wait, may_idle_inner = _dialogue_idle_queue_timeout(
            internal_idle_sec=internal_idle_sec,
            inner_speak_not_before=inner_speak_not_before,
        )
        try:
            kind, data = evt_q.get(timeout=t_wait)
        except queue.Empty:
            if may_idle_inner:
                mono = agent.internal_tick()
                if mono:
                    _print_internal_monologue_block(agent, mono, output_fn, leading_newline=True)
                    last_fb = mono
                    inner_speak_not_before = time.monotonic() + post_ex
            need_prompt = True
            continue

        if kind == "eof":
            output_fn("\n已到达输入结束。")
            stop.set()
            if save_tokenizer_path:
                agent.tokenizer.save_model(save_tokenizer_path)
                output_fn(f"分词模型已写回磁盘: {save_tokenizer_path}")
            if save_dialogue_path:
                agent.save_dialogue_model(
                    save_dialogue_path, compact=dialogue_save_compact
                )
                output_fn(f"对话模型已写回磁盘: {save_dialogue_path}")
            return

        line = data.strip()
        output_fn("")
        if line.lower() in {"exit", "quit", "q"}:
            output_fn("已退出对话。")
            stop.set()
            if save_tokenizer_path:
                agent.tokenizer.save_model(save_tokenizer_path)
                output_fn(f"分词模型已写回磁盘: {save_tokenizer_path}")
            if save_dialogue_path:
                agent.save_dialogue_model(
                    save_dialogue_path, compact=dialogue_save_compact
                )
                output_fn(f"对话模型已写回磁盘: {save_dialogue_path}")
            return
        if not line:
            need_prompt = True
            continue
        if line.lower() == "rest":
            agent.tokenizer.idle(steps=60)
            st = agent.tokenizer.resource_state()
            output_fn(
                f"已 rest：R={st.get('R', 0):.3f}, m={st.get('m', 0):.3f}, R_max={st.get('R_max', 0):.3f}"
            )
            need_prompt = True
            continue

        rew = _dialogue_feedback_reward(line)
        if rew is not None:
            if last_fb is None:
                output_fn("（没有对上一轮模型输出可打分。）")
            elif isinstance(last_fb, DialogueTurn):
                agent.apply_dialogue_feedback(rew, last_fb)
                output_fn(f"已记录对外轮反馈 reward={rew:+.0f}，分支={last_fb.output_branch}")
            else:
                agent.apply_internal_monologue_feedback(rew, last_fb)
                output_fn(f"已记录内心独白反馈 reward={rew:+.0f}，分支={last_fb.output_branch}")
            last_fb = None
            need_prompt = True
            continue

        last_fb = None
        turn_fb = _emit_user_line_with_internal_arbitration(
            agent,
            line,
            output_fn,
            inner_sole_reply_when_leading=inner_sole_reply_when_leading,
        )
        last_fb = turn_fb
        inner_speak_not_before = time.monotonic() + post_ex
        need_prompt = True


def run_interactive_dialogue(
    agent: CognitiveDialogueAgent,
    input_fn=input,
    output_fn=print,
    *,
    save_tokenizer_path: str = "",
    save_dialogue_path: str = "",
    dialogue_save_compact: bool = False,
    internal_idle_sec: float = 2.0,
    post_exchange_inner_idle_sec: Optional[float] = None,
    inner_sole_reply_when_leading: bool = False,
) -> None:
    """交互对话；stdin 在后台线程读取，主线程按 internal_idle_sec 轮询空闲内心（不依赖你发言）。

    ``input_fn`` 保留以兼容旧调用方；实际输入始终来自 ``sys.stdin``（与 ``input()`` 同源）。

    ``post_exchange_inner_idle_sec``：模型任意输出后再静默若干秒，才允许空闲时输出 [内心]；
    默认与 ``internal_idle_sec`` 相同；``0`` 表示不额外冷却。
    """
    _ = input_fn
    output_fn(
        "认知对话模式（基于已训练分词器 + 好奇/元认知；输入 exit 退出，输入 rest 恢复分词器资源 R）"
    )
    idle = max(_MIN_DIALOGUE_INTERNAL_IDLE_SEC, float(internal_idle_sec))
    if idle != float(internal_idle_sec):
        output_fn(
            f"空闲内心间隔已钳位为 {idle:.2f}s（不得小于 {_MIN_DIALOGUE_INTERNAL_IDLE_SEC:.2f}s，避免忙等）。"
        )
    post_ex = (
        max(0.0, float(post_exchange_inner_idle_sec))
        if post_exchange_inner_idle_sec is not None
        else idle
    )
    output_fn(
        f"空闲轮询：约每 {idle:.1f}s 检查一次 stdin；无输入且已过「回应后冷却 {post_ex:.1f}s」时，"
        "才允许空闲时跑 internal_tick 并输出 [内心]（内心状态始终在算，此处只控制何时对用户说出来）。"
    )
    if inner_sole_reply_when_leading:
        output_fn(
            "已开启：若本轮有内心且仲裁为「内心先」，则只用内心独白作答（仍内部执行对外 turn 以更新状态）。"
        )
    output_fn(
        "提示：ε_imprint 要对「曾经出现过的词」比较才有意义；同一词多聊几句或 exit 保存后再开，印记会累积。"
        " 输出由预期自由能（EFE）在候选句之间择优；冲突探针仅当语义惊奇≥阈值时进入候选池（--dialogue-conflict-threshold）。"
        " 每轮在回应你之前会先跑一拍内心扫描：若产生内心独白则与对外回复一并输出，顺序由 [仲裁] 行说明；"
        "调整入场券强度可用 --dialogue-arbitration-ticket-scale。"
        " 方案 B：模型回复后、你下一轮发言前，可输入 good / bad / meh 对上一轮回复打分（偏好学习，写入对话 JSON）。"
    )
    _run_interactive_dialogue_idle_internal(
        agent,
        output_fn,
        internal_idle_sec=idle,
        post_exchange_inner_idle_sec=post_ex,
        inner_sole_reply_when_leading=inner_sole_reply_when_leading,
        save_tokenizer_path=save_tokenizer_path,
        save_dialogue_path=save_dialogue_path,
        dialogue_save_compact=dialogue_save_compact,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="统一变分注意力演示")
    parser.add_argument(
        "--mode",
        choices=["base", "arithmetic", "tokenize", "dialogue", "dialogue-train"],
        default="base",
        help="Demo mode",
    )
    parser.add_argument("--steps", type=int, default=20, help="Simulation steps")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step")
    parser.add_argument("--kappa", type=float, default=0.8, help="DDM drift scale")
    parser.add_argument("--learn-rounds", type=int, default=30, help="Arithmetic learning rounds")
    parser.add_argument("--save-model", type=str, default="", help="保存算术模型到JSON文件")
    parser.add_argument("--load-model", type=str, default="", help="从JSON文件加载算术模型")
    parser.add_argument("--expr", type=str, default="", help="像计算器一样求值，例如 35+17")
    parser.add_argument("--interactive", action="store_true", help="进入交互式计算器模式")
    parser.add_argument("--text", type=str, default="", help="分词模式输入文本")
    parser.add_argument(
        "--corpus",
        type=str,
        nargs="*",
        default=[],
        help="分词模式训练语料，可传多个句子",
    )
    parser.add_argument(
        "--tokenizer-model",
        type=str,
        default="tokenizer_zh_from_chunks_v2.json",
        help="对话模式：已训练分词器 JSON 路径（默认 tokenizer_zh_from_chunks_v2.json）",
    )
    parser.add_argument(
        "--dialogue-no-tokenizer-learn",
        action="store_true",
        help="对话模式：关闭「倾听后并入用户表面字串统计」（默认开启，用于在线微调转移概率）",
    )
    parser.add_argument(
        "--save-tokenizer",
        type=str,
        default="",
        help="对话模式：结束后保存分词器的路径；省略则写回 --tokenizer-model（可用 --dialogue-no-save 关闭）",
    )
    parser.add_argument(
        "--dialogue-no-save",
        action="store_true",
        help="对话模式：结束后不写分词模型与对话模型（默认两者都写）",
    )
    parser.add_argument(
        "--dialogue-no-tokenizer-save",
        action="store_true",
        help="对话 / dialogue-train：仍保存对话 JSON（除非另有 --dialogue-no-save），但不写回分词器；"
        "用于冻结 QQ 训好的词表，只更新印记与方案 B 的 preference_state。",
    )
    parser.add_argument(
        "--dialogue-save-compact",
        action="store_true",
        help="写入 *.dialogue.json 时使用单行紧凑 JSON（无缩进），显著减小体积、加快读写；加载兼容。",
    )
    parser.add_argument(
        "--dialogue-model",
        type=str,
        default="",
        help="对话模式：加载对话状态 JSON；省略时若存在与 --tokenizer-model 同名的 *.dialogue.json 则自动加载",
    )
    parser.add_argument(
        "--dialogue-fresh",
        action="store_true",
        help="对话模式：不自动加载、也不使用 --dialogue-model，槽位 σ 从默认初值开始",
    )
    parser.add_argument(
        "--save-dialogue",
        type=str,
        default="",
        help="对话模式：保存对话状态的路径；省略则为与分词写入路径同主文件名、扩展名 .dialogue.json",
    )
    parser.add_argument(
        "--dialogue-conflict-threshold",
        type=float,
        default=0.10,
        help="语义惊奇≥此值的冲突探针稿进入 EFE 候选池（不强制选中；L2 归一化尺度）",
    )
    parser.add_argument(
        "--dialogue-conflict-min-uc",
        type=float,
        default=0.42,
        help="保留兼容；当前 EFE 版本不再用作硬门槛",
    )
    parser.add_argument(
        "--dialogue-arbitration-ticket-scale",
        type=float,
        default=1.0,
        help="用户发言且本轮同时有内心独白时：社会入场券 ticket=scale*EFE_W_SOCIAL*ε_social；"
        "越大外部净成本扣得越多、内心相对越易先说。",
    )
    parser.add_argument(
        "--dialogue-internal-idle-sec",
        type=float,
        default=2.0,
        help="交互对话：空闲超过该秒数未输入则主线程自动跑一拍 internal_tick（过阈可输出 [内心]，不依赖你发言）。"
        f"小于 {_MIN_DIALOGUE_INTERNAL_IDLE_SEC:.2f} 的值会钳位到该下限（避免忙等）。",
    )
    parser.add_argument(
        "--dialogue-post-reply-inner-idle-sec",
        type=float,
        default=-1.0,
        help="模型任意输出（对外轮或空闲[内心]）后，再经过多少秒无输入才允许输出下一条空闲[内心]。"
        f"-1 表示与 --dialogue-internal-idle-sec（钳位后）相同；0 表示不额外冷却。",
    )
    parser.add_argument(
        "--dialogue-inner-sole-reply-when-leading",
        action="store_true",
        help="用户一轮中若先有内心独白且仲裁为「内心先」，则只对用户展示内心（仍执行 turn 更新状态，不打印对外草稿）。",
    )
    parser.add_argument(
        "--dialogue-internal-global-k",
        type=int,
        default=8,
        help="内心扫描：每次 internal_tick 从全库印记中额外取 top-K 高失配词。",
    )
    parser.add_argument(
        "--dialogue-internal-jitter",
        type=float,
        default=0.015,
        help="内心扫描：概念自发波动幅度；只影响本次张力评分，不写回记忆。",
    )
    parser.add_argument(
        "--dialogue-internal-seed",
        type=int,
        default=None,
        help="内心扫描随机种子；省略则每次进程使用独立随机序列。",
    )
    parser.add_argument(
        "--dialogue-train-file",
        type=str,
        default="",
        help="离线对话预热：turns 格式为每行 {\"turns\":[...]}；"
        "corpus_jsonl 为规范 UTF-8 JSONL（text + 可选 meta），见 --dialogue-train-format。",
    )
    parser.add_argument(
        "--dialogue-train-format",
        type=str,
        choices=("turns", "corpus_jsonl", "extract", "haid"),
        default="turns",
        help="turns：显式多轮列表；corpus_jsonl：参照数据治理文档的正文+元数据 JSONL，"
        "训练侧仅流式读入与窗口切块（非完整离线流水线）。extract/haid 为旧别名。",
    )
    parser.add_argument(
        "--dialogue-train-corpus-extra",
        type=str,
        nargs="*",
        default=[],
        help="corpus_jsonl 下额外 JSONL 路径（顺序拼接）；支持 .jsonl.gz。",
    )
    parser.add_argument(
        "--dialogue-train-haid-extra",
        type=str,
        nargs="*",
        default=[],
        help="已弃用：请用 --dialogue-train-corpus-extra。",
    )
    parser.add_argument(
        "--dialogue-train-corpus-skip-bad-lines",
        action="store_true",
        help="语料 JSONL 遇损坏行跳过（默认解析失败即报错）。",
    )
    parser.add_argument(
        "--dialogue-train-haid-skip-bad-lines",
        action="store_true",
        help="已弃用：请用 --dialogue-train-corpus-skip-bad-lines。",
    )
    parser.add_argument(
        "--dialogue-train-raw-file",
        type=str,
        default="",
        help="原始 QQ 对话逐行文本；会先做最小清洗，再按固定窗口切成 episode。",
    )
    parser.add_argument(
        "--dialogue-train-chunked-jsonl",
        type=str,
        default="",
        help="结构化 QQ chunk_*.jsonl 文件；会按 type/system/recalled/time 做专用切分。",
    )
    parser.add_argument(
        "--dialogue-train-chunked-root",
        type=str,
        default="",
        help="结构化 chunked-jsonl 导出根目录；会递归读取 manifest.json 并汇总所有 chunks。",
    )
    parser.add_argument(
        "--dialogue-train-episode-size",
        type=int,
        default=4,
        help="原始 QQ 文本切 episode 的固定窗口大小。",
    )
    parser.add_argument(
        "--dialogue-train-min-turns",
        type=int,
        default=2,
        help="原始 QQ 文本切 episode 时，尾部保留所需的最少 turn 数。",
    )
    parser.add_argument(
        "--dialogue-train-epochs",
        type=int,
        default=3,
        help="离线对话预热轮数。",
    )
    parser.add_argument(
        "--dialogue-train-between-turn-ticks",
        type=int,
        default=0,
        help="离线对话预热：相邻用户回合之间插入多少次训练快路径内心 tick（默认 0 以降低后期吞吐断崖）；设为 1 更接近旧行为。",
    )
    parser.add_argument(
        "--dialogue-train-post-episode-ticks",
        type=int,
        default=1,
        help="离线对话预热：每个 episode 结束后插入多少次训练快路径内心 tick（默认 1；原为 3 会在长预热下显著增耗）。",
    )
    parser.add_argument(
        "--dialogue-train-fast-internal-surprise-hist-cap",
        type=int,
        default=24,
        help="训练快路径扫描内心候选时，对每条印记历史只比较末尾 K 条（缩短大容量下的惊奇计算）；0 表示不截断（与完整 internal_tick 一致，较慢）。",
    )
    parser.add_argument(
        "--dialogue-train-slow-final-episodes",
        type=int,
        default=0,
        help="全局最后几条 episode 改用「慢速」内心 tick（见 *_slow 参数）；进度「…轮」即本条数。"
        "单次启动内生效：非 chunked-root 直接用总轮数；chunked-root 未设 max-episodes 时会先预扫描统计总数（多一轮解析开销）。0 关闭。",
    )
    parser.add_argument(
        "--dialogue-train-slow-between-turn-ticks",
        type=int,
        default=1,
        help="slow-final 阶段：相邻用户回合之间的训练快路径内心 tick 次数（对齐旧版常用 1）。",
    )
    parser.add_argument(
        "--dialogue-train-slow-post-episode-ticks",
        type=int,
        default=3,
        help="slow-final 阶段：每个 episode 结束后的训练快路径内心 tick 次数（对齐旧版常用 3）。",
    )
    parser.add_argument(
        "--dialogue-train-shuffle",
        action="store_true",
        help="离线对话预热：每轮打乱 episode 顺序。",
    )
    parser.add_argument(
        "--dialogue-train-max-episodes",
        type=int,
        default=0,
        help="离线对话预热：限制载入的 episode 数量；0 表示全部。",
    )
    parser.add_argument(
        "--dialogue-train-seed",
        type=int,
        default=42,
        help="离线对话预热随机种子（shuffle 等）。",
    )
    parser.add_argument(
        "--dialogue-train-gap-seconds",
        type=int,
        default=180,
        help="结构化 chunked JSONL 切分时，超过该时间间隔即断开 episode。",
    )
    parser.add_argument(
        "--dialogue-train-device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="训练期词印记扫描设备；auto 优先尝试 CUDA，否则回退 CPU。",
    )
    parser.add_argument(
        "--dialogue-train-progress-every",
        type=int,
        default=20,
        help="每 N 个 episode 打印一次学习摘要；0 关闭。",
    )
    parser.add_argument(
        "--dialogue-train-live-progress",
        action="store_true",
        help="训练时输出单行实时进度与吞吐速度。",
    )
    parser.add_argument(
        "--dialogue-train-workers",
        type=int,
        default=0,
        help="目录级 chunked 训练时的并行加载线程数；0 表示自动。",
    )
    parser.add_argument(
        "--dialogue-train-prefetch-chunks",
        type=int,
        default=0,
        help="目录级 chunked 训练时预取的 chunk 数；0 表示按 2x workers 自动。",
    )
    parser.add_argument(
        "--dialogue-train-no-batch-tokenizer-updates",
        action="store_true",
        help="关闭 episode 级批量 tokenizer 统计更新，回退为逐条在线更新。",
    )
    parser.add_argument(
        "--dialogue-train-full-turn",
        action="store_true",
        help="训练时走完整 `agent.turn()` 路径；默认使用更快的 `train_step()` 轻量路径。",
    )
    parser.add_argument(
        "--tokenize-no-save",
        action="store_true",
        help="分词交互模式：退出时不自动写回（默认从 --load-model 加载时会写回该路径，或写 --save-model）",
    )
    args = parser.parse_args()

    if args.mode == "base":
        model = build_model(dt=args.dt)
        print("t\tF\t\tpi\t\t\tphi\t\t\tdelta\tgamma\talpha0")
        for t in range(args.steps):
            free_energy = model.free_energy()
            drift = model.ddm_drift(kappa=args.kappa)
            gamma = model.gamma_power_proxy()
            alpha0 = model.alpha_power_proxy(0)
            print(
                f"{t:02d}\t{free_energy:+.6f}\t{fmt(model.pi)}\t{fmt(model.phi)}\t"
                f"{drift:.6f}\t{gamma:.6f}\t{alpha0:.6f}"
            )
            model.step()
        return

    if args.mode == "tokenize":
        if args.load_model:
            tokenizer = PrecisionTokenizer.load_model(args.load_model)
        else:
            tokenizer = PrecisionTokenizer()
            corpus = args.corpus or [
                "注意力 模型 学习 分词",
                "预测 编码 引导 注意力",
                "分词 不是 切割 而是 精度 起伏",
            ]
            tokenizer.fit(corpus)

        if args.interactive:
            run_interactive_tokenizer(tokenizer)
            tok_dest = args.save_model or (
                "" if args.tokenize_no_save else (args.load_model or "")
            )
            if tok_dest:
                tokenizer.save_model(tok_dest)
                print(f"分词模型已保存到: {tok_dest}")
            elif args.load_model and not args.save_model:
                print("提示: 交互会改变 R/m 等状态，退出时未保存；可加 --save-model 路径或去掉 --tokenize-no-save 以写回 --load-model")
            return

        text = args.text or "注意力模型学习分词"
        trace = tokenizer.trace_tokenize(text)
        print("分词模式（惊奇度驱动的精度脉冲）")
        print(f"输入文本: {text}")
        print(f"分词结果: {' | '.join(trace['tokens'])}")
        print(f"边界索引: {trace['boundary_indices']}")
        state = trace["resource_state"]
        print(f"当前资源状态: R={state['R']:.3f}, m={state['m']:.3f}")
        if args.save_model:
            tokenizer.save_model(args.save_model)
            print(f"分词模型已保存到: {args.save_model}")
        return

    if args.mode in {"dialogue", "dialogue-train"}:
        dialogue_load = resolve_dialogue_model_load_path(
            args.tokenizer_model,
            args.dialogue_model,
            dialogue_fresh=args.dialogue_fresh,
        )
        hist_cap_arg = args.dialogue_train_fast_internal_surprise_hist_cap
        agent = CognitiveDialogueAgent.from_tokenizer_path(
            args.tokenizer_model,
            learn_tokenizer_from_user=not args.dialogue_no_tokenizer_learn,
            dialogue_model_path=dialogue_load,
            conflict_surprise_threshold=args.dialogue_conflict_threshold,
            conflict_min_uc=args.dialogue_conflict_min_uc,
            internal_global_scan_k=args.dialogue_internal_global_k,
            internal_spontaneous_jitter=args.dialogue_internal_jitter,
            internal_rng_seed=args.dialogue_internal_seed,
            train_fast_internal_surprise_hist_cap=(
                None if hist_cap_arg <= 0 else hist_cap_arg
            ),
        )
        agent.ARBITRATION_SOCIAL_TICKET_SCALE = max(
            0.0, float(args.dialogue_arbitration_ticket_scale)
        )
        if dialogue_load:
            print(f"已加载对话状态: {dialogue_load}")
        tok_save = ""
        dialogue_state_save = ""
        if not args.dialogue_no_save:
            tok_target = args.save_tokenizer or args.tokenizer_model
            dialogue_state_save = args.save_dialogue or default_dialogue_model_path(tok_target)
            tok_save = "" if args.dialogue_no_tokenizer_save else tok_target
        if args.mode == "dialogue-train":
            resolved_device, device_note = resolve_dialogue_train_device(
                args.dialogue_train_device
            )
            agent.set_compute_device(resolved_device)
            print(f"训练设备: {resolved_device}")
            if device_note:
                print(f"设备说明: {device_note}")
            worker_count = args.dialogue_train_workers
            if worker_count <= 0:
                worker_count = max(1, min(8, os.cpu_count() or 1))
            if args.dialogue_train_chunked_root:
                metrics = run_dialogue_training_chunked_root_streaming(
                    agent,
                    args.dialogue_train_chunked_root,
                    episode_size=args.dialogue_train_episode_size,
                    min_episode_turns=args.dialogue_train_min_turns,
                    gap_seconds=args.dialogue_train_gap_seconds,
                    epochs=args.dialogue_train_epochs,
                    between_turn_ticks=args.dialogue_train_between_turn_ticks,
                    post_episode_ticks=args.dialogue_train_post_episode_ticks,
                    slow_final_episodes=args.dialogue_train_slow_final_episodes,
                    between_turn_ticks_slow=args.dialogue_train_slow_between_turn_ticks,
                    post_episode_ticks_slow=args.dialogue_train_slow_post_episode_ticks,
                    shuffle=args.dialogue_train_shuffle,
                    seed=args.dialogue_train_seed,
                    progress_every=args.dialogue_train_progress_every,
                    live_progress=args.dialogue_train_live_progress,
                    device_label=resolved_device,
                    batch_tokenizer_updates=not args.dialogue_train_no_batch_tokenizer_updates,
                    workers=worker_count,
                    prefetch_chunks=args.dialogue_train_prefetch_chunks,
                    max_episodes=args.dialogue_train_max_episodes,
                    training_fast_path=not args.dialogue_train_full_turn,
                    skip_internal_efe=not args.dialogue_train_full_turn,
                )
            else:
                if args.dialogue_train_chunked_jsonl:
                    episodes = load_dialogue_training_chunked_jsonl(
                        args.dialogue_train_chunked_jsonl,
                        episode_size=args.dialogue_train_episode_size,
                        min_episode_turns=args.dialogue_train_min_turns,
                        gap_seconds=args.dialogue_train_gap_seconds,
                    )
                    if args.dialogue_train_max_episodes > 0:
                        episodes = episodes[: args.dialogue_train_max_episodes]
                    metrics = run_dialogue_training(
                        agent,
                        episodes,
                        epochs=args.dialogue_train_epochs,
                        between_turn_ticks=args.dialogue_train_between_turn_ticks,
                        post_episode_ticks=args.dialogue_train_post_episode_ticks,
                        slow_final_episodes=args.dialogue_train_slow_final_episodes,
                        between_turn_ticks_slow=args.dialogue_train_slow_between_turn_ticks,
                        post_episode_ticks_slow=args.dialogue_train_slow_post_episode_ticks,
                        shuffle=args.dialogue_train_shuffle,
                        seed=args.dialogue_train_seed,
                        progress_every=args.dialogue_train_progress_every,
                        live_progress=args.dialogue_train_live_progress,
                        device_label=resolved_device,
                        batch_tokenizer_updates=not args.dialogue_train_no_batch_tokenizer_updates,
                        training_fast_path=not args.dialogue_train_full_turn,
                        skip_internal_efe=not args.dialogue_train_full_turn,
                    )
                elif args.dialogue_train_raw_file:
                    episodes = load_dialogue_training_raw_text(
                        args.dialogue_train_raw_file,
                        episode_size=args.dialogue_train_episode_size,
                        min_episode_turns=args.dialogue_train_min_turns,
                    )
                    if args.dialogue_train_max_episodes > 0:
                        episodes = episodes[: args.dialogue_train_max_episodes]
                    metrics = run_dialogue_training(
                        agent,
                        episodes,
                        epochs=args.dialogue_train_epochs,
                        between_turn_ticks=args.dialogue_train_between_turn_ticks,
                        post_episode_ticks=args.dialogue_train_post_episode_ticks,
                        slow_final_episodes=args.dialogue_train_slow_final_episodes,
                        between_turn_ticks_slow=args.dialogue_train_slow_between_turn_ticks,
                        post_episode_ticks_slow=args.dialogue_train_slow_post_episode_ticks,
                        shuffle=args.dialogue_train_shuffle,
                        seed=args.dialogue_train_seed,
                        progress_every=args.dialogue_train_progress_every,
                        live_progress=args.dialogue_train_live_progress,
                        device_label=resolved_device,
                        batch_tokenizer_updates=not args.dialogue_train_no_batch_tokenizer_updates,
                        training_fast_path=not args.dialogue_train_full_turn,
                        skip_internal_efe=not args.dialogue_train_full_turn,
                    )
                elif args.dialogue_train_format in ("corpus_jsonl", "extract", "haid"):
                    if args.dialogue_train_shuffle:
                        raise ValueError(
                            "dialogue-train-format=corpus_jsonl 使用边读边训，不支持 --dialogue-train-shuffle"
                        )
                    corpus_paths: List[str] = []
                    if args.dialogue_train_file.strip():
                        corpus_paths.append(args.dialogue_train_file.strip())
                    corpus_paths.extend(
                        str(x).strip()
                        for x in args.dialogue_train_corpus_extra
                        if x is not None and str(x).strip()
                    )
                    corpus_paths.extend(
                        str(x).strip()
                        for x in args.dialogue_train_haid_extra
                        if x is not None and str(x).strip()
                    )
                    if not corpus_paths:
                        raise ValueError(
                            "dialogue-train-format=corpus_jsonl 需要 --dialogue-train-file "
                            "或至少一条 --dialogue-train-corpus-extra（旧参数 --dialogue-train-haid-extra 仍可用）"
                        )
                    lim = max(0, int(args.dialogue_train_max_episodes))
                    ep_sz = max(1, int(args.dialogue_train_episode_size))
                    min_t = max(1, int(args.dialogue_train_min_turns))
                    skip_bad = bool(args.dialogue_train_corpus_skip_bad_lines) or bool(
                        args.dialogue_train_haid_skip_bad_lines
                    )

                    def _corpus_episode_factory() -> Iterable[List[str]]:
                        base = iter_dialogue_training_corpus_episodes(
                            corpus_paths,
                            episode_size=ep_sz,
                            min_episode_turns=min_t,
                            skip_bad_lines=skip_bad,
                        )
                        if lim <= 0:
                            return base
                        return _iter_take(base, lim)

                    metrics = run_dialogue_training(
                        agent,
                        episode_iter_factory=_corpus_episode_factory,
                        epochs=args.dialogue_train_epochs,
                        between_turn_ticks=args.dialogue_train_between_turn_ticks,
                        post_episode_ticks=args.dialogue_train_post_episode_ticks,
                        slow_final_episodes=args.dialogue_train_slow_final_episodes,
                        between_turn_ticks_slow=args.dialogue_train_slow_between_turn_ticks,
                        post_episode_ticks_slow=args.dialogue_train_slow_post_episode_ticks,
                        shuffle=False,
                        seed=args.dialogue_train_seed,
                        progress_every=args.dialogue_train_progress_every,
                        live_progress=args.dialogue_train_live_progress,
                        device_label=resolved_device,
                        batch_tokenizer_updates=not args.dialogue_train_no_batch_tokenizer_updates,
                        training_fast_path=not args.dialogue_train_full_turn,
                        skip_internal_efe=not args.dialogue_train_full_turn,
                    )
                else:
                    if not args.dialogue_train_file:
                        raise ValueError(
                            "dialogue-train 模式需要 --dialogue-train-file、--dialogue-train-raw-file、--dialogue-train-chunked-jsonl 或 --dialogue-train-chunked-root"
                        )
                    episodes = load_dialogue_training_jsonl(args.dialogue_train_file)
                    if args.dialogue_train_max_episodes > 0:
                        episodes = episodes[: args.dialogue_train_max_episodes]
                    metrics = run_dialogue_training(
                        agent,
                        episodes,
                        epochs=args.dialogue_train_epochs,
                        between_turn_ticks=args.dialogue_train_between_turn_ticks,
                        post_episode_ticks=args.dialogue_train_post_episode_ticks,
                        slow_final_episodes=args.dialogue_train_slow_final_episodes,
                        between_turn_ticks_slow=args.dialogue_train_slow_between_turn_ticks,
                        post_episode_ticks_slow=args.dialogue_train_slow_post_episode_ticks,
                        shuffle=args.dialogue_train_shuffle,
                        seed=args.dialogue_train_seed,
                        progress_every=args.dialogue_train_progress_every,
                        live_progress=args.dialogue_train_live_progress,
                        device_label=resolved_device,
                        batch_tokenizer_updates=not args.dialogue_train_no_batch_tokenizer_updates,
                        training_fast_path=not args.dialogue_train_full_turn,
                        skip_internal_efe=not args.dialogue_train_full_turn,
                    )
            print("对话预热完成")
            print(
                f"轮次={metrics['episodes']}, 条数={metrics['turns']}, "
                f"内部tick={metrics['internal_ticks']}, 内部发声={metrics['internal_emits']}, "
                f"印记词={metrics['imprint_tokens']}, 新增印记词={metrics['new_imprint_tokens']}, "
                f"合并印记={metrics['merged_imprints']}, "
                f"活跃词={metrics['active_tokens']}, 休眠词={metrics['dormant_tokens']}, "
                f"关联激活词={metrics['associated_activated_tokens']}, 冷词补扫数={metrics['cold_probe_tokens']}, "
                f"关联探针={metrics['associative_probes']}, "
                f"耗时={metrics['elapsed_sec']:.2f}s, 设备={metrics['device']}, "
                f"训练路径={metrics['train_path']}, 内部EFE={metrics['internal_efe']}"
            )
            if tok_save:
                agent.tokenizer.save_model(tok_save)
                print(f"分词模型已写回磁盘: {tok_save}")
            if dialogue_state_save:
                agent.save_dialogue_model(
                    dialogue_state_save, compact=args.dialogue_save_compact
                )
                print(f"对话模型已写回磁盘: {dialogue_state_save}")
            return
        if args.text:
            turn = agent.turn(args.text)
            print(f"你: {args.text}")
            print(f"模型: {turn.reply}")
            print(
                f"u_curiosity={turn.u_curiosity:.3f}, u_task={turn.u_task:.3f}, π_statement={turn.pi_statement:.3f}, "
                f"ε_social={turn.epsilon_social_in:.3f}, ε_secondary={turn.epsilon_secondary:.3f}, 重规划={turn.replan_count}, "
                f"输出={turn.output_branch}, ε_imprint={turn.semantic_surprise_max:.3f}, 焦点={turn.conflict_focus_token or '—'}"
            )
            print(f"资源: {turn.resource_snapshot}")
            if tok_save:
                agent.tokenizer.save_model(tok_save)
                print(f"分词模型已写回磁盘: {tok_save}")
            if dialogue_state_save:
                agent.save_dialogue_model(
                    dialogue_state_save, compact=args.dialogue_save_compact
                )
                print(f"对话模型已写回磁盘: {dialogue_state_save}")
            return
        if args.interactive:
            post_arg = float(args.dialogue_post_reply_inner_idle_sec)
            post_kw: Optional[float] = None if post_arg < 0 else max(0.0, post_arg)
            run_interactive_dialogue(
                agent,
                save_tokenizer_path=tok_save,
                save_dialogue_path=dialogue_state_save,
                dialogue_save_compact=args.dialogue_save_compact,
                internal_idle_sec=float(args.dialogue_internal_idle_sec),
                post_exchange_inner_idle_sec=post_kw,
                inner_sole_reply_when_leading=bool(
                    args.dialogue_inner_sole_reply_when_leading
                ),
            )
            return
        print("对话模式请使用 --text 单轮，或 --interactive 多轮。")
        return

    arithmetic_dt = max(args.dt, 0.05)
    if args.load_model:
        coder = ArithmeticPredictiveCoder.load_model(args.load_model, seed=42)
        arithmetic_dt = coder.dt
    else:
        coder = ArithmeticPredictiveCoder(seed=42, dt=arithmetic_dt)

    if args.expr:
        result = coder.solve_expression(args.expr)
        print(f"计算表达式: {args.expr}")
        print(f"模型答案: {result.answer}")
        print(f"收敛步数: {result.steps}")
        if args.save_model:
            coder.save_model(args.save_model)
            print(f"模型已保存到: {args.save_model}")
        return

    if args.interactive:
        run_interactive_calculator(coder)
        if args.save_model:
            coder.save_model(args.save_model)
            print(f"模型已保存到: {args.save_model}")
        return
    curriculum = CurriculumA(seed=42)
    eval_set = curriculum.sample(count=40, ops=["+", "-", "*", "/"])
    train_set = curriculum.sample(count=24, ops=["+", "-", "*", "/"])

    before = Evaluator(coder).evaluate(eval_set)
    for _ in range(args.learn_rounds):
        for problem in train_set:
            coder.learn(problem)
    after = Evaluator(coder).evaluate(eval_set)

    print("算术模式（无 loss 的预测编码 + 局部 Hebbian）")
    print(f"时间步长 dt={arithmetic_dt:.3f}")
    print(f"训练前: 准确率={before['accuracy']:.3f}, 平均步数={before['mean_steps']:.3f}, 平均绝对误差={before['mean_abs_error']:.3f}")
    print(f"训练后: 准确率={after['accuracy']:.3f}, 平均步数={after['mean_steps']:.3f}, 平均绝对误差={after['mean_abs_error']:.3f}")
    print("样例轨迹:")
    probes = [
        ArithmeticProblem(op="+", a=8, b=7),
        ArithmeticProblem(op="-", a=11, b=7),
        ArithmeticProblem(op="*", a=6, b=7),
        ArithmeticProblem(op="/", a=8, b=2),
    ]
    for pb in probes:
        result = coder.solve(pb)
        print(f"  {pb.a}{pb.op}{pb.b} -> {result.answer}（目标={pb.target}, 步数={result.steps}）")

    if args.save_model:
        coder.save_model(args.save_model)
        print(f"模型已保存到: {args.save_model}")


if __name__ == "__main__":
    main()
