"""
词状态印记库：分词器外的独立结构。每个 surface token 维护固定容量的历史「内心快照」，
用于意义检索与语义惊奇（印记与当前状态失配）。
"""

from __future__ import annotations

import json
import math
import random
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, cast

try:
    import torch
except Exception:  # pragma: no cover - 可选依赖
    torch = None

WORD_STATE_MEMORY_FORMAT = "word_state_memory_v1"

# 关联词 / 一跳前沿缓存容量上限（长跑训练可避免数十万条目拖垮 GC 与 dict 操作）
_DEFAULT_STRUCTURAL_CACHE_MAX = 8192

# 用于 L2 前尺度的保守量纲（避免 F_ema、R 等量纲差过大）
_SCALE_F = 10.0
_SCALE_R = 3.0
_SCALE_M = 1.0
_SCALE_UC = 1.0
_SCALE_UT = 1.0


@dataclass
class WordStateImprint:
    F_ema: float
    R: float
    m: float
    u_curiosity: float
    u_task: float
    context_before: str = ""
    context_after: str = ""
    occurrence_count: int = 1

    def feature_vector(self) -> Tuple[float, float, float, float, float]:
        return (
            self.F_ema / _SCALE_F,
            self.R / _SCALE_R,
            self.m / _SCALE_M,
            self.u_curiosity / _SCALE_UC,
            self.u_task / _SCALE_UT,
        )


def _l2(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))


@dataclass
class TokenMeta:
    first_seen_order: int
    last_access_seq: int = 0
    last_trigger_seq: int = 0
    trigger_success_count: int = 0
    imprint_count: int = 0
    is_dormant: bool = False


class WordStateMemory:
    """key = surface token（与 tokenizer.tokenize 输出一致），value = 印记列表。"""

    def __init__(
        self, *, capacity: int = 100, cognitive_debt_surprise_gamma: float = 0.5
    ) -> None:
        self.capacity = max(1, int(capacity))
        self._store: Dict[str, List[WordStateImprint]] = {}
        self._token_meta: Dict[str, TokenMeta] = {}
        #: 与印记并存：用户负反馈注入的「认知债务」∈[0,1]，仅在 internal_tick 路径衰减。
        self._cognitive_debt: Dict[str, float] = {}
        self._cognitive_debt_surprise_gamma = max(0.0, float(cognitive_debt_surprise_gamma))
        self._active_tokens: Set[str] = set()
        self._active_token_limit: Optional[int] = None
        self._seq = 0
        self._cold_scan_cursor = 0
        self._merged_imprints_total = 0
        self._associated_activated_total = 0
        self._cold_probe_total = 0
        self._structural_cache_max = _DEFAULT_STRUCTURAL_CACHE_MAX
        self._associated_tokens_cache: OrderedDict[
            str, Tuple[Tuple[str, int], ...]
        ] = OrderedDict()
        self._associated_tokens_cache_hits = 0
        self._associated_tokens_cache_misses = 0
        self._frontier_one_hop_cache: OrderedDict[str, Tuple[str, ...]] = OrderedDict()
        self._lazy_soft_freeze_tick = 0
        self._lazy_soft_freeze_stride = 4
        self._frontier_one_hop_cache_hits = 0
        self._frontier_one_hop_cache_misses = 0
        self._active_trim_defer_depth = 0
        self._stats_active_tokens = 0
        self._stats_dormant_tokens = 0
        self._progress_new_vocab_since_checkpoint = 0
        self._progress_new_vocab_samples: List[str] = []
        self._compute_device = "cpu"
        self._cache_version = 0
        self._torch_cache: Dict[str, Dict[str, object]] = {}

    def set_compute_device(self, device: str) -> str:
        requested = str(device or "cpu").strip() or "cpu"
        self._compute_device = requested
        return self._compute_device

    def _invalidate_cache(self) -> None:
        self._cache_version += 1
        self._torch_cache = {}

    def _invalidate_associated_cache(self, token: str) -> None:
        s = token.strip()
        if not s:
            return
        self._associated_tokens_cache.pop(s, None)
        self._frontier_one_hop_cache.pop(s, None)

    def _assoc_cache_touch_get(self, token: str) -> Optional[Tuple[Tuple[str, int], ...]]:
        cache = self._associated_tokens_cache
        val = cache.get(token)
        if val is not None:
            cache.move_to_end(token)
        return val

    def _assoc_cache_insert(self, token: str, value: Tuple[Tuple[str, int], ...]) -> None:
        cache = self._associated_tokens_cache
        if token in cache:
            del cache[token]
        cache[token] = value
        max_sz = max(1, int(self._structural_cache_max))
        while len(cache) > max_sz:
            cache.popitem(last=False)

    def _frontier_hop_touch_get(self, token: str) -> Optional[Tuple[str, ...]]:
        cache = self._frontier_one_hop_cache
        val = cache.get(token)
        if val is not None:
            cache.move_to_end(token)
        return val

    def _frontier_hop_insert(self, token: str, value: Tuple[str, ...]) -> None:
        cache = self._frontier_one_hop_cache
        if token in cache:
            del cache[token]
        cache[token] = value
        max_sz = max(1, int(self._structural_cache_max))
        while len(cache) > max_sz:
            cache.popitem(last=False)

    def _frontier_one_hop_neighbors(self, token: str) -> Tuple[str, ...]:
        s = token.strip()
        if not s:
            return ()
        cached = self._frontier_hop_touch_get(s)
        if cached is not None:
            self._frontier_one_hop_cache_hits += 1
            return cached
        self._frontier_one_hop_cache_misses += 1
        out = tuple(
            assoc for assoc, _count in self.associated_tokens(s, k=max(1, self.capacity))
        )
        self._frontier_hop_insert(s, out)
        return out

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _ensure_meta(self, token: str) -> TokenMeta:
        meta = self._token_meta.get(token)
        if meta is None:
            meta = TokenMeta(first_seen_order=len(self._token_meta))
            self._token_meta[token] = meta
        return meta

    def _priority_key(self, token: str) -> Tuple[int, int, int, int, int]:
        meta = self._ensure_meta(token)
        return (
            -int(meta.trigger_success_count),
            -int(meta.last_trigger_seq),
            -int(meta.last_access_seq),
            -int(meta.imprint_count),
            int(meta.first_seen_order),
        )

    def _activate_token(self, token: str) -> None:
        s = token.strip()
        if not s or s not in self._store:
            return
        meta = self._ensure_meta(s)
        was_dormant = meta.is_dormant
        meta.is_dormant = False
        self._active_tokens.add(s)
        if was_dormant:
            self._stats_dormant_tokens = max(0, self._stats_dormant_tokens - 1)
            self._stats_active_tokens += 1

    def _deactivate_token(self, token: str) -> None:
        s = token.strip()
        if not s:
            return
        meta = self._ensure_meta(s)
        was_active = not meta.is_dormant
        meta.is_dormant = True
        self._active_tokens.discard(s)
        if was_active:
            self._stats_active_tokens = max(0, self._stats_active_tokens - 1)
            self._stats_dormant_tokens += 1

    def _trim_active_tokens(
        self,
        max_active_tokens: int,
        *,
        protected: Optional[Set[str]] = None,
    ) -> None:
        max_active_tokens = max(0, int(max_active_tokens))
        protected_tokens = {tok for tok in (protected or set()) if tok in self._active_tokens}
        active_now = [tok for tok in self._active_tokens if tok in self._store]
        if len(active_now) <= max_active_tokens:
            return
        self._active_tokens = set(active_now)
        removable = [tok for tok in active_now if tok not in protected_tokens]
        removable.sort(key=lambda tok: (self._priority_key(tok), tok), reverse=True)
        while len(self._active_tokens) > max_active_tokens and removable:
            self._deactivate_token(removable.pop())

    def _maybe_trim_active_tokens(
        self,
        max_active_tokens: int,
        *,
        protected: Optional[Set[str]] = None,
    ) -> None:
        if self._active_trim_defer_depth > 0:
            return
        self._trim_active_tokens(max_active_tokens, protected=protected)

    def note_access(self, token: str) -> None:
        s = token.strip()
        if not s or s not in self._store:
            return
        meta = self._ensure_meta(s)
        self._activate_token(s)
        meta.last_access_seq = self._next_seq()
        if self._active_token_limit is not None:
            self._maybe_trim_active_tokens(self._active_token_limit, protected={s})

    def note_trigger_success(self, token: str) -> None:
        s = token.strip()
        if not s or s not in self._store:
            return
        meta = self._ensure_meta(s)
        seq = self._next_seq()
        self._activate_token(s)
        meta.last_access_seq = seq
        meta.last_trigger_seq = seq
        meta.trigger_success_count += 1
        if self._active_token_limit is not None:
            self._maybe_trim_active_tokens(self._active_token_limit, protected={s})

    @staticmethod
    def _can_merge_imprint(existing: WordStateImprint, new: WordStateImprint) -> bool:
        if existing.context_before != new.context_before:
            return False
        if existing.context_after != new.context_after:
            return False
        return all(
            abs(a - b) <= 1e-6
            for a, b in zip(existing.feature_vector(), new.feature_vector(), strict=True)
        )

    def record(self, token: str, imprint: WordStateImprint) -> None:
        if not token.strip():
            return
        s = token.strip()
        q = self._store.setdefault(s, [])
        is_new_vocab = len(q) == 0
        meta = self._ensure_meta(s)
        if is_new_vocab:
            self._stats_active_tokens += 1
            self._progress_new_vocab_since_checkpoint += 1
            if len(self._progress_new_vocab_samples) < 6:
                self._progress_new_vocab_samples.append(s)
        self._activate_token(s)
        for idx, old in enumerate(q):
            if self._can_merge_imprint(old, imprint):
                q[idx] = WordStateImprint(
                    F_ema=old.F_ema,
                    R=old.R,
                    m=old.m,
                    u_curiosity=old.u_curiosity,
                    u_task=old.u_task,
                    context_before=old.context_before,
                    context_after=old.context_after,
                    occurrence_count=int(old.occurrence_count) + max(1, int(imprint.occurrence_count)),
                )
                self._merged_imprints_total += max(1, int(imprint.occurrence_count))
                meta.imprint_count = len(q)
                self._invalidate_associated_cache(s)
                return
        if len(q) >= self.capacity:
            q[random.randrange(len(q))] = imprint
        else:
            q.append(imprint)
        meta.imprint_count = len(q)
        if self._active_token_limit is not None:
            self._maybe_trim_active_tokens(self._active_token_limit, protected={s})
        self._invalidate_associated_cache(s)
        self._invalidate_cache()

    def progress_new_vocab_snapshot(self) -> Tuple[int, List[str]]:
        """训练进度用：自上次快照以来新增的印记词数量及至多 6 个示例名；O(1) 重置。"""
        n = int(self._progress_new_vocab_since_checkpoint)
        samples = list(self._progress_new_vocab_samples)
        self._progress_new_vocab_since_checkpoint = 0
        self._progress_new_vocab_samples.clear()
        return n, samples

    def _rebuild_vocab_stats_counts(self) -> None:
        """从磁盘加载或一致性修复后重建活跃/休眠计数。"""
        active = 0
        dormant = 0
        for tok in self._store:
            meta = self._ensure_meta(tok)
            if meta.is_dormant:
                dormant += 1
            else:
                active += 1
        self._stats_active_tokens = active
        self._stats_dormant_tokens = dormant

    def tokens(self) -> List[str]:
        """所有已有印记的 surface token（按首次进入字典的顺序返回）。"""
        return list(self._store.keys())

    def record_tokens_with_context(
        self,
        tokens: List[str],
        *,
        F_ema: float,
        R: float,
        m: float,
        u_curiosity: float,
        u_task: float,
    ) -> None:
        """对句中每个 token 写入一条带前后邻接词的印记。"""
        touched: Set[str] = set()
        self._active_trim_defer_depth += 1
        try:
            for i, t in enumerate(tokens):
                if not t.strip():
                    continue
                touched.add(t.strip())
                cb = tokens[i - 1] if i > 0 else ""
                ca = tokens[i + 1] if i + 1 < len(tokens) else ""
                self.record(
                    t,
                    WordStateImprint(F_ema, R, m, u_curiosity, u_task, cb, ca),
                )
        finally:
            self._active_trim_defer_depth = max(0, self._active_trim_defer_depth - 1)
        if self._active_token_limit is not None:
            self._trim_active_tokens(self._active_token_limit, protected=touched)

    def semantic_surprise_for_token(
        self,
        token: str,
        current: WordStateImprint,
        *,
        hist_scan_limit: Optional[int] = None,
    ) -> float:
        """
        与当前向量最近的库存印记距离；无库存时返回较高默认值（「从未在这种语境里见过」）。
        ``hist_scan_limit``：若给定正整数，只对每条印记序列的末尾若干条求最小距离，
        用于训练期内心快路径在大容量印记下避免每条回合 O(capacity) 放大。
        """
        hist = self._store.get(token, [])
        if not hist:
            # 尚无印记时不视为「失配」，避免首轮全靠默认值误触发冲突稿
            return 0.0
        lim = hist_scan_limit
        if lim is not None and int(lim) > 0:
            hist = hist[-int(lim) :]
        cur = current.feature_vector()
        best = min(_l2(cur, h.feature_vector()) for h in hist)
        base = float(best)
        return base * self._cognitive_debt_surprise_scale(token)

    def _cognitive_debt_surprise_scale(self, token: str) -> float:
        s = str(token).strip()
        d = float(self._cognitive_debt.get(s, 0.0)) if s else 0.0
        return 1.0 + float(self._cognitive_debt_surprise_gamma) * d

    def inject_cognitive_debt(self, token: str, delta: float) -> None:
        """累加认知债务（上界 1）；不写回印记本体。"""
        s = str(token).strip()
        if not s or delta <= 0.0:
            return
        old = float(self._cognitive_debt.get(s, 0.0))
        self._cognitive_debt[s] = min(1.0, old + float(delta))

    def decay_cognitive_debt_tick(self, factor: float) -> None:
        """与内心 pending 同频衰减（由 ``internal_tick`` 路径调用）。"""
        if not self._cognitive_debt:
            return
        fac = max(0.0, min(1.0, float(factor)))
        self._cognitive_debt = {
            k: v * fac
            for k, v in self._cognitive_debt.items()
            if v * fac > 1e-6
        }

    def max_surprise_in_tokens(
        self, tokens: List[str], current: WordStateImprint
    ) -> Tuple[float, str]:
        if not tokens:
            return 0.0, ""
        surps: List[Tuple[float, str]] = [
            (self.semantic_surprise_for_token(t, current), t) for t in tokens if t.strip()
        ]
        if not surps:
            return 0.0, ""
        return max(surps, key=lambda x: x[0])

    def top_surprises(
        self,
        current: WordStateImprint,
        *,
        k: int,
        exclude: Optional[Set[str]] = None,
        jitter: float = 0.0,
        rng: Optional[random.Random] = None,
    ) -> List[Tuple[float, str]]:
        """遍历全库，返回当前快照下失配度最高的前 k 个 token。"""
        if k <= 0:
            return []
        excluded = exclude or set()
        noise = rng or random
        torch_scored = self._top_surprises_torch(current, exclude=excluded)
        if torch_scored is not None:
            if jitter > 0.0:
                torch_scored = [
                    (max(0.0, surp + noise.uniform(-jitter, jitter)), tok)
                    for surp, tok in torch_scored
                ]
            torch_scored = [
                (
                    float(surp) * self._cognitive_debt_surprise_scale(tok),
                    tok,
                )
                for surp, tok in torch_scored
            ]
            torch_scored.sort(key=lambda x: (-x[0], x[1]))
            return torch_scored[:k]
        scored: List[Tuple[float, str]] = []
        for tok in self._store:
            if tok in excluded:
                continue
            surp = self.semantic_surprise_for_token(tok, current)
            if jitter > 0.0:
                surp = max(0.0, surp + noise.uniform(-jitter, jitter))
            scored.append((float(surp), tok))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return scored[:k]

    def hot_tokens(
        self,
        *,
        limit: int,
        exclude: Optional[Set[str]] = None,
    ) -> List[str]:
        if limit <= 0:
            return []
        excluded = exclude or set()
        scored: List[Tuple[Tuple[int, int, int, int, int], str]] = []
        for tok in sorted(self._active_tokens):
            if tok in excluded:
                continue
            if tok not in self._store:
                continue
            scored.append((self._priority_key(tok), tok))
        scored.sort(key=lambda x: (x[0], x[1]))
        return [tok for _prio, tok in scored[:limit]]

    def cold_scan_tokens(
        self,
        *,
        limit: int,
        shard_count: int = 16,
        exclude: Optional[Set[str]] = None,
        include_dormant: bool = False,
    ) -> List[str]:
        if limit <= 0:
            return []
        if not self._store:
            return []
        excluded = exclude or set()
        shard_count = max(1, int(shard_count))
        shard_idx = self._cold_scan_cursor % shard_count
        self._cold_scan_cursor = (self._cold_scan_cursor + 1) % shard_count
        out: List[str] = []
        # 直接枚举 _store，避免 tokens() 复制全部键（十万级以上时每轮 cold 分配巨大列表）
        for idx, tok in enumerate(self._store):
            if tok in excluded:
                continue
            meta = self._ensure_meta(tok)
            if meta.is_dormant and not include_dormant:
                continue
            if (idx % shard_count) != shard_idx:
                continue
            out.append(tok)
            if len(out) >= limit:
                break
        return out

    def soft_freeze_cold_tokens(self, *, max_active_tokens: int) -> List[str]:
        max_active_tokens = max(0, int(max_active_tokens))
        scored: List[Tuple[Tuple[int, int, int, int, int], str]] = []
        for tok in self._store:
            scored.append((self._priority_key(tok), tok))
        scored.sort(key=lambda x: (x[0], x[1]))
        keep = {tok for _prio, tok in scored[:max_active_tokens]}
        self._active_tokens = set(keep)
        self._active_token_limit = max_active_tokens
        frozen: List[str] = []
        for _prio, tok in scored:
            if tok in keep:
                self._activate_token(tok)
            else:
                self._deactivate_token(tok)
                frozen.append(tok)
        return frozen

    def activation_frontier_tokens(
        self,
        *,
        recent_tokens: Optional[List[str]] = None,
        pending_tokens: Optional[Set[str]] = None,
        limit: int,
        one_hop_budget: int = 8,
        two_hop_budget: int = 4,
        two_hop_per_token: int = 2,
        cold_budget: int = 2,
        shard_count: int = 16,
        max_active_tokens: Optional[int] = None,
        exclude: Optional[Set[str]] = None,
    ) -> List[str]:
        if limit <= 0:
            return []
        excluded = set(exclude or set())
        selected: List[str] = []
        selected_set: Set[str] = set()
        first_hop: List[str] = []

        def _append(tok: str) -> bool:
            s = tok.strip()
            if not s or s in excluded or s in selected_set or s not in self._store:
                return False
            selected.append(s)
            selected_set.add(s)
            return True

        for tok in recent_tokens or []:
            if len(selected) >= limit:
                break
            _append(tok)
        for tok in sorted(pending_tokens or set()):
            if len(selected) >= limit:
                break
            _append(tok)
        if not selected and limit > 0:
            for tok in self.hot_tokens(limit=min(limit, max(1, int(one_hop_budget))), exclude=excluded):
                if len(selected) >= limit:
                    break
                _append(tok)

        remaining_first = max(0, int(one_hop_budget))
        seeds = list(selected)
        for tok in seeds:
            if remaining_first <= 0 or len(selected) >= limit:
                break
            neighbors = self._frontier_one_hop_neighbors(tok)
            for assoc in neighbors[:remaining_first]:
                if remaining_first <= 0 or len(selected) >= limit:
                    break
                if _append(assoc):
                    first_hop.append(assoc)
                    self._associated_activated_total += 1
                    remaining_first -= 1

        remaining_second = max(0, int(two_hop_budget))
        per_token_second = max(0, int(two_hop_per_token))
        for tok in first_hop:
            if remaining_second <= 0 or len(selected) >= limit:
                break
            neighbors = self._frontier_one_hop_neighbors(tok)
            for assoc in neighbors[: min(per_token_second, remaining_second)]:
                if remaining_second <= 0 or len(selected) >= limit:
                    break
                if _append(assoc):
                    self._associated_activated_total += 1
                    remaining_second -= 1

        if cold_budget > 0 and len(selected) < limit:
            for tok in self.cold_scan_tokens(
                limit=max(0, int(cold_budget)),
                shard_count=shard_count,
                exclude=selected_set | excluded,
                include_dormant=True,
            ):
                if len(selected) >= limit:
                    break
                if _append(tok):
                    self._cold_probe_total += 1

        self._active_trim_defer_depth += 1
        try:
            for tok in selected:
                self.note_access(tok)
        finally:
            self._active_trim_defer_depth = max(0, self._active_trim_defer_depth - 1)
        if max_active_tokens is not None:
            self._active_token_limit = max(0, int(max_active_tokens))
            self._trim_active_tokens(self._active_token_limit, protected=selected_set)
        return selected[:limit]

    def top_surprises_lazy(
        self,
        current: WordStateImprint,
        *,
        k: int,
        recent_tokens: Optional[List[str]] = None,
        pending_tokens: Optional[Set[str]] = None,
        hot_budget: int = 8,
        cold_budget: int = 8,
        shard_count: int = 16,
        max_active_tokens: Optional[int] = None,
        exclude: Optional[Set[str]] = None,
        jitter: float = 0.0,
        rng: Optional[random.Random] = None,
    ) -> List[Tuple[float, str]]:
        if k <= 0:
            return []
        if max_active_tokens is not None:
            self._lazy_soft_freeze_tick += 1
            limit = max(0, int(max_active_tokens))
            if (
                self._active_token_limit != limit
                or self._lazy_soft_freeze_tick % self._lazy_soft_freeze_stride == 1
            ):
                self.soft_freeze_cold_tokens(max_active_tokens=limit)
        excluded = set(exclude or set())
        selected: List[str] = []

        def _append(tok: str) -> None:
            s = tok.strip()
            if not s or s in excluded or s in selected or s not in self._store:
                return
            selected.append(s)
            self.note_access(s)

        for tok in recent_tokens or []:
            _append(tok)
        for tok in sorted(pending_tokens or set()):
            _append(tok)
        if len(selected) < k:
            for tok in self.hot_tokens(limit=max(0, hot_budget), exclude=set(selected) | excluded):
                _append(tok)
                if len(selected) >= k:
                    break
        if len(selected) < k:
            for tok in self.cold_scan_tokens(
                limit=max(0, cold_budget),
                shard_count=shard_count,
                exclude=set(selected) | excluded,
            ):
                _append(tok)
                if len(selected) >= k:
                    break

        noise = rng or random
        scored: List[Tuple[float, str]] = []
        for tok in selected:
            surp = self.semantic_surprise_for_token(tok, current)
            if jitter > 0.0:
                surp = max(0.0, surp + noise.uniform(-jitter, jitter))
            scored.append((float(surp), tok))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return scored[:k]

    def _top_surprises_torch(
        self,
        current: WordStateImprint,
        *,
        exclude: Set[str],
    ) -> Optional[List[Tuple[float, str]]]:
        if torch is None:
            return None
        device = str(self._compute_device or "cpu")
        if device == "cpu":
            return None
        cache = self._get_torch_cache(device)
        if cache is None:
            return None
        token_list = cast(List[str], cache["tokens"])
        feats = cache.get("features")
        owners = cache.get("owners")
        if feats is None or owners is None or not token_list:
            return []
        cur = torch.tensor(
            current.feature_vector(),
            dtype=torch.float32,
            device=device,
        )
        dists = torch.linalg.vector_norm(feats - cur, dim=1)
        mins = torch.full(
            (len(token_list),),
            float("inf"),
            dtype=torch.float32,
            device=device,
        )
        try:
            mins.scatter_reduce_(0, owners, dists, reduce="amin", include_self=True)
        except Exception:
            return None
        out: List[Tuple[float, str]] = []
        for tok, surp in zip(token_list, mins.detach().cpu().tolist(), strict=True):
            if tok in exclude:
                continue
            out.append((float(surp), tok))
        return out

    def _get_torch_cache(self, device: str) -> Optional[Dict[str, object]]:
        if torch is None:
            return None
        cached = self._torch_cache.get(device)
        if cached and int(cached.get("version", -1)) == self._cache_version:
            return cached
        token_list: List[str] = []
        feat_rows: List[Tuple[float, float, float, float, float]] = []
        owner_rows: List[int] = []
        for idx, (tok, imprints) in enumerate(self._store.items()):
            if not imprints:
                continue
            token_list.append(tok)
            for imprint in imprints:
                feat_rows.append(imprint.feature_vector())
                owner_rows.append(idx)
        if not feat_rows:
            cache = {"version": self._cache_version, "tokens": token_list, "features": None, "owners": None}
            self._torch_cache[device] = cache
            return cache
        try:
            features = torch.tensor(feat_rows, dtype=torch.float32, device=device)
            owners = torch.tensor(owner_rows, dtype=torch.long, device=device)
        except Exception:
            return None
        cache = {
            "version": self._cache_version,
            "tokens": token_list,
            "features": features,
            "owners": owners,
        }
        self._torch_cache[device] = cache
        return cache

    @staticmethod
    def _is_association_token(token: str) -> bool:
        s = token.strip()
        return bool(s) and any(ch.isalnum() for ch in s)

    def associated_tokens(self, token: str, *, k: int = 5) -> List[Tuple[str, int]]:
        """从 token 的历史前后邻接印记中统计关联词。"""
        if k <= 0:
            return []
        s_token = token.strip()
        if not s_token:
            return []
        cached = self._assoc_cache_touch_get(s_token)
        if cached is not None:
            self._associated_tokens_cache_hits += 1
            return list(cached[:k])
        self._associated_tokens_cache_misses += 1
        counts: Dict[str, int] = {}
        for imprint in self._store.get(s_token, []):
            weight = max(1, int(imprint.occurrence_count))
            for assoc in (imprint.context_before, imprint.context_after):
                s = assoc.strip()
                if s == s_token or not self._is_association_token(s):
                    continue
                counts[s] = counts.get(s, 0) + weight
        out = tuple(sorted(counts.items(), key=lambda x: (-x[1], x[0])))
        self._assoc_cache_insert(s_token, out)
        return list(out[:k])

    def stats(self) -> Dict[str, int]:
        return {
            "total_tokens": len(self._store),
            "active_tokens": int(self._stats_active_tokens),
            "dormant_tokens": int(self._stats_dormant_tokens),
            "merged_imprints": int(self._merged_imprints_total),
            "associated_activated_tokens": int(self._associated_activated_total),
            "cold_probe_tokens": int(self._cold_probe_total),
        }

    def to_dict(self) -> Dict[str, object]:
        out: Dict[str, List[Dict[str, object]]] = {}
        for tok, lst in self._store.items():
            out[tok] = [cast(Dict[str, object], asdict(im)) for im in lst]
        meta_out: Dict[str, Dict[str, object]] = {}
        for tok, meta in self._token_meta.items():
            meta_out[tok] = cast(Dict[str, object], asdict(meta))
        return {
            "format": WORD_STATE_MEMORY_FORMAT,
            "capacity": self.capacity,
            "tokens": out,
            "token_meta": meta_out,
            "seq": self._seq,
            "cold_scan_cursor": self._cold_scan_cursor,
            "merged_imprints_total": self._merged_imprints_total,
            "associated_activated_total": self._associated_activated_total,
            "cold_probe_total": self._cold_probe_total,
            "cognitive_debt": {
                str(k): float(v) for k, v in self._cognitive_debt.items()
            },
            "cognitive_debt_surprise_gamma": float(self._cognitive_debt_surprise_gamma),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "WordStateMemory":
        fmt = data.get("format")
        if fmt != WORD_STATE_MEMORY_FORMAT:
            raise ValueError(f"不支持的 word imprints format: {fmt!r}")
        cap = int(data.get("capacity", 100))
        gamma_raw = data.get("cognitive_debt_surprise_gamma")
        gamma = 0.5 if gamma_raw is None else float(cast(object, gamma_raw))
        mem = cls(capacity=cap, cognitive_debt_surprise_gamma=gamma)
        raw_tok = data.get("tokens")
        if isinstance(raw_tok, dict):
            for tok, lst in raw_tok.items():
                if not isinstance(tok, str) or not isinstance(lst, list):
                    continue
                for item in lst:
                    if not isinstance(item, dict):
                        continue
                    mem._store.setdefault(tok, []).append(
                        WordStateImprint(
                            F_ema=float(item.get("F_ema", 0.0)),
                            R=float(item.get("R", 0.0)),
                            m=float(item.get("m", 0.0)),
                            u_curiosity=float(item.get("u_curiosity", 0.0)),
                            u_task=float(item.get("u_task", 0.0)),
                            context_before=str(item.get("context_before", "")),
                            context_after=str(item.get("context_after", "")),
                            occurrence_count=max(1, int(item.get("occurrence_count", 1))),
                        )
                    )
                if len(mem._store.get(tok, [])) > mem.capacity:
                    mem._store[tok] = mem._store[tok][-mem.capacity :]
                meta = mem._ensure_meta(tok)
                meta.imprint_count = len(mem._store.get(tok, []))
        raw_cd = data.get("cognitive_debt")
        if isinstance(raw_cd, dict):
            mem._cognitive_debt = {
                str(k): max(0.0, min(1.0, float(cast(object, v))))
                for k, v in raw_cd.items()
                if str(k).strip()
            }
        raw_meta = data.get("token_meta")
        if isinstance(raw_meta, dict):
            for tok, item in raw_meta.items():
                if not isinstance(tok, str) or not isinstance(item, dict):
                    continue
                meta = mem._ensure_meta(tok)
                meta.first_seen_order = int(item.get("first_seen_order", meta.first_seen_order))
                meta.last_access_seq = int(item.get("last_access_seq", 0))
                meta.last_trigger_seq = int(item.get("last_trigger_seq", 0))
                meta.trigger_success_count = int(item.get("trigger_success_count", 0))
                meta.imprint_count = int(item.get("imprint_count", meta.imprint_count))
                meta.is_dormant = bool(item.get("is_dormant", False))
        mem._active_tokens = {
            tok
            for tok in mem.tokens()
            if not mem._ensure_meta(tok).is_dormant
        }
        mem._rebuild_vocab_stats_counts()
        mem._progress_new_vocab_since_checkpoint = 0
        mem._progress_new_vocab_samples.clear()
        mem._seq = int(data.get("seq", 0))
        mem._cold_scan_cursor = int(data.get("cold_scan_cursor", 0))
        mem._merged_imprints_total = int(data.get("merged_imprints_total", 0))
        mem._associated_activated_total = int(data.get("associated_activated_total", 0))
        mem._cold_probe_total = int(data.get("cold_probe_total", 0))
        return mem

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "WordStateMemory":
        with Path(path).open(encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            raise ValueError("印记文件顶层必须是对象")
        return cls.from_dict(cast(Dict[str, object], raw))


__all__ = [
    "WORD_STATE_MEMORY_FORMAT",
    "WordStateImprint",
    "WordStateMemory",
]
