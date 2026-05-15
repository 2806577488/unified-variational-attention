from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
from typing import Callable, Dict, List, Literal, Optional, Union, cast

from .checkpoint_json import infer_compression_from_path, read_json_document, write_json_document


def _unigram_to_compact_cols(unigram: Dict[str, int]) -> Dict[str, List[Union[str, int]]]:
    keys = sorted(unigram.keys())
    return {"k": [str(k) for k in keys], "v": [int(unigram[k]) for k in keys]}


def _unigram_from_compact(raw: object) -> Dict[str, int]:
    """兼容列式 ``{"k":[],"v":[]}`` 与旧式 ``[[tok,cnt],...]``。"""
    if isinstance(raw, dict):
        ks = raw.get("k")
        vs = raw.get("v")
        if isinstance(ks, list) and isinstance(vs, list) and len(ks) == len(vs):
            return {str(k): int(cast(object, v)) for k, v in zip(ks, vs, strict=True)}
        return {}
    out: Dict[str, int] = {}
    if not isinstance(raw, list):
        return out
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            out[str(item[0])] = int(cast(object, item[1]))
    return out


def _nested_transition_to_compact(
    nested: Dict[str, Dict[str, int]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for prev in sorted(nested.keys()):
        row = nested[prev]
        nk = [str(k) for k, v in sorted(row.items())]
        nv = [int(v) for _, v in sorted(row.items())]
        rows.append({"p": str(prev), "nk": nk, "nv": nv})
    return rows


def _nested_transition_from_compact(raw: object) -> Dict[str, Dict[str, int]]:
    """兼容 ``{"p","nk","nv"}`` 行式与旧式 ``[prev,[[next,cnt],...]]``。"""
    out: Dict[str, Dict[str, int]] = {}
    if not isinstance(raw, list):
        return out
    for row in raw:
        if isinstance(row, dict):
            prev_s = str(row.get("p", ""))
            nk = row.get("nk")
            nv = row.get("nv")
            if (
                isinstance(nk, list)
                and isinstance(nv, list)
                and len(nk) == len(nv)
                and prev_s
            ):
                out[prev_s] = {
                    str(k): int(cast(object, v)) for k, v in zip(nk, nv, strict=True)
                }
            continue
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            continue
        prev_s = str(row[0])
        inner_raw = row[1]
        if not isinstance(inner_raw, list):
            continue
        d: Dict[str, int] = {}
        for pair in inner_raw:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                d[str(pair[0])] = int(cast(object, pair[1]))
        out[prev_s] = d
    return out


def _apply_ngram_dict_to_tokenizer(
    tokenizer: "PrecisionTokenizer", data: Dict[str, object]
) -> None:
    uc = data.get("unigram_c")
    if isinstance(uc, (dict, list)):
        tokenizer.unigram = _unigram_from_compact(uc)
    else:
        tokenizer.unigram = dict(data.get("unigram", {}))

    if isinstance(data.get("bigram_c"), list):
        tokenizer.bigram = _nested_transition_from_compact(data["bigram_c"])
    else:
        tokenizer.bigram = {
            str(k): dict(v) for k, v in dict(data.get("bigram", {})).items()
        }

    if isinstance(data.get("follow_c"), list):
        tokenizer.follow_counts = _nested_transition_from_compact(data["follow_c"])
    else:
        tokenizer.follow_counts = {
            str(k): dict(v) for k, v in dict(data.get("follow_counts", {})).items()
        }


@dataclass
class TokenizationTrace:
    tokens: List[str]
    precision: List[float]
    surprise: List[float]
    boundary_indices: List[int]
    resource: List[float]
    mind_wander: List[float]
    auto_rest_count: int
    mean_surprise: float = 0.0


TraceAccelImpl = Callable[..., TokenizationTrace]
TraceBatchAccelImpl = Callable[["PrecisionTokenizer", List[str]], List[float]]

try:
    from ._tokenizer_accel import (  # type: ignore[import-not-found]
        trace_mean_surprise_batch as _TRACE_ACCEL_BATCH_IMPL,
        trace_tokenize as _TRACE_ACCEL_IMPL,
    )
except ImportError:
    _TRACE_ACCEL_IMPL = None
    _TRACE_ACCEL_BATCH_IMPL = None


class PrecisionTokenizer:
    def __init__(
        self,
        *,
        alpha: float = 1.1,
        beta: float = 0.3,
        pi_min: float = 0.2,
        sigma0: float = 0.8,
        decay: float = 0.25,
        boundary_threshold: float = 0.8,
        surprise_threshold: float = 1.6,
        dt: float = 0.8,
        R_max: float = 1.0,
        rho: float = 0.12,
        lambda_deplete: float = 0.04,
        tau_m: float = 8.0,
        theta_F: float = 1.2,
        R_crit: float = 0.35,
        auto_rest_threshold: float = 0.2,
        auto_resume_threshold: float = 0.5,
        auto_rest_steps: int = 8,
        R_base: float = 0.5,
        R_max_cap: float = 3.0,
        tau_grow: float = 400.0,
        eta_learn: float = 0.04,
        lambda_grow: float = 0.002,
        F_ema_beta: float = 0.04,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.pi_min = pi_min
        self.sigma0 = sigma0
        self.decay = decay
        self.boundary_threshold = boundary_threshold
        self.surprise_threshold = surprise_threshold
        self.dt = dt
        self.R_max = R_max
        self.rho = rho
        self.lambda_deplete = lambda_deplete
        self.tau_m = tau_m
        self.theta_F = theta_F
        self.R_crit = R_crit
        self.auto_rest_threshold = auto_rest_threshold
        self.auto_resume_threshold = auto_resume_threshold
        self.auto_rest_steps = auto_rest_steps
        self.R_base = R_base
        self.R_max_cap = R_max_cap
        self.tau_grow = tau_grow
        self.eta_learn = eta_learn
        self.lambda_grow = lambda_grow
        self.F_ema_beta = F_ema_beta

        self.unigram: Dict[str, int] = {}
        self.bigram: Dict[str, Dict[str, int]] = {}
        self.follow_counts: Dict[str, Dict[str, int]] = {}
        self.total_chars = 0
        self.fitted = False
        self.R = self.R_max
        self.m = 0.0
        self.F_ema = max(0.1, float(theta_F))

    def _reset_stats(self) -> None:
        self.unigram = {}
        self.bigram = {}
        self.follow_counts = {}
        self.total_chars = 0
        self.fitted = False
        self.F_ema = max(0.1, float(self.theta_F))

    def _observe_line(self, line: str) -> None:
        padded = "^" + line + "$"
        pi_local = self.pi_min
        f_sum = 0.0
        f_count = 0
        for i, ch in enumerate(padded):
            if i > 0:
                prev = padded[i - 1]
                prob = self._next_prob(prev, ch)
                surprise = -math.log(max(prob, 1e-9))
                novelty = self.novelty_signal(surprise)
                d_pi = self.alpha * novelty - self.decay * (pi_local - self.pi_min)
                pi_local = max(self.pi_min, pi_local + self.dt * d_pi * (1.0 - self.m))
                usage = max(0.0, pi_local - self.pi_min)
                free_energy = surprise * (0.2 + pi_local)
                f_sum += free_energy
                f_count += 1
                self._update_resource(usage=usage, free_energy=free_energy)

            self.unigram[ch] = self.unigram.get(ch, 0) + 1
            self.total_chars += 1
            if i > 0:
                prev = padded[i - 1]
                self.bigram.setdefault(prev, {})
                self.bigram[prev][ch] = self.bigram[prev].get(ch, 0) + 1
        for i in range(1, len(line)):
            prev = line[i - 1]
            nxt = line[i]
            self.follow_counts.setdefault(prev, {})
            self.follow_counts[prev][nxt] = self.follow_counts[prev].get(nxt, 0) + 1

        if f_count > 0:
            self._grow_R_max_from_mean_free_energy(f_sum / f_count)

    def ingest_interaction_line(self, line: str) -> None:
        """
        在线并入用户（或环境）表面字串：只更新 unigram / bigram / follow_counts，
        不重复跑训练用精度内环，也不再扣一轮 R、m（对话已用 trace 付过成本）。
        后续同一串的转移概率会略变，惊奇与涌现边界随之缓慢漂移。
        """
        if not line:
            return
        padded = "^" + line + "$"
        unigram = self.unigram
        bigram = self.bigram
        follow_counts = self.follow_counts
        for ch in padded:
            unigram[ch] = unigram.get(ch, 0) + 1
        self.total_chars += len(padded)
        for prev, ch in zip(padded, padded[1:]):
            row = bigram.get(prev)
            if row is None:
                row = {}
                bigram[prev] = row
            row[ch] = row.get(ch, 0) + 1
        for prev, nxt in zip(line, line[1:]):
            row = follow_counts.get(prev)
            if row is None:
                row = {}
                follow_counts[prev] = row
            row[nxt] = row.get(nxt, 0) + 1
        self.fitted = True

    def ingest_interaction_lines(self, lines: List[str]) -> int:
        """
        批量并入多条表面字串，语义上等价于多次 `ingest_interaction_line`，
        但尽量先在局部聚合，减少高频字典写入。
        """
        valid_lines = [line for line in lines if line]
        if not valid_lines:
            return 0
        unigram_add: Counter[str] = Counter()
        bigram_add: Dict[str, Counter[str]] = {}
        follow_add: Dict[str, Counter[str]] = {}
        total_chars = 0
        for line in valid_lines:
            padded = "^" + line + "$"
            unigram_add.update(padded)
            total_chars += len(padded)
            for prev, ch in zip(padded, padded[1:]):
                row = bigram_add.setdefault(prev, Counter())
                row[ch] += 1
            for prev, nxt in zip(line, line[1:]):
                row = follow_add.setdefault(prev, Counter())
                row[nxt] += 1

        unigram = self.unigram
        for ch, count in unigram_add.items():
            unigram[ch] = unigram.get(ch, 0) + count

        bigram = self.bigram
        for prev, counts in bigram_add.items():
            row = bigram.get(prev)
            if row is None:
                row = {}
                bigram[prev] = row
            for ch, count in counts.items():
                row[ch] = row.get(ch, 0) + count

        follow_counts = self.follow_counts
        for prev, counts in follow_add.items():
            row = follow_counts.get(prev)
            if row is None:
                row = {}
                follow_counts[prev] = row
            for nxt, count in counts.items():
                row[nxt] = row.get(nxt, 0) + count

        self.total_chars += total_chars
        self.fitted = True
        return len(valid_lines)

    def _grow_R_max_from_mean_free_energy(self, mean_F: float) -> None:
        self.F_ema = (1.0 - self.F_ema_beta) * self.F_ema + self.F_ema_beta * max(mean_F, 1e-6)
        d_rmax = (1.0 / max(1e-6, self.tau_grow)) * (
            self.eta_learn / max(self.F_ema, 0.05) - self.lambda_grow * (self.R_max - self.R_base)
        )
        self.R_max = min(self.R_max_cap, max(self.R_base, self.R_max + d_rmax))
        self.R = min(self.R, self.R_max)

    def fit(self, corpus: List[str]) -> None:
        self._reset_stats()
        for line in corpus:
            self._observe_line(line)
        self.fitted = True

    def fit_stream(self, lines) -> int:
        self._reset_stats()
        seen = 0
        for line in lines:
            if not line:
                continue
            self._observe_line(line)
            seen += 1
        self.fitted = True
        return seen

    def partial_fit(self, corpus: List[str]) -> int:
        seen = 0
        for line in corpus:
            if not line:
                continue
            self._observe_line(line)
            seen += 1
        self.fitted = True
        return seen

    def partial_fit_stream(self, lines) -> int:
        seen = 0
        for line in lines:
            if not line:
                continue
            self._observe_line(line)
            seen += 1
        self.fitted = True
        return seen

    def _sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def _update_resource(self, usage: float, free_energy: float) -> None:
        target = self._sigmoid(free_energy - self.theta_F) * max(0.0, self.R_crit - self.R)
        dm = (target - self.m) / max(1e-6, self.tau_m)
        self.m = min(1.0, max(0.0, self.m + self.dt * dm))

        dR = self.rho * self.m * (self.R_max - self.R) - self.lambda_deplete * max(0.0, usage)
        self.R = min(self.R_max, max(0.0, self.R + self.dt * dR))

    def idle(self, steps: int = 1) -> None:
        for _ in range(max(0, steps)):
            self._update_resource(usage=0.0, free_energy=0.0)

    def resource_state(self) -> Dict[str, float]:
        return {"R": self.R, "m": self.m, "R_max": self.R_max, "F_ema": self.F_ema}

    def to_dict(
        self, *, ngram_layout: Literal["compact_v1", "dict"] = "compact_v1"
    ) -> Dict[str, object]:
        """
        序列化。默认 ``compact_v1``：
        ``unigram_c`` 为列式 ``{"k":[字符…],"v":[计数…]}``；
        ``bigram_c`` / ``follow_c`` 为按 ``prev`` 排序的行列表
        ``[{"p":prev,"nk":[后继字符…],"nv":[计数…]}, …]``。
        加载时仍兼容旧式 ``[[prev,[[next,cnt],…]], …]`` 与明文 ``unigram``/``bigram`` 嵌套字典。
        ``dict``：旧式嵌套字典，便于人工阅读。
        """
        core: Dict[str, object] = {
            "alpha": self.alpha,
            "beta": self.beta,
            "pi_min": self.pi_min,
            "sigma0": self.sigma0,
            "decay": self.decay,
            "boundary_threshold": self.boundary_threshold,
            "surprise_threshold": self.surprise_threshold,
            "dt": self.dt,
            "R_max": self.R_max,
            "rho": self.rho,
            "lambda_deplete": self.lambda_deplete,
            "tau_m": self.tau_m,
            "theta_F": self.theta_F,
            "R_crit": self.R_crit,
            "auto_rest_threshold": self.auto_rest_threshold,
            "auto_resume_threshold": self.auto_resume_threshold,
            "auto_rest_steps": self.auto_rest_steps,
            "R_base": self.R_base,
            "R_max_cap": self.R_max_cap,
            "tau_grow": self.tau_grow,
            "eta_learn": self.eta_learn,
            "lambda_grow": self.lambda_grow,
            "F_ema_beta": self.F_ema_beta,
            "F_ema": self.F_ema,
            "R": self.R,
            "m": self.m,
            "total_chars": self.total_chars,
            "fitted": self.fitted,
        }
        if ngram_layout == "compact_v1":
            core["unigram_c"] = _unigram_to_compact_cols(self.unigram)
            core["bigram_c"] = _nested_transition_to_compact(self.bigram)
            core["follow_c"] = _nested_transition_to_compact(self.follow_counts)
            core["ngram_layout"] = "compact_v1"
        elif ngram_layout == "dict":
            core["unigram"] = self.unigram
            core["bigram"] = self.bigram
            core["follow_counts"] = self.follow_counts
        else:
            raise ValueError(f"未知 ngram_layout: {ngram_layout!r}")
        return core

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "PrecisionTokenizer":
        tokenizer = cls(
            alpha=float(data.get("alpha", 1.1)),
            beta=float(data.get("beta", 0.3)),
            pi_min=float(data.get("pi_min", 0.2)),
            sigma0=float(data.get("sigma0", 0.8)),
            decay=float(data.get("decay", 0.25)),
            boundary_threshold=float(data.get("boundary_threshold", 0.8)),
            surprise_threshold=float(data.get("surprise_threshold", 1.6)),
            dt=float(data.get("dt", 0.8)),
            R_max=float(data.get("R_max", 1.0)),
            rho=float(data.get("rho", 0.12)),
            lambda_deplete=float(data.get("lambda_deplete", 0.04)),
            tau_m=float(data.get("tau_m", 8.0)),
            theta_F=float(data.get("theta_F", 1.2)),
            R_crit=float(data.get("R_crit", 0.35)),
            auto_rest_threshold=float(data.get("auto_rest_threshold", 0.2)),
            auto_resume_threshold=float(data.get("auto_resume_threshold", 0.5)),
            auto_rest_steps=int(data.get("auto_rest_steps", 8)),
            R_base=float(data.get("R_base", 0.5)),
            R_max_cap=float(data.get("R_max_cap", 3.0)),
            tau_grow=float(data.get("tau_grow", 400.0)),
            eta_learn=float(data.get("eta_learn", 0.04)),
            lambda_grow=float(data.get("lambda_grow", 0.002)),
            F_ema_beta=float(data.get("F_ema_beta", 0.04)),
        )
        _apply_ngram_dict_to_tokenizer(tokenizer, data)
        tokenizer.total_chars = int(data.get("total_chars", 0))
        tokenizer.fitted = bool(data.get("fitted", False))
        tokenizer.R = float(data.get("R", tokenizer.R_max))
        tokenizer.m = float(data.get("m", 0.0))
        tokenizer.F_ema = float(data.get("F_ema", max(0.1, tokenizer.theta_F)))
        return tokenizer

    def save_model(
        self,
        path: str,
        *,
        compact: bool = True,
        ngram_layout: Literal["compact_v1", "dict"] = "compact_v1",
        compression: Optional[Literal["gzip", "zstd"]] = None,
    ) -> None:
        """写入 JSON（默认紧凑 + compact_v1 ngram）。路径以 ``.gz`` / ``.zst`` 结尾时自动 gzip/zstd 压缩。"""
        comp = compression if compression is not None else infer_compression_from_path(path)
        payload = self.to_dict(ngram_layout=ngram_layout)
        write_json_document(path, payload, compact=compact, compression=comp)

    @classmethod
    def load_model(cls, path: str) -> "PrecisionTokenizer":
        data = read_json_document(path)
        if not isinstance(data, dict):
            raise ValueError("Tokenizer model file format is invalid.")
        return cls.from_dict(data)

    def novelty_signal(self, surprise: float) -> float:
        return (surprise * surprise) / (self.sigma0 * self.sigma0 + surprise * surprise)

    def _next_prob(self, prev_ch: str, ch: str) -> float:
        prev_count = self.unigram.get(prev_ch, 0)
        vocab = max(1, len(self.unigram))
        next_count = self.bigram.get(prev_ch, {}).get(ch, 0)
        return (next_count + 1.0) / (prev_count + vocab)

    def _trace_python(
        self,
        text: str,
        *,
        collect_precision: bool = True,
        collect_surprise: bool = True,
        collect_boundary_indices: bool = True,
        collect_resource: bool = True,
        collect_mind: bool = True,
    ) -> TokenizationTrace:
        tokens: List[str] = []
        current = ""
        precision: List[float] = [] if collect_precision else []
        surprise_list: List[float] = [] if collect_surprise else []
        boundary_indices: List[int] = [] if collect_boundary_indices else []
        resource_trace: List[float] = [] if collect_resource else []
        mind_trace: List[float] = [] if collect_mind else []
        prev_surprise = 0.0
        prev_free_energy = 0.0
        prev_m = self.m
        cooldown = 0
        auto_rest_count = 0
        f_sum = 0.0
        f_count = 0
        surprise_sum = 0.0

        pi = self.pi_min
        prev_pi = pi
        prev_ch = "^"
        unigram = self.unigram
        bigram = self.bigram
        pi_min = self.pi_min
        alpha = self.alpha
        beta = self.beta
        dt = self.dt
        decay = self.decay
        surprise_threshold = self.surprise_threshold
        novelty_signal = self.novelty_signal
        update_resource = self._update_resource
        log_fn = math.log
        auto_rest_threshold = self.auto_rest_threshold
        auto_resume_threshold = self.auto_resume_threshold
        auto_rest_steps = self.auto_rest_steps

        for idx, ch in enumerate(text):
            while self.R <= auto_rest_threshold:
                self.idle(steps=auto_rest_steps)
                auto_rest_count += 1
                if self.R >= auto_resume_threshold:
                    break

            if ch.isspace():
                if current:
                    tokens.append(current)
                    current = ""
                prev_ch = "^"
                pi = pi_min
                prev_pi = pi
                prev_surprise = 0.0
                prev_free_energy = 0.0
                prev_m = self.m
                cooldown = 0
                continue

            prev_count = unigram.get(prev_ch, 0)
            vocab = max(1, len(unigram))
            next_count = bigram.get(prev_ch, {}).get(ch, 0)
            prob = (next_count + 1.0) / (prev_count + vocab)
            surprise = -log_fn(max(prob, 1e-9))
            novelty = novelty_signal(surprise)
            task = 0.2 if prev_ch.isalnum() and ch.isalnum() else 0.05
            gain = max(0.0, 1.0 - self.m)
            d_pi = gain * (alpha * novelty + beta * task) - decay * (pi - pi_min)
            pi = max(pi_min, pi + dt * d_pi)
            free_energy = surprise * (0.2 + pi)
            usage = max(0.0, pi - self.pi_min)
            update_resource(usage=usage, free_energy=free_energy)
            f_sum += free_energy
            f_count += 1
            surprise_sum += surprise
            if collect_resource:
                resource_trace.append(self.R)
            if collect_mind:
                mind_trace.append(self.m)

            surprise_jump = surprise - prev_surprise
            free_energy_jump = free_energy - prev_free_energy
            mind_jump = self.m - prev_m
            event_score = (
                0.45 * max(0.0, surprise_jump)
                + 0.25 * max(0.0, free_energy_jump)
                + 0.20 * max(0.0, pi - prev_pi)
                + 0.10 * max(0.0, mind_jump)
            )
            is_boundary = (
                current != ""
                and len(current) >= 3
                and cooldown <= 0
                and surprise > surprise_threshold
                and event_score > 0.36
            )
            if is_boundary:
                tokens.append(current)
                current = ch
                if collect_boundary_indices:
                    boundary_indices.append(idx)
                cooldown = 2
            else:
                current += ch
                cooldown = max(0, cooldown - 1)

            if collect_precision:
                precision.append(pi)
            if collect_surprise:
                surprise_list.append(surprise)
            prev_pi = pi
            prev_ch = ch
            prev_surprise = surprise
            prev_free_energy = free_energy
            prev_m = self.m

        if current:
            tokens.append(current)

        if f_count > 0:
            self._grow_R_max_from_mean_free_energy(f_sum / f_count)

        return TokenizationTrace(
            tokens=tokens,
            precision=precision,
            surprise=surprise_list,
            boundary_indices=boundary_indices,
            resource=resource_trace,
            mind_wander=mind_trace,
            auto_rest_count=auto_rest_count,
            mean_surprise=(surprise_sum / f_count) if f_count > 0 else 0.0,
        )

    def _trace(
        self,
        text: str,
        *,
        collect_precision: bool = True,
        collect_surprise: bool = True,
        collect_boundary_indices: bool = True,
        collect_resource: bool = True,
        collect_mind: bool = True,
    ) -> TokenizationTrace:
        if not self.fitted:
            raise ValueError("请先调用 fit(corpus) 训练分词器。")

        accel_impl = _TRACE_ACCEL_IMPL
        if accel_impl is not None:
            return accel_impl(
                self,
                text,
                collect_precision=collect_precision,
                collect_surprise=collect_surprise,
                collect_boundary_indices=collect_boundary_indices,
                collect_resource=collect_resource,
                collect_mind=collect_mind,
            )
        return self._trace_python(
            text,
            collect_precision=collect_precision,
            collect_surprise=collect_surprise,
            collect_boundary_indices=collect_boundary_indices,
            collect_resource=collect_resource,
            collect_mind=collect_mind,
        )

    def tokenize(self, text: str) -> List[str]:
        return self._trace(
            text,
            collect_precision=False,
            collect_surprise=False,
            collect_boundary_indices=False,
            collect_resource=False,
            collect_mind=False,
        ).tokens

    def mean_surprise(self, text: str) -> float:
        trace = self._trace(
            text,
            collect_precision=False,
            collect_surprise=False,
            collect_boundary_indices=False,
            collect_resource=False,
            collect_mind=False,
        )
        return float(trace.mean_surprise)

    def mean_surprise_batch(self, texts: List[str]) -> List[float]:
        """批量计算多条文本的平均惊奇，语义上等价于逐条调用 `mean_surprise`。"""
        if not self.fitted:
            raise ValueError("请先调用 fit(corpus) 训练分词器。")
        accel_batch_impl = _TRACE_ACCEL_BATCH_IMPL
        if accel_batch_impl is not None:
            return accel_batch_impl(self, texts)
        return [self.mean_surprise(text) for text in texts]

    def trace_tokenize(self, text: str) -> Dict[str, object]:
        trace = self._trace(text)
        return {
            "tokens": trace.tokens,
            "precision": trace.precision,
            "surprise": trace.surprise,
            "boundary_indices": trace.boundary_indices,
            "resource": trace.resource,
            "mind_wander": trace.mind_wander,
            "auto_rest_count": trace.auto_rest_count,
            "resource_state": self.resource_state(),
        }
