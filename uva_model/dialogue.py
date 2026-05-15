from __future__ import annotations

"""
认知对话回路 —— **动作选择由单一标量预期自由能（EFE）最小化驱动**，不用分支 if 树决定「该说哪一类话」。

**闭环**
1) 观测：用户文本经分词器 → 社会惊奇均值 ε_o,social；同时更新槽位 σ（均匀耦合惊奇）。
2) 内部动作候选：由意图与计划生成若干 **可执行的字符串动作**（保守叙述 / 回声式探索 / 印记冲突探针 × 若干「诚实度前缀」变体）。
   前缀 ``"" | "可能" | "我不知道，"`` 并列进入候选池，由 EFE 择优，而非 π 阈值硬插 hedge；
   以「我不知道」起头的草稿在 EFE 上加额外惩罚（见 ``EFE_W_UNKNOWN_HEDGE``），避免轻易选退路式回复。
3) **EFE(y)**（越小越好，在同一观测与冻结资源快照下评估草稿 y）：
   - **感官（语言）项**：ε_secondary(y)（冻结 R,m,R_max,F_ema 下对 y 再 trace）；探索动作乘社会风险
     ``1 + α·(1−rf)``，rf 为资源因子。
   - **与社会观测对齐**：``max(0, ε_secondary − κ·ε_o,social)``，惩罚「对用户闯入解释过头」的冗长内部惊奇。
   - **动作代价**：长度 × (疲劳 proxy)，疲劳随 m 与 R 枯竭升高 —— R 低、m 高时长输出自发变贵。
   - **认识价值（好奇）**：``−λ_explore·u^{curiosity}·𝟙[探索]``，好奇且资源许可时压低探索类动作的 EFE。
   - **印记消解倾向**：``−γ·min(2, ε_imprint)`` 仅计入「冲突探针」候选，鼓励在印记失配时外投可验证差异。
4) **重规划**：在资源因子 rf 决定的预算 ``depth ∈ {0,…,K}`` 内，将语义计划逐层收缩（删末段），
   每层重新枚举候选并取 **全局 EFE 最小**；胜出深度记为 ``replan_count``。无 meta_ok / pi_quick 捷径。
5) 词印记库：倾听结束写入 token 快照；冲突候选始终参与（若有焦点词），是否胜出完全由 EFE 决定。

意图萌发 `_germinate_intent` 仍为稀疏动力学读写（非输出模板）；后续可进一步并入同一 EFE 框架。

**最近注意（Phase 0）**：每轮外部 `produce_turn` 结束时将用户句 token 与印记焦点词写入固定长度 ``deque``。

**内心时间线（Phase 1）**：`internal_tick()` 在新→旧去重的 ``recent_attention_words`` 上算张力
（印记惊奇×(0.5+u_c)×(0.3+rf)+pending），惊奇与 EFE 共用 ``_internal_monologue_imprint_snap``（``u_task=0``，不随 trigger 文本变化）；
过阈则用 `_efe_best_reply(..., internal_monologue=True)`（无社会对齐超额项）；
落选且 EFE 过高则 `pending` 累加；外部 `produce_turn` 开场对 pending 衰减。不写印记、不 `hear`。
"""

import json
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Literal, Optional, Tuple, cast

from .checkpoint_json import infer_compression_from_path, read_json_document, write_json_document
from .tokenizer import PrecisionTokenizer
from .word_imprints import WORD_STATE_MEMORY_FORMAT, WordStateImprint, WordStateMemory

DIALOGUE_MODEL_FORMAT = "cognitive_dialogue_agent_v1"
DIALOGUE_MODEL_FORMAT_VERSION = 1
PREFERENCE_STATE_FORMAT = "preference_state_v1"

OutputBranch = Literal[
    "conservative", "explore_echo", "conflict_probe", "associative_probe"
]

SpeechAct = Literal["explore", "respond", "clarify"]


@dataclass
class CommunicativeIntent:
    """语义层交流意图（非词汇挑选表）。"""

    speech_act: SpeechAct
    anchor_slot: str
    tension_slot: str
    u_curiosity: float
    u_task: float


@dataclass
class DialogueTurn:
    user_text: str
    reply: str
    u_curiosity: float
    u_task: float
    pi_statement: float
    epsilon_social_in: float
    epsilon_secondary: float
    intent: CommunicativeIntent
    replan_count: int
    resource_snapshot: Dict[str, float] = field(default_factory=dict)
    output_branch: OutputBranch = "conservative"
    semantic_surprise_max: float = 0.0
    conflict_focus_token: str = ""
    #: 联想探针归因：与 ``associated_tokens`` 调用键一致；仅 ``associative_probe`` 时非空。
    association_trigger: str = ""
    association_pick: str = ""
    #: 胜出草稿的原始 EFE（未扣分支偏置）；供与内心独白仲裁对标。
    best_efe: float = 0.0

    @property
    def curiosity_u(self) -> float:
        """兼容旧字段名。"""
        return self.u_curiosity

    @property
    def mean_surprise_in(self) -> float:
        return self.epsilon_social_in

    @property
    def mean_surprise_sim(self) -> float:
        return self.epsilon_secondary


@dataclass
class InternalMonologue:
    """内心独白一次输出（无外部 hear；EFE 使用 internal_monologue= True，不含社会对齐超额项）。"""

    reply: str
    trigger_token: str
    tension: float
    best_efe: float
    output_branch: OutputBranch
    epsilon_secondary: float
    semantic_surprise_max: float
    u_curiosity: float
    association_trigger: str = ""
    association_pick: str = ""


class CognitiveDialogueAgent:
    """σ 槽位：四个并行的抽象不确定度维度（与社会惊奇均匀耦合），名称仅为区分键，非话题实体。"""

    SLOT_NAMES: Tuple[str, ...] = (
        "sigma_slot_1",
        "sigma_slot_2",
        "sigma_slot_3",
        "sigma_slot_4",
    )
    #: 旧版对话 JSON 中的 σ 键名（与 `SLOT_NAMES` 按下标一一对应）；加载时自动迁移。
    SIGMA_LEGACY_KEYS: Tuple[str, ...] = ("小明", "游戏", "类型", "偏好")
    #: EFE 中 ε_secondary 相对社会惊奇的松弛系数 κ。
    EFE_SOCIAL_ALIGN_KAPPA: float = 1.35
    #: 社会对齐项权重。
    EFE_W_SOCIAL: float = 0.45
    #: 长度代价权重（字符）。
    EFE_W_LEN: float = 0.012
    #: 探索动作在社会资源枯竭时的额外承诺成本 ∝ (1−rf)，避免「短回声惊奇过低」钻空子。
    EFE_EXPLORATION_COMMIT: float = 2.0
    #: 草稿以「我不知道」类诚实前缀起头时，加在 EFE 上的额外代价（越大越难被选中）。
    EFE_W_UNKNOWN_HEDGE: float = 5.0
    #: 仲裁时从外部 EFE 扣除的「社会入场券」强度：ticket = scale * EFE_W_SOCIAL * ε_social。
    ARBITRATION_SOCIAL_TICKET_SCALE: float = 1.0

    def __init__(
        self,
        tokenizer: PrecisionTokenizer,
        *,
        learn_tokenizer_from_user: bool = True,
        lambda_explore_0: float = 0.55,
        social_risk_alpha: float = 0.5,
        word_imprint_capacity: int = 100,
        conflict_surprise_threshold: float = 0.10,
        conflict_surprise_gain: float = 0.45,
        conflict_min_pi_h: float = 0.10,
        conflict_min_uc: float = 0.42,
        recent_attention_capacity: int = 64,
        internal_tension_threshold: float = 0.12,
        internal_pending_increment: float = 0.09,
        internal_max_efe_to_speak: float = 500.0,
        internal_pending_decay_on_external: float = 0.88,
        internal_pending_decay_on_tick: float = 0.995,
        internal_global_scan_k: int = 8,
        internal_spontaneous_jitter: float = 0.015,
        internal_rng_seed: Optional[int] = None,
        train_fast_internal_surprise_hist_cap: Optional[int] = 24,
        preference_decay: float = 0.98,
        preference_learning_rate: float = 0.08,
        preference_value_max: float = 2.0,
        branch_bias_weight: float = 0.06,
        preference_beta_token: float = 0.03,
        preference_beta_association: float = 1.0,
        cognitive_debt_surprise_gamma: float = 0.5,
        curiosity_injection_pool_max: float = 1.0,
        curiosity_injection_per_event_cap: float = 0.42,
        curiosity_injection_pool_refill_external: float = 0.06,
        curiosity_injection_w_pending: float = 1.0,
        curiosity_injection_w_debt: float = 1.0,
        curiosity_injection_w_sigma: float = 1.0,
    ) -> None:
        self.tokenizer = tokenizer
        self._sigma: Dict[str, float] = {k: 0.5 for k in self.SLOT_NAMES}
        self._learn_tokenizer_from_user = learn_tokenizer_from_user
        self.lambda_explore_0 = float(lambda_explore_0)
        # 社会风险：对探索稿 pred 的乘性放大系数 α（与 ε_secondary 同量纲，非加性惩罚）
        self.social_risk_alpha = float(social_risk_alpha)
        self._word_memory = WordStateMemory(
            capacity=int(word_imprint_capacity),
            cognitive_debt_surprise_gamma=float(cognitive_debt_surprise_gamma),
        )
        self.conflict_surprise_threshold = float(conflict_surprise_threshold)
        self.conflict_surprise_gain = float(conflict_surprise_gain)
        # 保留 CLI/序列化兼容；EFE 框架下不再用作硬门槛。
        self.conflict_min_pi_h = float(conflict_min_pi_h)
        self.conflict_min_uc = float(conflict_min_uc)
        cap = max(1, int(recent_attention_capacity))
        self._recent_attention: Deque[str] = deque(maxlen=cap)
        self.internal_tension_threshold = float(internal_tension_threshold)
        self.internal_pending_increment = float(internal_pending_increment)
        self.internal_max_efe_to_speak = float(internal_max_efe_to_speak)
        self.internal_pending_decay_on_external = float(internal_pending_decay_on_external)
        self.internal_pending_decay_on_tick = float(internal_pending_decay_on_tick)
        self.internal_global_scan_k = max(0, int(internal_global_scan_k))
        self.internal_spontaneous_jitter = max(0.0, float(internal_spontaneous_jitter))
        self._internal_rng = random.Random(internal_rng_seed)
        self._internal_pending: Dict[str, float] = {}
        wp = max(0.0, float(curiosity_injection_w_pending))
        wd = max(0.0, float(curiosity_injection_w_debt))
        ws = max(0.0, float(curiosity_injection_w_sigma))
        wsum = wp + wd + ws
        self._curiosity_inj_w: Tuple[float, float, float] = (
            (wp / wsum, wd / wsum, ws / wsum)
            if wsum > 0.0
            else (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
        )
        self.curiosity_injection_pool_max = max(1e-6, float(curiosity_injection_pool_max))
        self.curiosity_injection_per_event_cap = max(0.0, float(curiosity_injection_per_event_cap))
        self.curiosity_injection_pool_refill_external = max(
            0.0, float(curiosity_injection_pool_refill_external)
        )
        self._curiosity_injection_pool = float(self.curiosity_injection_pool_max)
        sf_hist_cap = train_fast_internal_surprise_hist_cap
        if sf_hist_cap is None:
            self.train_fast_internal_surprise_hist_cap = None
        else:
            ci = int(sf_hist_cap)
            self.train_fast_internal_surprise_hist_cap = (
                None if ci <= 0 else max(1, ci)
            )
        self.preference_decay = max(0.0, min(1.0, float(preference_decay)))
        self.preference_learning_rate = max(0.0, float(preference_learning_rate))
        self.preference_value_max = max(1e-6, float(preference_value_max))
        self.branch_bias_weight = max(0.0, float(branch_bias_weight))
        self.preference_beta_token = max(0.0, float(preference_beta_token))
        self.preference_beta_association = max(0.0, float(preference_beta_association))
        self._branch_bias: Dict[str, float] = {}
        self._token_value: Dict[str, float] = {}
        self._association_value: Dict[str, Dict[str, float]] = {}

    def set_compute_device(self, device: str) -> str:
        """设置词印记相似度计算设备；当前主要影响 internal_tick 的全库扫描。"""
        return self._word_memory.set_compute_device(device)

    def sigma_state(self) -> Dict[str, float]:
        """公开当前 sigma 快照，供训练摘要与诊断使用。"""
        return {k: float(v) for k, v in self._sigma.items()}

    @property
    def recent_attention_words(self) -> List[str]:
        """最近注意过的 surface token（环形缓冲，旧项自动丢弃）。供 `internal_scan` 等内部时间线使用。"""
        return list(self._recent_attention)

    def _record_recent_attention(self, toks_in: List[str], focus_tok: str) -> None:
        """将本轮用户句分词与印记焦点先后写入缓冲；焦点与句末 token 重复时只记一次句内顺序。"""
        for t in toks_in:
            s = str(t).strip()
            if s:
                self._recent_attention.append(s)
        ft = str(focus_tok or "").strip()
        last = str(toks_in[-1]).strip() if toks_in else ""
        if ft and ft != last:
            self._recent_attention.append(ft)

    def dialogue_model_to_dict(self) -> Dict[str, object]:
        """对话侧可持久化状态（槽位 σ；在线分词学习开关由启动参数决定，不写入本文件）。"""
        return {
            "format": DIALOGUE_MODEL_FORMAT,
            "format_version": int(DIALOGUE_MODEL_FORMAT_VERSION),
            "sigma": {k: float(self._sigma[k]) for k in self.SLOT_NAMES},
            "word_imprints": self._word_memory.to_dict(),
            "preference_state": self._preference_state_to_dict(),
        }

    def _sigma_dict_from_dialogue_payload(self, sig: Dict[str, object]) -> Dict[str, float]:
        """从对话 JSON 的 ``sigma`` 解析四槽；缺任一则抛错（支持旧版中文键名）。"""
        legacy = self.SIGMA_LEGACY_KEYS
        out: Dict[str, float] = {}
        for i, k in enumerate(self.SLOT_NAMES):
            if k in sig:
                out[k] = float(cast(object, sig[k]))
            elif i < len(legacy) and legacy[i] in sig:
                out[k] = float(cast(object, sig[legacy[i]]))
            else:
                raise ValueError(
                    f"sigma 不完整：缺少槽位 {k!r}（或旧版键 {legacy[i]!r}），严格快照拒绝加载"
                )
        return out

    def _reset_preference_state_to_defaults(self) -> None:
        """``preference_state`` 省略时：显式空偏好 + 好奇心注入池回满。"""
        self._branch_bias = {}
        self._token_value = {}
        self._association_value = {}
        self._curiosity_injection_pool = float(self.curiosity_injection_pool_max)

    def _apply_preference_state_payload(self, raw: Dict[str, object]) -> None:
        """已校验 ``format == preference_state_v1``；整表覆盖写入。"""
        bb = raw.get("branch_bias")
        self._branch_bias = (
            {str(k): float(cast(object, v)) for k, v in bb.items()}
            if isinstance(bb, dict)
            else {}
        )
        tv = raw.get("token_value")
        self._token_value = (
            {str(k): float(cast(object, v)) for k, v in tv.items()}
            if isinstance(tv, dict)
            else {}
        )
        av = raw.get("association_value")
        if isinstance(av, dict):
            merged: Dict[str, Dict[str, float]] = {}
            for tr, inner in av.items():
                if not isinstance(inner, dict):
                    continue
                merged[str(tr)] = {
                    str(a): float(cast(object, val)) for a, val in inner.items()
                }
            self._association_value = merged
        else:
            self._association_value = {}
        cip = raw.get("curiosity_injection_pool")
        if isinstance(cip, (int, float)):
            self._curiosity_injection_pool = max(
                0.0,
                min(self.curiosity_injection_pool_max, float(cip)),
            )
        else:
            self._curiosity_injection_pool = float(self.curiosity_injection_pool_max)

    def _merge_preference_state_patch(self, raw: Dict[str, object]) -> None:
        """``patch_dialogue_model_dict``：仅合并出现的子键。"""
        bb = raw.get("branch_bias")
        if isinstance(bb, dict):
            for k, v in bb.items():
                self._branch_bias[str(k)] = float(cast(object, v))
        tv = raw.get("token_value")
        if isinstance(tv, dict):
            for k, v in tv.items():
                self._token_value[str(k)] = float(cast(object, v))
        av = raw.get("association_value")
        if isinstance(av, dict):
            for tr, inner in av.items():
                if not isinstance(inner, dict):
                    continue
                inner_m = self._association_value.setdefault(str(tr), {})
                for a, val in inner.items():
                    inner_m[str(a)] = float(cast(object, val))
        if "curiosity_injection_pool" in raw:
            cip = raw.get("curiosity_injection_pool")
            if isinstance(cip, (int, float)):
                self._curiosity_injection_pool = max(
                    0.0,
                    min(self.curiosity_injection_pool_max, float(cip)),
                )

    def apply_dialogue_model_dict(self, data: Dict[str, object]) -> None:
        """
        严格快照：加载后状态**完全**来自 ``data``，不保留内存中旧 σ / 印记 / 偏好。

        - ``sigma``、``word_imprints`` 必须存在且可解析；缺一则 ``ValueError``。
        - ``preference_state`` 可省略 → 默认空偏好；若出现则 ``format`` 须合法，否则 ``ValueError``。
        """
        fmt = data.get("format")
        if fmt != DIALOGUE_MODEL_FORMAT:
            raise ValueError(f"不支持的对话模型 format: {fmt!r}")
        sig = data.get("sigma")
        if not isinstance(sig, dict):
            raise ValueError("对话模型缺少 sigma（严格快照：必须可完整还原槽位）")
        new_sigma = self._sigma_dict_from_dialogue_payload(sig)

        wi = data.get("word_imprints")
        if not isinstance(wi, dict) or wi.get("format") != WORD_STATE_MEMORY_FORMAT:
            raise ValueError("对话模型缺少或格式错误的 word_imprints")
        try:
            new_memory = WordStateMemory.from_dict(cast(Dict[str, object], wi))
        except Exception as exc:
            raise ValueError(f"word_imprints 解析失败: {exc}") from exc

        ps = data.get("preference_state")
        if ps is None:
            pref_mode = "defaults"
            pref_payload: Optional[Dict[str, object]] = None
        elif isinstance(ps, dict):
            if str(ps.get("format", "")) != PREFERENCE_STATE_FORMAT:
                raise ValueError(
                    f"preference_state 格式无效: {ps.get('format')!r}（严格快照拒绝加载）"
                )
            pref_mode = "payload"
            pref_payload = ps
        else:
            raise ValueError("preference_state 必须为对象或省略（省略时使用默认空偏好）")

        self._sigma = new_sigma
        self._word_memory = new_memory
        if pref_mode == "defaults":
            self._reset_preference_state_to_defaults()
        else:
            assert pref_payload is not None
            self._apply_preference_state_payload(pref_payload)
        # 兼容旧文件中的 learn_tokenizer_from_user（已废弃，忽略）

    def patch_dialogue_model_dict(self, data: Dict[str, object]) -> None:
        """
        增量补丁：只更新 ``data`` 中出现的顶层键，其余保持当前内存。

        - 若含 ``format``，须为 ``cognitive_dialogue_agent_v1``。
        - ``sigma``：仅覆盖对象中出现的槽位（``sigma_slot_*`` 或旧版中文键）。
        - ``word_imprints``：若提供，须为完整合法印记快照，**整体替换** ``WordStateMemory``。
        - ``preference_state``：若提供，须为合法 ``preference_state_v1``，子 dict **合并**写入。
        """
        if "format" in data and data.get("format") != DIALOGUE_MODEL_FORMAT:
            raise ValueError(f"不支持的对话模型 format: {data.get('format')!r}")
        if "sigma" in data:
            sig = data["sigma"]
            if not isinstance(sig, dict):
                raise ValueError("patch 中 sigma 必须为对象")
            legacy = self.SIGMA_LEGACY_KEYS
            for i, k in enumerate(self.SLOT_NAMES):
                if k in sig:
                    self._sigma[k] = float(cast(object, sig[k]))
                elif i < len(legacy) and legacy[i] in sig:
                    self._sigma[k] = float(cast(object, sig[legacy[i]]))
        if "word_imprints" in data:
            wi = data["word_imprints"]
            if not isinstance(wi, dict) or wi.get("format") != WORD_STATE_MEMORY_FORMAT:
                raise ValueError("patch 中 word_imprints 缺失或格式错误")
            self._word_memory = WordStateMemory.from_dict(cast(Dict[str, object], wi))
        if "preference_state" in data:
            ps = data["preference_state"]
            if ps is None:
                self._reset_preference_state_to_defaults()
            elif isinstance(ps, dict):
                if str(ps.get("format", "")) != PREFERENCE_STATE_FORMAT:
                    raise ValueError(
                        f"preference_state 格式无效: {ps.get('format')!r}（patch 拒绝应用）"
                    )
                self._merge_preference_state_patch(ps)
            else:
                raise ValueError("preference_state 必须为对象或 null")

    def _preference_state_to_dict(self) -> Dict[str, object]:
        assoc_out: Dict[str, Dict[str, float]] = {}
        for tr, inner in self._association_value.items():
            assoc_out[str(tr)] = {str(a): float(v) for a, v in inner.items()}
        return {
            "format": PREFERENCE_STATE_FORMAT,
            "branch_bias": {str(k): float(v) for k, v in self._branch_bias.items()},
            "token_value": {str(k): float(v) for k, v in self._token_value.items()},
            "association_value": assoc_out,
            "curiosity_injection_pool": float(self._curiosity_injection_pool),
        }

    def save_dialogue_model(
        self,
        path: str,
        *,
        compact: bool = False,
        compression: Optional[Literal["gzip", "zstd"]] = None,
    ) -> None:
        """写入对话 JSON。compact=True 时不缩进；路径以 ``.gz`` / ``.zst`` 结尾时自动压缩。"""
        comp = compression if compression is not None else infer_compression_from_path(path)
        write_json_document(
            path,
            self.dialogue_model_to_dict(),
            compact=compact,
            compression=comp,
        )

    def load_dialogue_model(self, path: str) -> None:
        raw = read_json_document(path)
        if not isinstance(raw, dict):
            raise ValueError("对话模型 JSON 顶层必须是对象")
        self.apply_dialogue_model_dict(cast(Dict[str, object], raw))

    @classmethod
    def from_tokenizer_path(
        cls,
        path: str,
        *,
        learn_tokenizer_from_user: bool = True,
        dialogue_model_path: str = "",
        lambda_explore_0: float = 0.55,
        social_risk_alpha: float = 0.5,
        word_imprint_capacity: int = 100,
        conflict_surprise_threshold: float = 0.10,
        conflict_surprise_gain: float = 0.45,
        conflict_min_pi_h: float = 0.10,
        conflict_min_uc: float = 0.42,
        recent_attention_capacity: int = 64,
        internal_tension_threshold: float = 0.12,
        internal_pending_increment: float = 0.09,
        internal_max_efe_to_speak: float = 500.0,
        internal_pending_decay_on_external: float = 0.88,
        internal_global_scan_k: int = 8,
        internal_spontaneous_jitter: float = 0.015,
        internal_rng_seed: Optional[int] = None,
        train_fast_internal_surprise_hist_cap: Optional[int] = 24,
        preference_decay: float = 0.98,
        preference_learning_rate: float = 0.08,
        preference_value_max: float = 2.0,
        branch_bias_weight: float = 0.06,
        preference_beta_token: float = 0.03,
        preference_beta_association: float = 1.0,
    ) -> "CognitiveDialogueAgent":
        agent = cls(
            PrecisionTokenizer.load_model(path),
            learn_tokenizer_from_user=learn_tokenizer_from_user,
            lambda_explore_0=lambda_explore_0,
            social_risk_alpha=social_risk_alpha,
            word_imprint_capacity=word_imprint_capacity,
            conflict_surprise_threshold=conflict_surprise_threshold,
            conflict_surprise_gain=conflict_surprise_gain,
            conflict_min_pi_h=conflict_min_pi_h,
            conflict_min_uc=conflict_min_uc,
            recent_attention_capacity=recent_attention_capacity,
            internal_tension_threshold=internal_tension_threshold,
            internal_pending_increment=internal_pending_increment,
            internal_max_efe_to_speak=internal_max_efe_to_speak,
            internal_pending_decay_on_external=internal_pending_decay_on_external,
            internal_global_scan_k=internal_global_scan_k,
            internal_spontaneous_jitter=internal_spontaneous_jitter,
            internal_rng_seed=internal_rng_seed,
            train_fast_internal_surprise_hist_cap=train_fast_internal_surprise_hist_cap,
            preference_decay=preference_decay,
            preference_learning_rate=preference_learning_rate,
            preference_value_max=preference_value_max,
            branch_bias_weight=branch_bias_weight,
            preference_beta_token=preference_beta_token,
            preference_beta_association=preference_beta_association,
        )
        if dialogue_model_path:
            agent.load_dialogue_model(dialogue_model_path)
        return agent

    @staticmethod
    def _mean(xs: List[float]) -> float:
        if not xs:
            return 0.0
        return sum(xs) / len(xs)

    def pi_statement(self) -> float:
        """π_statement（原始尺度）：(R/R_max)·(1−m)，供日志；诚实度前缀由 EFE 在候选池中择优。"""
        r, m, r_max, _f = self._freeze_resource()
        return max(0.0, min(1.0, (r / r_max) * (1.0 - m)))

    def u_curiosity(self) -> float:
        """u^{curiosity}：对内部最不确定概念的张力（主动探询估值，取 max σ）。"""
        return max(self._sigma.values()) if self._sigma else 0.0

    def u_task(self, user_text: str, epsilon_social: float) -> float:
        """u^{task}：外部提问/强社会扰动带来的任务需求。"""
        q = 1.0 if ("?" in user_text or "？" in user_text) else 0.0
        surge = min(1.0, max(0.0, epsilon_social / 2.2))
        return max(q, surge)

    def _freeze_resource(self) -> Tuple[float, float, float, float]:
        return (
            float(self.tokenizer.R),
            float(self.tokenizer.m),
            float(self.tokenizer.R_max),
            float(self.tokenizer.F_ema),
        )

    def _restore_resource(self, snap: Tuple[float, float, float, float]) -> None:
        self.tokenizer.R, self.tokenizer.m, self.tokenizer.R_max, self.tokenizer.F_ema = snap

    def epsilon_secondary_on_draft(self, draft: str) -> float:
        """次级预测误差：若我这样说，内部语言模型会有多吃惊？（模拟不改动冻结前的外交资源状态）。"""
        if not draft.strip():
            return 0.0
        snap = self._freeze_resource()
        out = float(self.tokenizer.mean_surprise(draft))
        self._restore_resource(snap)
        return out

    def _listen_trace(self, user_text: str) -> Tuple[float, List[str]]:
        """
        倾听一次并返回：
        - 社会惊奇均值
        - 本轮 trace 得到的 tokens

        这样 `produce_turn` / `train_step` 不需要再为同一句话重复 tokenize 一次。
        """
        trace = self.tokenizer._trace(  # noqa: SLF001
            user_text,
            collect_precision=False,
            collect_surprise=False,
            collect_boundary_indices=False,
            collect_resource=False,
            collect_mind=False,
        )
        eps = float(trace.mean_surprise)
        for k in self._sigma:
            self._sigma[k] = min(1.0, self._sigma[k] + 0.05 * eps)
        if self._learn_tokenizer_from_user:
            self.tokenizer.ingest_interaction_line(user_text)
        return eps, trace.tokens

    def hear(self, user_text: str) -> float:
        """倾听：社会预测误差推高各 σ 槽位不确定度（对各槽位等量耦合惊奇，非按词表分话题）。"""
        eps, _ = self._listen_trace(user_text)
        return eps

    def _germinate_intent(self, user_text: str, eps_social: float) -> CommunicativeIntent:
        uc = self.u_curiosity()
        ut = self.u_task(user_text, eps_social)
        tension = max(self._sigma, key=self._sigma.get)
        anchor = min(self._sigma, key=self._sigma.get)
        if ut >= uc and ut > 0.35:
            act: SpeechAct = "respond"
        elif uc > 0.55:
            act = "explore"
        else:
            act = "clarify"
        return CommunicativeIntent(
            speech_act=act,
            anchor_slot=anchor,
            tension_slot=tension,
            u_curiosity=uc,
            u_task=ut,
        )

    def _germinate_internal_intent(self) -> CommunicativeIntent:
        """内心独白意图：不把候选 trigger 当作外部任务刺激（``u_task`` 恒由空输入 + ``eps=0`` 决定）。"""
        return self._germinate_intent("", 0.0)

    def _internal_monologue_imprint_snap(
        self, base_resource_snap: Tuple[float, float, float, float]
    ) -> WordStateImprint:
        """内心路径统一印记快照：资源取自当前冻结态，``u_curiosity`` 取全局 max(σ)，``u_task`` 强制为 0。"""
        return WordStateImprint(
            F_ema=float(base_resource_snap[3]),
            R=float(base_resource_snap[0]),
            m=float(base_resource_snap[1]),
            u_curiosity=float(self.u_curiosity()),
            u_task=0.0,
        )

    def _sparse_plan(self, intent: CommunicativeIntent) -> List[Tuple[str, str]]:
        """内部稀疏语义计划（角色，槽位）。"""
        if intent.speech_act == "explore":
            return [("探询", intent.tension_slot), ("锚定", intent.anchor_slot)]
        if intent.speech_act == "respond":
            return [("消解扰动", "社会输入"), ("锚定", intent.anchor_slot), ("张紧", intent.tension_slot)]
        return [("澄清", "社会输入"), ("锚定", intent.anchor_slot)]

    def _resource_factor(self) -> float:
        """
        对话输出里用于 λ_explore、探索惩罚与 hedge 的资源因子，裁剪到 [0,1]。

        语料训练后 R_max 常被 F_ema 顶到 ≫1，而对话 trace 只耗掉一部分 R，此时纯 R/R_max
        会接近 0，导致 λ_explore≈0 且探索惩罚拉满，回声稿在打分上永远输给保守稿。
        分母改用 max(α·R_max, R_floor)，使「中等绝对 R」仍对应中等冒险许可。
        """
        r, m, r_max, _f = self._freeze_resource()
        alpha = 0.15
        r_floor = 0.22
        denom = max(r_max * alpha, r_floor, 1e-6)
        return max(0.0, min(1.0, (r / denom) * max(0.0, min(1.0, 1.0 - m))))

    def _lambda_explore(self, rf: float | None = None) -> float:
        if rf is None:
            rf = self._resource_factor()
        return self.lambda_explore_0 * rf

    def _effective_pred_risk(self, draft: str, *, is_explore: bool) -> float:
        """
        保守稿：pred = ε_secondary。
        探索稿：pred = ε_secondary · (1 + α·(1−rf))；社会风险放大生成成本而非与之相加，避免量纲错配。
        rf 高时因子→1，回声与保守在 −pred 项上可比；rf 低时乘性放大、压低回声。
        """
        base = float(self.epsilon_secondary_on_draft(draft))
        return self._effective_pred_risk_from_eps(base, is_explore=is_explore)

    def _effective_pred_risk_from_eps(
        self, eps_sec: float, *, is_explore: bool, rf: float | None = None
    ) -> float:
        base = float(eps_sec)
        if not is_explore:
            return base
        if rf is None:
            rf = self._resource_factor()
        social_risk_factor = 1.0 + self.social_risk_alpha * (1.0 - rf)
        return base * social_risk_factor

    @staticmethod
    def _draft_starts_with_unknown_hedge(draft: str) -> bool:
        s = str(draft).lstrip()
        return s.startswith("我不知道")

    def social_arbitration_ticket(self, eps_social: float) -> float:
        """外部回应相对内心独白仲裁时扣除的社会负担（ε_social 越高越贵，内心相对更易先说）。"""
        eps = max(0.0, float(eps_social))
        return float(self.EFE_W_SOCIAL) * float(self.ARBITRATION_SOCIAL_TICKET_SCALE) * eps

    def arbitration_external_net_cost(self, turn: "DialogueTurn") -> float:
        """外部净认知成本（用于与内心 best_efe 对标）。"""
        return float(turn.best_efe) - self.social_arbitration_ticket(float(turn.epsilon_social_in))

    def arbitration_external_first(self, mono: "InternalMonologue", turn: "DialogueTurn") -> bool:
        """
        True：外部先说；False：内心先说。
        比较外部净成本与内心 EFE（4a）；平局时外部优先。
        """
        g_ext_net = self.arbitration_external_net_cost(turn)
        g_int = float(mono.best_efe)
        return g_ext_net <= g_int

    def _expected_free_energy(
        self,
        draft: str,
        *,
        eps_social: float,
        u_curiosity: float,
        is_explore: bool,
        imprint_surprise: float,
        internal_monologue: bool = False,
    ) -> float:
        """
        标量预期自由能 G(y)：最小化即输出动作。

        含：(i) 风险项 pred_eff（探索时乘社会风险），(ii) 与社会惊奇对齐的超出部分，
        (iii) 长度×疲劳的动作代价，减去 (iv) 好奇驱动的探索奖励与 (v) 印记失配的消解倾向。

        ``internal_monologue=True`` 时不计 (ii)（内心独白不对用户句负责，设计 4a）。
        """
        rf = self._resource_factor()
        eps_sec = float(self.epsilon_secondary_on_draft(draft))
        pred = self._effective_pred_risk_from_eps(
            eps_sec, is_explore=is_explore, rf=rf
        )
        soc = max(1e-6, float(eps_social))
        rel_excess = (
            0.0
            if internal_monologue
            else max(0.0, eps_sec - self.EFE_SOCIAL_ALIGN_KAPPA * soc)
        )
        st = self.tokenizer.resource_state()
        m = float(st.get("m", 0.0))
        r = float(st.get("R", 0.0))
        r_max = max(float(st.get("R_max", 1.0)), 1e-6)
        fatigue = m + 0.15 * (1.0 - r / r_max)
        len_cost = self.EFE_W_LEN * max(1, len(draft)) * (0.5 + fatigue)
        explore_bonus = self._lambda_explore(rf) * float(u_curiosity) * (
            1.0 if is_explore else 0.0
        )
        imprint_bonus = self.conflict_surprise_gain * min(2.0, float(imprint_surprise))
        explore_commit = (
            self.EFE_EXPLORATION_COMMIT * (1.0 - rf) if is_explore else 0.0
        )
        unknown_hedge = (
            float(self.EFE_W_UNKNOWN_HEDGE)
            if self._draft_starts_with_unknown_hedge(draft)
            else 0.0
        )
        return (
            pred
            + self.EFE_W_SOCIAL * rel_excess
            + len_cost
            + explore_commit
            + unknown_hedge
            - explore_bonus
            - imprint_bonus
        )

    def _preference_clamped_ema(self, old: float, reward: float) -> float:
        d = self.preference_decay
        lr = self.preference_learning_rate
        vmax = self.preference_value_max
        new = d * old + lr * reward
        return max(-vmax, min(vmax, new))

    def _refill_curiosity_injection_pool_external(self) -> None:
        """外部回合后略回补共享注入预算（与 ``_decay_internal_pending`` 同步调用）。"""
        self._curiosity_injection_pool = min(
            self.curiosity_injection_pool_max,
            self._curiosity_injection_pool + self.curiosity_injection_pool_refill_external,
        )

    def _inject_curiosity_on_conservative_penalty(
        self, turn: DialogueTurn, abs_reward: float
    ) -> None:
        """保守分支负反馈：从共享池取量，按比例写入 pending、认知债务与 tension_slot σ。"""
        pool = float(self._curiosity_injection_pool)
        total_raw = self.curiosity_injection_per_event_cap * float(abs_reward)
        total = min(pool, total_raw)
        if total <= 1e-12:
            return
        wp, wd, ws = self._curiosity_inj_w
        p_amt = total * wp
        d_amt = total * wd
        s_amt = total * ws
        w = (turn.conflict_focus_token or "").strip()
        slot = str(turn.intent.tension_slot or "").strip()
        if w:
            p = float(self._internal_pending.get(w, 0.0))
            self._internal_pending[w] = min(1.0, p + p_amt * max(0.0, 1.0 - p))
            self._word_memory.inject_cognitive_debt(w, d_amt)
        else:
            s_amt += p_amt + d_amt
        if slot in self._sigma and s_amt > 0:
            sig = float(self._sigma[slot])
            self._sigma[slot] = min(1.0, sig + s_amt * max(0.0, 1.0 - sig))
        elif s_amt > 0 and w:
            p2 = float(self._internal_pending.get(w, 0.0))
            self._internal_pending[w] = min(1.0, p2 + s_amt * max(0.0, 1.0 - p2))
        self._curiosity_injection_pool = max(0.0, pool - total)

    def apply_dialogue_feedback(self, reward: float, turn: DialogueTurn) -> None:
        """方案 B：根据交互反馈更新分支偏置、焦点 token 价值与联想对价值（EMA + 裁剪）。"""
        r = max(-1.0, min(1.0, float(reward)))
        br = turn.output_branch
        self._branch_bias[br] = self._preference_clamped_ema(
            self._branch_bias.get(br, 0.0), r
        )
        ft = (turn.conflict_focus_token or "").strip()
        if ft:
            self._token_value[ft] = self._preference_clamped_ema(
                self._token_value.get(ft, 0.0), r
            )
        if (
            turn.output_branch == "associative_probe"
            and turn.association_trigger.strip()
            and turn.association_pick.strip()
        ):
            tr = turn.association_trigger.strip()
            pk = turn.association_pick.strip()
            inner = self._association_value.setdefault(tr, {})
            inner[pk] = self._preference_clamped_ema(inner.get(pk, 0.0), r)
        if r < 0.0 and br == "conservative":
            self._inject_curiosity_on_conservative_penalty(turn, abs(r))

    def apply_internal_monologue_feedback(
        self, reward: float, mono: InternalMonologue
    ) -> None:
        """对 ``internal_tick`` 产出独白反馈（念头 trigger + 可选联想归因）。"""
        r = max(-1.0, min(1.0, float(reward)))
        br = mono.output_branch
        self._branch_bias[br] = self._preference_clamped_ema(
            self._branch_bias.get(br, 0.0), r
        )
        tt = (mono.trigger_token or "").strip()
        if tt:
            self._token_value[tt] = self._preference_clamped_ema(
                self._token_value.get(tt, 0.0), r
            )
        if (
            mono.output_branch == "associative_probe"
            and mono.association_trigger.strip()
            and mono.association_pick.strip()
        ):
            tr = mono.association_trigger.strip()
            pk = mono.association_pick.strip()
            inner = self._association_value.setdefault(tr, {})
            inner[pk] = self._preference_clamped_ema(inner.get(pk, 0.0), r)

    def _biased_associated_tokens(self, trigger: str, *, k: int) -> List[Tuple[str, int]]:
        t = trigger.strip()
        if not t or k <= 0:
            return []
        pool = max(k * 12, 48)
        raw = self._word_memory.associated_tokens(t, k=pool)
        if not raw:
            return []
        beta = self.preference_beta_association

        def sort_key(it: Tuple[str, int]) -> Tuple[float, int, str]:
            assoc, cnt = it
            boost = self._association_value.get(t, {}).get(assoc, 0.0)
            score = float(cnt) + beta * float(boost)
            return (-score, -cnt, assoc)

        ranked = sorted(raw, key=sort_key)
        return ranked[:k]

    def _efe_best_reply(
        self,
        *,
        intent: CommunicativeIntent,
        plan: List[Tuple[str, str]],
        pi_s: float,
        user_text: str,
        eps_in: float,
        listen_resource_snap: Tuple[float, float, float, float],
        max_surp: float,
        focus_tok: str,
        internal_monologue: bool,
        listen_tokens: List[str] | None = None,
    ) -> Tuple[str, float, OutputBranch, int, float, str, str]:
        """与 ``produce_turn`` / ``internal_tick`` 共用的 EFE 择优（倾听快照已冻结）。"""
        uc = float(intent.u_curiosity)
        rf = self._resource_factor()
        max_depth = max(0, min(4, int(round(4.0 * rf))))
        if listen_tokens is None:
            listen_tokens = (
                self.tokenizer.tokenize(user_text) if str(user_text).strip() else []
            )

        draft = ""
        eps_sec = 0.0
        branch: OutputBranch = "conservative"
        best_g_eff = float("inf")
        best_efe_raw = float("inf")
        replans = 0
        assoc_trigger_out = ""
        assoc_pick_out = ""

        honesty_prefixes = ("", "可能", "我不知道，")

        for depth in range(max_depth + 1):
            plan_d = self._plan_at_depth(plan, depth)
            self._restore_resource(listen_resource_snap)
            candidates: List[Tuple[str, OutputBranch, bool, float, str]] = []
            for hx in honesty_prefixes:
                candidates.append(
                    (
                        self._realize(
                            intent,
                            plan_d,
                            pi_s,
                            eps_in,
                            user_text,
                            depth,
                            hedge_prefix=hx,
                            listen_tokens=listen_tokens,
                        ),
                        "conservative",
                        False,
                        0.0,
                        "",
                    )
                )
                candidates.append(
                    (
                        self._build_explore_echo_draft(
                            intent,
                            plan_d,
                            pi_s,
                            eps_in,
                            user_text,
                            depth,
                            hedge_prefix=hx,
                            listen_tokens=listen_tokens,
                        ),
                        "explore_echo",
                        True,
                        0.0,
                        "",
                    )
                )
            if focus_tok and max_surp >= self.conflict_surprise_threshold:
                for hx in honesty_prefixes:
                    candidates.append(
                        (
                            self._build_conflict_probe_draft(
                                focus_tok,
                                intent,
                                pi_s,
                                user_text,
                                depth,
                                hedge_prefix=hx,
                            ),
                            "conflict_probe",
                            False,
                            float(max_surp),
                            "",
                        )
                    )

            trigger_key = (focus_tok or user_text).strip()
            associates = self._biased_associated_tokens(trigger_key, k=5)
            if trigger_key and associates:
                for hx in honesty_prefixes:
                    for assoc_draft, assoc_pick in self._build_associative_probe_drafts(
                        trigger_key,
                        associates,
                        depth,
                        hedge_prefix=hx,
                    ):
                        candidates.append(
                            (
                                assoc_draft,
                                "associative_probe",
                                True,
                                float(max_surp),
                                assoc_pick,
                            )
                        )

            for text, br, is_explore, impr_s, assoc_pick in candidates:
                g = self._expected_free_energy(
                    text,
                    eps_social=float(eps_in),
                    u_curiosity=uc,
                    is_explore=is_explore,
                    imprint_surprise=float(impr_s),
                    internal_monologue=internal_monologue,
                )
                bias = self._branch_bias.get(br, 0.0)
                g_eff = g - self.branch_bias_weight * bias
                if g_eff < best_g_eff:
                    best_g_eff = g_eff
                    best_efe_raw = g
                    draft = text
                    branch = br
                    eps_sec = float(self.epsilon_secondary_on_draft(text))
                    replans = depth
                    if br == "associative_probe" and assoc_pick:
                        assoc_trigger_out = trigger_key.strip()
                        assoc_pick_out = assoc_pick.strip()
                    else:
                        assoc_trigger_out = ""
                        assoc_pick_out = ""

        self._restore_resource(listen_resource_snap)
        return (
            draft,
            eps_sec,
            branch,
            replans,
            best_efe_raw,
            assoc_trigger_out,
            assoc_pick_out,
        )

    def _decay_internal_pending(self) -> None:
        """外部回合后积压张力略衰，避免永久爆表。"""
        if self._internal_pending:
            fac = self.internal_pending_decay_on_external
            self._internal_pending = {
                k: v * fac
                for k, v in self._internal_pending.items()
                if v * fac > 1e-5
            }
        self._refill_curiosity_injection_pool_external()

    def _decay_internal_pending_tick(self) -> None:
        """内部 tick 间的极弱衰减，避免长预热时 pending 只增不减；认知债务同频衰减。"""
        self._word_memory.decay_cognitive_debt_tick(self.internal_pending_decay_on_tick)
        if not self._internal_pending:
            return
        fac = self.internal_pending_decay_on_tick
        self._internal_pending = {
            k: v * fac
            for k, v in self._internal_pending.items()
            if v * fac > 1e-5
        }

    def _internal_frontier_candidates(
        self,
        base_resource_snap: Tuple[float, float, float, float],
        *,
        cold_budget: Optional[int] = None,
        surprise_hist_scan_cap: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        imprint_snap = self._internal_monologue_imprint_snap(base_resource_snap)
        k_scan = max(1, int(self.internal_global_scan_k))
        pending_tokens = {k for k, v in self._internal_pending.items() if v > 1e-5}
        eff_cold = (
            max(1, min(2, k_scan))
            if cold_budget is None
            else max(0, int(cold_budget))
        )
        frontier_tokens = self._word_memory.activation_frontier_tokens(
            recent_tokens=list(reversed(self._recent_attention)),  # noqa: SLF001
            pending_tokens=pending_tokens,
            limit=max(4, k_scan * 4),
            one_hop_budget=max(4, k_scan * 2),
            two_hop_budget=max(2, k_scan),
            two_hop_per_token=max(1, k_scan // 2),
            cold_budget=eff_cold,
            shard_count=16,
            max_active_tokens=max(64, k_scan * 16),
        )
        if not frontier_tokens:
            return []
        ordered: List[Tuple[str, float]] = []
        for tok in frontier_tokens:
            surp = float(
                self._word_memory.semantic_surprise_for_token(
                    tok,
                    imprint_snap,
                    hist_scan_limit=surprise_hist_scan_cap,
                )
            )
            if self.internal_spontaneous_jitter > 0.0:
                surp = max(
                    0.0,
                    surp
                    + self._internal_rng.uniform(
                        -self.internal_spontaneous_jitter,
                        self.internal_spontaneous_jitter,
                    ),
                )
            ordered.append((tok, surp))
        ordered.sort(key=lambda x: (-x[1], x[0]))
        return ordered

    def internal_tick(self) -> Optional[InternalMonologue]:
        """
        内心时间线一拍：扫描 ``recent_attention_words``（新→旧去重）与全库 top-K 失配词，
        张力 = 印记惊奇×(0.5+u_c)×(0.3+rf)+pending[token]。超过阈值则用与外部相同的候选构造 + EFE（4a：无社会对齐项）。
        不写印记、不 hear；胜出则清空该 token 的 pending。
        """
        self._decay_internal_pending_tick()
        base_resource_snap = self._freeze_resource()
        uc = float(self.u_curiosity())
        rf = max(
            0.0,
            min(
                1.0,
                (base_resource_snap[0] / max(base_resource_snap[2] * 0.15, 0.22, 1e-6))
                * max(0.0, min(1.0, 1.0 - base_resource_snap[1])),
            ),
        )
        ordered = self._internal_frontier_candidates(base_resource_snap)
        if not ordered:
            return None

        internal_intent = self._germinate_internal_intent()
        best_mono: Optional[InternalMonologue] = None
        best_efe_global = float("inf")
        pi_s = max(
            0.0,
            min(
                1.0,
                (base_resource_snap[0] / max(base_resource_snap[2], 1e-6))
                * (1.0 - base_resource_snap[1]),
            ),
        )

        for tok, precomputed_surprise in ordered:
            eps_fake = 0.0
            surp = float(precomputed_surprise)
            base_tension = surp * (0.5 + uc) * (0.3 + rf)
            pend = float(self._internal_pending.get(tok, 0.0))
            tv = self.preference_beta_token * self._token_value.get(str(tok).strip(), 0.0)
            tension = base_tension + pend + tv
            if tension < self.internal_tension_threshold:
                continue

            plan = self._sparse_plan(internal_intent)
            max_surp = surp
            focus_tok = tok
            draft, eps_sec, branch, _replans, best_efe, a_tr, a_pk = self._efe_best_reply(
                intent=internal_intent,
                plan=plan,
                pi_s=pi_s,
                user_text=tok,
                eps_in=eps_fake,
                listen_resource_snap=base_resource_snap,
                max_surp=max_surp,
                focus_tok=focus_tok or "",
                internal_monologue=True,
                listen_tokens=[tok],
            )

            if best_efe > self.internal_max_efe_to_speak:
                self._internal_pending[tok] = pend + self.internal_pending_increment
                continue

            if best_efe < best_efe_global:
                best_efe_global = best_efe
                best_mono = InternalMonologue(
                    reply=draft,
                    trigger_token=tok,
                    tension=tension,
                    best_efe=best_efe,
                    output_branch=branch,
                    epsilon_secondary=eps_sec,
                    semantic_surprise_max=max_surp,
                    u_curiosity=uc,
                    association_trigger=a_tr,
                    association_pick=a_pk,
                )

        if best_mono is not None:
            self._word_memory.note_trigger_success(best_mono.trigger_token)
            self._internal_pending.pop(best_mono.trigger_token, None)
            return best_mono

        return None

    def internal_tick_train_fast(self) -> Optional[InternalMonologue]:
        """
        训练期内部快路径：只扫描张力并累积/衰减 pending，不做候选生成与 EFE 评估。
        这样保留内心激活的统计痕迹，同时避免预热时为不会被回应的独白付出高开销。
        """
        self._decay_internal_pending_tick()
        base_resource_snap = self._freeze_resource()
        uc = float(self.u_curiosity())
        rf = max(
            0.0,
            min(
                1.0,
                (base_resource_snap[0] / max(base_resource_snap[2] * 0.15, 0.22, 1e-6))
                * max(0.0, min(1.0, 1.0 - base_resource_snap[1])),
            ),
        )
        ordered = self._internal_frontier_candidates(
            base_resource_snap,
            cold_budget=0,
            surprise_hist_scan_cap=self.train_fast_internal_surprise_hist_cap,
        )
        for tok, precomputed_surprise in ordered:
            surp = float(precomputed_surprise)
            tv = self.preference_beta_token * self._token_value.get(str(tok).strip(), 0.0)
            tension = surp * (0.5 + uc) * (0.3 + rf) + float(
                self._internal_pending.get(tok, 0.0)
            ) + tv
            if tension >= self.internal_tension_threshold:
                self._internal_pending[tok] = float(
                    self._internal_pending.get(tok, 0.0)
                ) + self.internal_pending_increment
        return None

    def internal_tick_train_fast_many(self, steps: int) -> Optional[InternalMonologue]:
        """
        训练期内部快路径的批量版本：在没有外部输入介入的连续若干 tick 中，
        复用同一批候选 token 与基础张力，减少重复扫描/构造开销。
        """
        steps = max(0, int(steps))
        if steps <= 0:
            return None
        base_resource_snap = self._freeze_resource()
        uc = float(self.u_curiosity())
        rf = max(
            0.0,
            min(
                1.0,
                (base_resource_snap[0] / max(base_resource_snap[2] * 0.15, 0.22, 1e-6))
                * max(0.0, min(1.0, 1.0 - base_resource_snap[1])),
            ),
        )
        ordered = self._internal_frontier_candidates(
            base_resource_snap,
            cold_budget=0,
            surprise_hist_scan_cap=self.train_fast_internal_surprise_hist_cap,
        )
        if not ordered:
            for _ in range(steps):
                self._decay_internal_pending_tick()
            return None

        prepared: List[Tuple[str, float]] = []
        for tok, precomputed_surprise in ordered:
            bt = float(precomputed_surprise) * (0.5 + uc) * (0.3 + rf)
            tv = self.preference_beta_token * self._token_value.get(str(tok).strip(), 0.0)
            prepared.append((tok, bt + tv))

        for _ in range(steps):
            self._decay_internal_pending_tick()
            for tok, base_tension in prepared:
                tension = base_tension + float(self._internal_pending.get(tok, 0.0))
                if tension >= self.internal_tension_threshold:
                    self._internal_pending[tok] = float(
                        self._internal_pending.get(tok, 0.0)
                    ) + self.internal_pending_increment
        return None

    @staticmethod
    def _shrink_plan_once(plan: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        if len(plan) <= 1:
            return list(plan)
        return plan[:-1]

    def _plan_at_depth(
        self, base_plan: List[Tuple[str, str]], depth: int
    ) -> List[Tuple[str, str]]:
        p = list(base_plan)
        for _ in range(max(0, depth)):
            p = self._shrink_plan_once(p)
        return p

    def _snapshot_imprint(self, intent: CommunicativeIntent) -> WordStateImprint:
        st = self.tokenizer.resource_state()
        return WordStateImprint(
            F_ema=float(st.get("F_ema", 0.0)),
            R=float(st.get("R", 0.0)),
            m=float(st.get("m", 0.0)),
            u_curiosity=float(intent.u_curiosity),
            u_task=float(intent.u_task),
        )

    def _build_conflict_probe_draft(
        self,
        focus_token: str,
        intent: CommunicativeIntent,
        pi_s: float,
        user_text: str,
        replan_level: int,
        *,
        hedge_prefix: str = "",
    ) -> str:
        """印记失配时的极简外投：把焦点词抛出，征求能降低自由能的反馈。"""
        st = self.tokenizer.resource_state()
        r = float(st.get("R", 0.0))
        r_max = max(float(st.get("R_max", 1.0)), 1e-6)
        budget = max(1, int(2.5 * (r / r_max)) + (1 if pi_s > 0.45 else 0))
        budget = max(1, budget - replan_level)
        parts: List[str] = []
        if hedge_prefix:
            parts.append(hedge_prefix)
        parts.append(f"{focus_token}？现在？")
        parts.append(f"（内心最张紧「{intent.tension_slot}」。）")
        joined = "".join(parts)
        body = joined[: max(28, budget * 18)]
        if len(joined) > len(body):
            body += "…"
        return body

    def _build_associative_probe_drafts(
        self,
        trigger_token: str,
        associates: List[Tuple[str, int]],
        replan_level: int,
        *,
        hedge_prefix: str = "",
    ) -> List[Tuple[str, str]]:
        """把触发词与历史邻接关联词拼成少量短探针；返回 (草稿, 归因联想词 surface)。"""
        trigger = trigger_token.strip()
        if not trigger:
            return []
        st = self.tokenizer.resource_state()
        r = float(st.get("R", 0.0))
        r_max = max(float(st.get("R_max", 1.0)), 1e-6)
        budget = max(1, int(2.5 * (r / r_max)))
        budget = max(1, budget - replan_level)
        max_len = max(18, budget * 14)

        drafts: List[Tuple[str, str]] = []
        seen: set[str] = set()

        def _append(text: str, assoc_surface: str) -> bool:
            body = text[:max_len]
            if len(text) > len(body):
                body += "…"
            if body in seen:
                return False
            drafts.append((body, assoc_surface))
            seen.add(body)
            return len(drafts) >= 3

        for assoc, _count in associates[:3]:
            a = assoc.strip()
            if not a or a == trigger:
                continue
            form = f"{trigger}？{a}？"
            text = f"{hedge_prefix}{form}" if hedge_prefix else form
            if _append(text, a):
                return drafts

        for assoc, _count in associates[:3]:
            a = assoc.strip()
            if not a or a == trigger:
                continue
            for form in (f"{a}...{trigger}？", f"{trigger}和{a}？"):
                text = f"{hedge_prefix}{form}" if hedge_prefix else form
                if _append(text, a):
                    return drafts
        return drafts

    def _realize(
        self,
        intent: CommunicativeIntent,
        plan: List[Tuple[str, str]],
        pi_s: float,
        eps_social: float,
        user_text: str,
        replan_level: int,
        *,
        hedge_prefix: str = "",
        listen_tokens: List[str] | None = None,
    ) -> str:
        st = self.tokenizer.resource_state()
        r = float(st.get("R", 0.0))
        r_max = max(float(st.get("R_max", 1.0)), 1e-6)
        budget = max(1, int(2.5 * (r / r_max)) + (1 if pi_s > 0.45 else 0))
        budget = max(1, budget - replan_level)

        toks = (
            listen_tokens
            if listen_tokens is not None
            else self.tokenizer.tokenize(user_text)
        )
        surface = toks[0] if toks else "这句"

        parts: List[str] = []
        if hedge_prefix:
            parts.append(hedge_prefix)

        if intent.speech_act == "respond":
            parts.append(f"关于「{surface}」，我在压低由你造成的惊奇；")
            parts.append(f"内部最张紧的是「{intent.tension_slot}」。")
        elif intent.speech_act == "explore":
            parts.append(f"我想先对「{intent.tension_slot}」做输出式探索；")
            parts.append(f"此刻最结晶的是「{intent.anchor_slot}」。")
        else:
            parts.append("我在听；社会预测误差正在绑定高层状态；")
            parts.append(f"先稳住「{intent.anchor_slot}」。")

        body = "".join(parts)[: budget * 18]
        if len("".join(parts)) > len(body):
            body += "…"
        return body

    def _build_explore_echo_draft(
        self,
        intent: CommunicativeIntent,
        plan: List[Tuple[str, str]],
        pi_s: float,
        _eps_social: float,
        user_text: str,
        replan_level: int,
        *,
        hedge_prefix: str = "",
        listen_tokens: List[str] | None = None,
    ) -> str:
        """
        「回声」候选：短序列以降低 ε_secondary；张力槽位短注仅在预算充足时出现。
        """
        st = self.tokenizer.resource_state()
        r = float(st.get("R", 0.0))
        r_max = max(float(st.get("R_max", 1.0)), 1e-6)
        budget = max(1, int(2.5 * (r / r_max)) + (1 if pi_s > 0.45 else 0))
        budget = max(1, budget - replan_level)

        toks = (
            listen_tokens
            if listen_tokens is not None
            else self.tokenizer.tokenize(user_text)
        )
        surface = toks[0] if toks else "这句"
        parts: List[str] = []
        if hedge_prefix:
            parts.append(hedge_prefix)
        parts.append(f"「{surface}」——回声。")
        if budget >= 2:
            parts.append(f"（「{intent.tension_slot}」）")
        joined = "".join(parts)
        body = joined[: max(32, budget * 18)]
        if len(joined) > len(body):
            body += "…"
        return body

    def produce_turn(self, user_text: str) -> DialogueTurn:
        eps_in, toks_in = self._listen_trace(user_text)
        self._decay_internal_pending()
        intent = self._germinate_intent(user_text, eps_in)
        plan = self._sparse_plan(intent)
        pi_s = self.pi_statement()
        snap_pre = self._snapshot_imprint(intent)
        max_surp, focus_tok = self._word_memory.max_surprise_in_tokens(toks_in, snap_pre)
        st_listen = dict(self.tokenizer.resource_state())
        # 候选生成里会 tokenize(user_text)，会改 R、m；EFE 必须基于「倾听结束」同一快照比较。
        listen_resource_snap = self._freeze_resource()

        draft, eps_sec, branch, replans, best_efe_raw, a_tr, a_pk = self._efe_best_reply(
            intent=intent,
            plan=plan,
            pi_s=pi_s,
            user_text=user_text,
            eps_in=float(eps_in),
            listen_resource_snap=listen_resource_snap,
            max_surp=max_surp,
            focus_tok=focus_tok or "",
            internal_monologue=False,
            listen_tokens=toks_in,
        )

        self._word_memory.record_tokens_with_context(
            toks_in,
            F_ema=float(st_listen.get("F_ema", 0.0)),
            R=float(st_listen.get("R", 0.0)),
            m=float(st_listen.get("m", 0.0)),
            u_curiosity=float(intent.u_curiosity),
            u_task=float(intent.u_task),
        )

        self._record_recent_attention(toks_in, focus_tok or "")

        return DialogueTurn(
            user_text=user_text,
            reply=draft,
            u_curiosity=intent.u_curiosity,
            u_task=intent.u_task,
            pi_statement=pi_s,
            epsilon_social_in=eps_in,
            epsilon_secondary=eps_sec,
            intent=intent,
            replan_count=replans,
            resource_snapshot=dict(self.tokenizer.resource_state()),
            output_branch=branch,
            semantic_surprise_max=max_surp,
            conflict_focus_token=focus_tok or "",
            association_trigger=a_tr,
            association_pick=a_pk,
            best_efe=float(best_efe_raw),
        )

    def train_step(self, user_text: str) -> Dict[str, object]:
        """
        训练期轻量路径：只做倾听、sigma/印记/最近注意更新，不生成外部回复与 EFE 候选。
        用于离线预热时减少大量 draft 评估开销。
        """
        eps_in, toks_in = self._listen_trace(user_text)
        self._decay_internal_pending()
        intent = self._germinate_intent(user_text, eps_in)
        snap_pre = self._snapshot_imprint(intent)
        max_surp, focus_tok = self._word_memory.max_surprise_in_tokens(toks_in, snap_pre)
        st_listen = dict(self.tokenizer.resource_state())
        self._word_memory.record_tokens_with_context(
            toks_in,
            F_ema=float(st_listen.get("F_ema", 0.0)),
            R=float(st_listen.get("R", 0.0)),
            m=float(st_listen.get("m", 0.0)),
            u_curiosity=float(intent.u_curiosity),
            u_task=float(intent.u_task),
        )
        self._record_recent_attention(toks_in, focus_tok or "")
        return {
            "user_text": user_text,
            "u_curiosity": float(intent.u_curiosity),
            "u_task": float(intent.u_task),
            "epsilon_social_in": float(eps_in),
            "semantic_surprise_max": float(max_surp),
            "conflict_focus_token": focus_tok or "",
            "resource_snapshot": dict(self.tokenizer.resource_state()),
            "output_branch": "train_fast",
        }

    def train_step_batch(self, user_texts: List[str]) -> List[Dict[str, object]]:
        """
        训练期批量轻量路径。
        第一版先保持与逐条 `train_step()` 完全等价，为后续把 tokenizer 热路径下沉到
        Cython/原生实现提供稳定批量边界。
        """
        return [self.train_step(text) for text in user_texts if text.strip()]

    def turn(self, user_text: str) -> DialogueTurn:
        """对外 API：一轮完整认知对话回路。"""
        return self.produce_turn(user_text)


__all__ = [
    "CognitiveDialogueAgent",
    "DialogueTurn",
    "CommunicativeIntent",
    "DIALOGUE_MODEL_FORMAT",
    "DIALOGUE_MODEL_FORMAT_VERSION",
    "PREFERENCE_STATE_FORMAT",
    "InternalMonologue",
    "OutputBranch",
]
