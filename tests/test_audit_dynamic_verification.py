"""
动态审计测试：不修改 uva_model 等业务实现，仅用 mock / 计数 / 行为断言
验证「重复计算、状态不对齐、多余 IO、API 契约落空」等问题。
"""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from typing import Optional, Set
from unittest.mock import MagicMock, patch

from uva_model.checkpoint_json import read_json_document
from uva_model.corpus_jsonl import iter_corpus_jsonl_records, iter_corpus_jsonl_training_episodes
from uva_model.dialogue import CognitiveDialogueAgent, DialogueTurn, InternalMonologue
from uva_model.tokenizer import PrecisionTokenizer
from uva_model.word_imprints import WordStateImprint, WordStateMemory


class DialogueDynamicAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = PrecisionTokenizer()
        self.tokenizer.fit(["你好 世界", "你好 朋友", "今天 天气 怎样"])

    def test_expected_free_energy_calls_resource_factor_once(self) -> None:
        """EFE 单次评估入口只算一次 rf，供 λ_explore / explore_commit / pred 复用。"""
        agent = CognitiveDialogueAgent(self.tokenizer, learn_tokenizer_from_user=False)
        with patch.object(
            agent,
            "_resource_factor",
            wraps=agent._resource_factor,
        ) as spy:
            _ = agent._expected_free_energy(
                "短",
                eps_social=0.5,
                u_curiosity=0.4,
                is_explore=True,
                imprint_surprise=0.1,
                internal_monologue=False,
            )
        self.assertEqual(spy.call_count, 1)

    def test_produce_turn_reuses_listen_tokens_without_extra_tokenize(self) -> None:
        """倾听后候选构造复用 listen_tokens，不再对整句 user_text 反复 tokenize。"""
        agent = CognitiveDialogueAgent(self.tokenizer, learn_tokenizer_from_user=False)
        user = "你好今天天气"
        with patch.object(
            agent.tokenizer,
            "tokenize",
            wraps=agent.tokenizer.tokenize,
        ) as spy_tok:
            _ = agent.produce_turn(user)
        self.assertEqual(
            spy_tok.call_count,
            0,
            msg="produce_turn 应仅 _listen_trace 分词，候选不再调用 tokenize",
        )

    def test_internal_monologue_imprint_snap_unifies_frontier_and_tick(self) -> None:
        """内心路径：前沿惊奇与 internal_tick 的 max_surp 共用同一快照（u_task=0，不随 trigger 文本变化）。"""
        agent = CognitiveDialogueAgent(self.tokenizer, learn_tokenizer_from_user=False)
        mem = agent._word_memory
        mem.record(
            "词甲",
            WordStateImprint(F_ema=1.0, R=0.9, m=0.05, u_curiosity=0.2, u_task=0.2),
        )
        base_snap = (0.9, 0.05, 1.0, 1.0)
        internal_snap = agent._internal_monologue_imprint_snap(base_snap)
        self.assertEqual(internal_snap.u_task, 0.0)
        s_internal = mem.semantic_surprise_for_token("词甲", internal_snap)
        s_internal_repeat = mem.semantic_surprise_for_token("词甲", internal_snap)
        self.assertAlmostEqual(s_internal, s_internal_repeat, places=9)
        q_intent = agent._germinate_intent("词甲？", 0.0)
        snap_q = WordStateImprint(
            F_ema=1.0,
            R=0.9,
            m=0.05,
            u_curiosity=float(q_intent.u_curiosity),
            u_task=float(q_intent.u_task),
        )
        s_q = mem.semantic_surprise_for_token("词甲", snap_q)
        if float(q_intent.u_task) > 0.0:
            self.assertGreater(
                abs(s_q - s_internal),
                1e-9,
                msg="旧路径用 trigger 抬高 u_task 会改变惊奇；内心路径不应如此",
            )

    def test_internal_tick_does_not_recompute_semantic_surprise_in_loop(self) -> None:
        """frontier 已算惊奇；internal_tick 循环内复用 precomputed_surprise，不再二次 semantic_surprise_for_token。"""
        agent = CognitiveDialogueAgent(
            self.tokenizer,
            learn_tokenizer_from_user=False,
            internal_tension_threshold=0.0,
            internal_max_efe_to_speak=500.0,
            internal_spontaneous_jitter=0.0,
            internal_global_scan_k=4,
        )
        for tok in ("甲", "乙", "丙"):
            agent._word_memory.record(
                tok,
                WordStateImprint(F_ema=0.2, R=0.5, m=0.1, u_curiosity=0.3, u_task=0.2),
            )
        agent.turn("你好今天")
        with patch.object(
            agent._word_memory,
            "semantic_surprise_for_token",
            wraps=agent._word_memory.semantic_surprise_for_token,
        ) as spy:
            _ = agent.internal_tick()
        frontier_only = spy.call_count
        self.assertGreaterEqual(
            frontier_only,
            1,
            msg="至少应在 _internal_frontier_candidates 中为候选 token 计算一次惊奇",
        )
        # 修复前：循环内对每个过阈候选再算 max_surp，次数约为 2×候选数
        self.assertLess(
            frontier_only,
            8,
            msg="internal_tick 循环内不应再对每个 token 二次调用 semantic_surprise_for_token",
        )

    def test_arbitration_tie_favors_external(self) -> None:
        """arbitration_external_first：g_ext_net == g_int 时返回 True（外部优先）。"""
        agent = CognitiveDialogueAgent(self.tokenizer, learn_tokenizer_from_user=False)
        turn = MagicMock(spec=DialogueTurn)
        turn.best_efe = 1.0
        turn.epsilon_social_in = 0.0
        mono = MagicMock(spec=InternalMonologue)
        mono.best_efe = 1.0
        with patch.object(
            agent,
            "social_arbitration_ticket",
            return_value=0.0,
        ):
            self.assertTrue(
                agent.arbitration_external_first(mono, turn),
                msg="净成本相等时应为外部先说（<=）",
            )


class PopCountingWordStateMemory(WordStateMemory):
    """
    仅用于测试：与 WordStateMemory._trim_active_tokens 逻辑同步（见 word_imprints 同方法），
    额外统计 pop(0) 次数。Python 3.11+ 无法 patch list.pop，故用子类复制小循环。
    """

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self.pop0_trim_hits = 0

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
            self.pop0_trim_hits += 1
            self._deactivate_token(removable.pop(0))


class TrimActiveTokensPopZeroDynamicTests(unittest.TestCase):
    """_trim_active_tokens 使用 removable.pop(0)：动态计数 + 与 pop 尾端对比的耗时比（不修改产品代码）。"""

    def test_trim_active_tokens_uses_pop_index_zero_per_deactivation(self) -> None:
        mem = PopCountingWordStateMemory(capacity=1)
        imp = WordStateImprint(1.0, 1.0, 0.0, 0.1, 0.1)
        n = 48
        for i in range(n):
            mem.record(f"w{i}", imp)
        self.assertGreater(len(mem._active_tokens), 10)
        target_keep = 4
        mem._trim_active_tokens(target_keep, protected=set())
        self.assertLessEqual(len(mem._active_tokens), target_keep)
        need = max(0, n - target_keep)
        self.assertEqual(
            mem.pop0_trim_hits,
            need,
            msg="每休眠一个 token 对应一次 pop(0)，与 O(n) 单次移动叠加形成 O(n²) 风险",
        )

    def test_pop_zero_trim_quadratic_vs_pop_tail_reference(self) -> None:
        """纯 Python：同等「从表头删到剩 k 个」时 pop(0) 比反复 pop() 尾删慢一个量级以上（环境无关的算法对比）。"""
        n = 8000
        k = 50

        def trim_head_pop0() -> None:
            removable = list(range(n))
            removable.sort(reverse=True)
            while len(removable) > k:
                removable.pop(0)

        def trim_tail_pop() -> None:
            removable = list(range(n))
            removable.sort(reverse=True)
            while len(removable) > k:
                removable.pop()

        t0 = time.perf_counter()
        trim_head_pop0()
        t_head = time.perf_counter() - t0
        t1 = time.perf_counter()
        trim_tail_pop()
        t_tail = time.perf_counter() - t1
        self.assertLess(
            t_tail,
            t_head * 0.5,
            msg="表头 pop(0) 在 n 较大时应显著慢于尾 pop（同一 n、k 的参考实现）",
        )

    def test_subclass_trim_matches_production_final_active_count(self) -> None:
        """子类计数版与库内 WordStateMemory 裁剪结果一致（防测试双写漂移）。"""
        n = 20
        target_keep = 5
        imp = WordStateImprint(1.0, 1.0, 0.0, 0.1, 0.1)
        sub = PopCountingWordStateMemory(capacity=1)
        vanilla = WordStateMemory(capacity=1)
        for i in range(n):
            sub.record(f"t{i}", imp)
            vanilla.record(f"t{i}", imp)
        sub._trim_active_tokens(target_keep, protected=set())
        vanilla._trim_active_tokens(target_keep, protected=set())
        self.assertEqual(len(sub._active_tokens), len(vanilla._active_tokens))
        self.assertEqual(sub.pop0_trim_hits, n - target_keep)


class ProduceTurnTraceCountDynamicTests(unittest.TestCase):
    """PrecisionTokenizer._trace：倾听、tokenize、mean_surprise 均走 _trace，对整轮 EFE 做上下界。"""

    def setUp(self) -> None:
        self.tokenizer = PrecisionTokenizer()
        self.tokenizer.fit(["你好 世界", "你好 朋友", "今天 天气 怎样"])

    def test_produce_turn_trace_call_count_has_sane_bounds(self) -> None:
        agent = CognitiveDialogueAgent(self.tokenizer, learn_tokenizer_from_user=False)
        user = "你好今天天气怎样"
        with patch.object(
            self.tokenizer,
            "_trace",
            wraps=self.tokenizer._trace,
        ) as spy:
            _ = agent.produce_turn(user)
        c = spy.call_count
        self.assertGreaterEqual(
            c,
            8,
            msg="倾听 1 次 + 草稿 mean_surprise；复用 tokens 后 _trace 仍应多于裸听",
        )
        self.assertLessEqual(
            c,
            120,
            msg="优化后 EFE 枚举上界下调；异常暴涨可能暗示死循环或重复全句 trace",
        )

    def test_efe_best_reply_trace_call_count_bounds_without_focus(self) -> None:
        tok = PrecisionTokenizer()
        tok.fit(["你好 世界", "测试 句子"])
        agent = CognitiveDialogueAgent(tok, learn_tokenizer_from_user=False, internal_rng_seed=1)
        intent = agent._germinate_intent("测试", 0.3)
        plan = agent._sparse_plan(intent)
        snap = agent._freeze_resource()
        with patch.object(tok, "_trace", wraps=tok._trace) as spy:
            _ = agent._efe_best_reply(
                intent=intent,
                plan=plan,
                pi_s=0.5,
                user_text="测试句子流",
                eps_in=0.2,
                listen_resource_snap=snap,
                max_surp=0.0,
                focus_tok="",
                internal_monologue=False,
            )
        c = spy.call_count
        self.assertGreaterEqual(c, 3, msg="无焦点时草稿 mean_surprise → _trace")
        self.assertLessEqual(c, 80)

    def test_train_step_invokes_fewer_traces_than_produce_turn_same_text(self) -> None:
        """train_step 不做 EFE；同句下 _trace 次数应显著少于 produce_turn（性能对比）。"""
        user = "你好今天天气"
        tok_a = PrecisionTokenizer()
        tok_a.fit(["你好 世界", "今天 天气"])
        agent_a = CognitiveDialogueAgent(tok_a, learn_tokenizer_from_user=False)
        with patch.object(tok_a, "_trace", wraps=tok_a._trace) as spy_a:
            agent_a.train_step(user)
        c_train = spy_a.call_count
        tok_b = PrecisionTokenizer()
        tok_b.fit(["你好 世界", "今天 天气"])
        agent_b = CognitiveDialogueAgent(tok_b, learn_tokenizer_from_user=False)
        with patch.object(tok_b, "_trace", wraps=tok_b._trace) as spy_b:
            agent_b.produce_turn(user)
        c_prod = spy_b.call_count
        self.assertLess(
            c_train,
            c_prod // 2,
            msg="produce_turn 的 EFE 候选对草稿反复 mean_surprise/tokenize，_trace 应远高于 train_step",
        )


class WordStateMemoryDynamicAuditTests(unittest.TestCase):
    def test_each_record_bumps_global_cache_version(self) -> None:
        """任意 record 触发 _invalidate_cache，版本单调增，GPU 张量缓存被整体作废。"""
        mem = WordStateMemory(capacity=10)
        imp = WordStateImprint(1.0, 1.0, 0.0, 0.1, 0.1)
        v0 = mem._cache_version
        mem.record("a", imp)
        v1 = mem._cache_version
        mem.record("b", imp)
        v2 = mem._cache_version
        self.assertGreater(v1, v0)
        self.assertGreater(v2, v1)

    def test_merge_record_does_not_bump_global_cache_version(self) -> None:
        """合并命中仅失效关联缓存，不整表 _invalidate_cache。"""
        mem = WordStateMemory(capacity=10)
        same = WordStateImprint(
            1.0,
            1.0,
            0.0,
            0.2,
            0.2,
            context_before="x",
            context_after="y",
        )
        mem.record("k", same)
        v_after_first = mem._cache_version
        mem.record("k", same)
        self.assertEqual(mem._cache_version, v_after_first)

    def test_soft_freeze_iterates_store_without_tokens_copy(self) -> None:
        """soft_freeze_cold_tokens 直接遍历 _store，不调用 tokens()。"""
        mem = WordStateMemory(capacity=5)
        imp = WordStateImprint(1.0, 1.0, 0.0, 0.1, 0.1)
        for i in range(12):
            mem.record(f"s{i}", imp)
        calls = 0

        def counting_tokens() -> list[str]:
            nonlocal calls
            calls += 1
            return list(mem._store.keys())

        with patch.object(mem, "tokens", side_effect=counting_tokens):
            mem.soft_freeze_cold_tokens(max_active_tokens=4)
            mem.soft_freeze_cold_tokens(max_active_tokens=3)
        self.assertEqual(calls, 0)

    def test_associated_tokens_hit_reuses_cached_tuple(self) -> None:
        """缓存命中返回 list 视图，底层 tuple 复用。"""
        mem = WordStateMemory(capacity=10)
        mem.record(
            "hub",
            WordStateImprint(1, 1, 0, 0.1, 0.1, context_before="", context_after="a"),
        )
        mem.record(
            "a",
            WordStateImprint(1, 1, 0, 0.1, 0.1, context_before="hub", context_after="b"),
        )
        mem.record(
            "b",
            WordStateImprint(1, 1, 0, 0.1, 0.1, context_before="a", context_after=""),
        )
        r1 = mem.associated_tokens("hub", k=2)
        r2 = mem.associated_tokens("hub", k=2)
        self.assertEqual(r1, r2)
        self.assertIsInstance(r1, list)


class CorpusJsonlDynamicAuditTests(unittest.TestCase):
    def test_training_episodes_iterator_supports_strict_flag(self) -> None:
        """iter_corpus_jsonl_training_episodes 可将 strict 透传到 records。"""
        lines = [
            json.dumps({"meta": {}}, ensure_ascii=False),
        ]
        with tempfile.NamedTemporaryFile(
            "w",
            suffix=".jsonl",
            encoding="utf-8",
            delete=False,
        ) as fh:
            fh.write("\n".join(lines) + "\n")
            path = fh.name
        try:
            with self.assertRaises(ValueError):
                list(
                    iter_corpus_jsonl_training_episodes(
                        [path],
                        episode_size=2,
                        min_episode_turns=1,
                        skip_bad_lines=False,
                        strict=True,
                    )
                )
            episodes = list(
                iter_corpus_jsonl_training_episodes(
                    [path],
                    episode_size=2,
                    min_episode_turns=1,
                    skip_bad_lines=False,
                    strict=False,
                )
            )
            self.assertEqual(len(episodes), 0)
        finally:
            Path(path).unlink(missing_ok=True)


class CheckpointJsonDynamicAuditTests(unittest.TestCase):
    def test_read_json_document_reads_entire_file_bytes(self) -> None:
        """read_json_document 通过 read_bytes 整文件入内存；大文件峰值与流式 API 目标不一致。"""
        payload = {"ok": True, "x": list(range(50))}
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            suffix=".json",
            delete=False,
        ) as fh:
            json.dump(payload, fh)
            path = fh.name
        try:
            p = Path(path)
            orig_read = Path.read_bytes
            reads: list[int] = []

            def tracking_read_bytes(slf: Path) -> bytes:
                reads.append(1)
                data = orig_read(slf)
                self.assertGreater(len(data), 0)
                return data

            with patch.object(Path, "read_bytes", tracking_read_bytes):
                out = read_json_document(path)
            self.assertEqual(reads, [1], msg="整文件一次性 read_bytes，非分块流式解析 JSON")
            self.assertTrue(out.get("ok"))
        finally:
            Path(path).unlink(missing_ok=True)


class TrainStepBatchDynamicAuditTests(unittest.TestCase):
    def test_train_step_batch_skips_whitespace_only_strings(self) -> None:
        """过滤条件为 `if text.strip()`，纯空白不再进入 train_step。"""
        tok = PrecisionTokenizer()
        tok.fit(["a b"])
        agent = CognitiveDialogueAgent(tok, learn_tokenizer_from_user=False)
        out = agent.train_step_batch(["  \t  "])
        self.assertEqual(len(out), 0)


class DialoguePartialLoadDynamicAuditTests(unittest.TestCase):
    def test_apply_dialogue_model_missing_sigma_raises(self) -> None:
        """严格快照：仅提供 word_imprints 不提供 sigma → ValueError，且不修改已有 σ。"""
        tok = PrecisionTokenizer()
        tok.fit(["a b", "c d"])
        agent = CognitiveDialogueAgent(tok, learn_tokenizer_from_user=False)
        agent._sigma[agent.SLOT_NAMES[0]] = 0.99
        snap = {k: 0.11 for k in agent.SLOT_NAMES}
        with self.assertRaises(ValueError):
            agent.apply_dialogue_model_dict(
                {
                    "format": "cognitive_dialogue_agent_v1",
                    "word_imprints": agent._word_memory.to_dict(),
                }
            )
        self.assertAlmostEqual(agent._sigma[agent.SLOT_NAMES[0]], 0.99)
        agent.apply_dialogue_model_dict(
            {
                "format": "cognitive_dialogue_agent_v1",
                "sigma": snap,
                "word_imprints": agent._word_memory.to_dict(),
            }
        )
        self.assertAlmostEqual(agent._sigma[agent.SLOT_NAMES[0]], 0.11)

    def test_malformed_preference_state_raises_and_leaves_state_unchanged(self) -> None:
        """preference_state format 错误 → ValueError；加载前状态不被部分覆盖。"""
        tok = PrecisionTokenizer()
        tok.fit(["x y"])
        agent = CognitiveDialogueAgent(tok, learn_tokenizer_from_user=False)
        agent._branch_bias["conservative"] = 0.77
        with self.assertRaises(ValueError):
            agent.apply_dialogue_model_dict(
                {
                    "format": "cognitive_dialogue_agent_v1",
                    "sigma": {k: 0.5 for k in agent.SLOT_NAMES},
                    "word_imprints": agent._word_memory.to_dict(),
                    "preference_state": {"format": "wrong", "branch_bias": {"conservative": -9.0}},
                }
            )
        self.assertAlmostEqual(agent._branch_bias.get("conservative", 0.0), 0.77)


class EfeRestoreSpamDynamicAuditTests(unittest.TestCase):
    def test_efe_best_reply_restore_bounded_per_depth(self) -> None:
        """_efe_best_reply 每 depth 至多一次 restore + 收尾，不再按候选数膨胀。"""
        tok = PrecisionTokenizer()
        tok.fit(["你好 世界", "测试 句子"])
        agent = CognitiveDialogueAgent(
            tok,
            learn_tokenizer_from_user=False,
            internal_rng_seed=0,
        )
        intent = agent._germinate_intent("测试", 0.3)
        plan = agent._sparse_plan(intent)
        listen_snap = agent._freeze_resource()
        with patch.object(
            agent,
            "epsilon_secondary_on_draft",
            side_effect=lambda _draft: 0.1,
        ):
            with patch.object(
                agent, "_restore_resource", wraps=agent._restore_resource
            ) as spy:
                _ = agent._efe_best_reply(
                    intent=intent,
                    plan=plan,
                    pi_s=0.5,
                    user_text="测试句子",
                    eps_in=0.2,
                    listen_resource_snap=listen_snap,
                    max_surp=0.0,
                    focus_tok="",
                    internal_monologue=False,
                    listen_tokens=["测试", "句子"],
                )
        self.assertLessEqual(
            spy.call_count,
            12,
            msg="_efe_best_reply 本体每 depth 一次 restore + 收尾，不再按候选×前缀膨胀",
        )


class EpsilonSecondaryRestoreIntegrityTests(unittest.TestCase):
    def test_epsilon_secondary_on_draft_leaves_frozen_resource_snapshot(self) -> None:
        """草稿 trace 会改 R/m；epsilon_secondary_on_draft 末尾应恢复到倾听前快照。"""
        tok = PrecisionTokenizer()
        tok.fit(["abc def", "ghi jkl"])
        agent = CognitiveDialogueAgent(tok, learn_tokenizer_from_user=False)
        snap = agent._freeze_resource()
        _ = agent.epsilon_secondary_on_draft("这是一段用于消耗资源的较长草稿文本" * 3)
        after = agent._freeze_resource()
        self.assertEqual(after, snap)


if __name__ == "__main__":
    unittest.main()
