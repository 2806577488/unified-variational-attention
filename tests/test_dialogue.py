import json
import tempfile
import unittest
from pathlib import Path
from typing import Dict, cast

from uva_model.dialogue import (
    CognitiveDialogueAgent,
    CommunicativeIntent,
    DIALOGUE_MODEL_FORMAT,
    DIALOGUE_MODEL_FORMAT_VERSION,
    DialogueTurn,
    InternalMonologue,
)
from uva_model.tokenizer import PrecisionTokenizer
from uva_model.word_imprints import WordStateImprint


class DialogueTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = PrecisionTokenizer()
        self.tokenizer.fit(["你好 世界", "你好 朋友", "今天 天气"])
        self.agent = CognitiveDialogueAgent(self.tokenizer)

    def test_recent_attention_ring_records_user_tokens(self) -> None:
        agent = CognitiveDialogueAgent(self.tokenizer, recent_attention_capacity=32)
        self.assertEqual(agent.recent_attention_words, [])
        agent.turn("你好今天")
        buf = agent.recent_attention_words
        self.assertGreater(len(buf), 0)
        joined = "".join(buf)
        self.assertIn("你好", joined)

    def test_internal_tick_none_without_recent_attention(self) -> None:
        agent = CognitiveDialogueAgent(self.tokenizer, learn_tokenizer_from_user=False)
        self.assertIsNone(agent.internal_tick())

    def test_internal_monologue_imprint_snap_forces_zero_u_task(self) -> None:
        agent = CognitiveDialogueAgent(self.tokenizer, learn_tokenizer_from_user=False)
        snap = agent._freeze_resource()
        imprint = agent._internal_monologue_imprint_snap(snap)
        self.assertEqual(imprint.u_task, 0.0)
        self.assertEqual(imprint.u_curiosity, agent.u_curiosity())
        intent = agent._germinate_internal_intent()
        self.assertEqual(intent.u_task, 0.0)
        self.assertEqual(intent.u_curiosity, agent.u_curiosity())

    def test_internal_tick_emits_when_threshold_zero(self) -> None:
        agent = CognitiveDialogueAgent(
            self.tokenizer,
            learn_tokenizer_from_user=False,
            internal_tension_threshold=0.0,
            internal_max_efe_to_speak=500.0,
        )
        agent.turn("你好今天")
        mono = agent.internal_tick()
        self.assertIsNotNone(mono)
        self.assertGreater(len(mono.reply), 0)

    def test_internal_tick_scans_global_imprints_without_recent_attention(self) -> None:
        agent = CognitiveDialogueAgent(
            self.tokenizer,
            learn_tokenizer_from_user=False,
            internal_tension_threshold=0.0,
            internal_global_scan_k=4,
            internal_spontaneous_jitter=0.0,
        )
        agent._word_memory.record(  # noqa: SLF001
            "旧事",
            WordStateImprint(F_ema=0.1, R=0.1, m=0.0, u_curiosity=0.0, u_task=0.0),
        )
        self.assertEqual(agent.recent_attention_words, [])
        mono = agent.internal_tick()
        self.assertIsNotNone(mono)
        self.assertEqual(mono.trigger_token, "旧事")

    def test_internal_tick_is_deterministic_when_jitter_zero(self) -> None:
        agent = CognitiveDialogueAgent(
            self.tokenizer,
            learn_tokenizer_from_user=False,
            internal_tension_threshold=0.0,
            internal_global_scan_k=4,
            internal_spontaneous_jitter=0.0,
        )
        agent._word_memory.record(  # noqa: SLF001
            "旧事",
            WordStateImprint(F_ema=0.1, R=0.1, m=0.0, u_curiosity=0.0, u_task=0.0),
        )
        mono0 = agent.internal_tick()
        mono1 = agent.internal_tick()
        self.assertIsNotNone(mono0)
        self.assertIsNotNone(mono1)
        self.assertEqual(mono0.trigger_token, mono1.trigger_token)
        self.assertEqual(mono0.reply, mono1.reply)

    def test_internal_tick_uses_tiered_scan_not_full_memory(self) -> None:
        agent = CognitiveDialogueAgent(
            self.tokenizer,
            learn_tokenizer_from_user=False,
            internal_tension_threshold=0.0,
            internal_global_scan_k=2,
            internal_spontaneous_jitter=0.0,
        )
        for tok in ("冷1", "冷2", "冷3", "冷4"):
            agent._word_memory.record(  # noqa: SLF001
                tok,
                WordStateImprint(F_ema=0.1, R=0.1, m=0.0, u_curiosity=0.0, u_task=0.0),
            )
        agent._word_memory.record(  # noqa: SLF001
            "热词",
            WordStateImprint(F_ema=0.1, R=0.1, m=0.0, u_curiosity=0.0, u_task=0.0),
        )
        agent._word_memory.note_trigger_success("热词")  # noqa: SLF001
        mono = agent.internal_tick()
        self.assertIsNotNone(mono)
        self.assertIn(mono.trigger_token, {"热词", "冷1", "冷2", "冷3", "冷4"})

    def test_associative_probe_drafts_use_associated_tokens(self) -> None:
        agent = CognitiveDialogueAgent(self.tokenizer, learn_tokenizer_from_user=False)
        drafts = agent._build_associative_probe_drafts(  # noqa: SLF001
            "跑路",
            [("群", 2), ("封禁", 1)],
            replan_level=0,
        )
        self.assertGreater(len(drafts), 0)
        self.assertTrue(
            any("跑路" in d and "群" in d for d, _assoc_pick in drafts)
        )

    def test_associative_probe_can_win_efe_competition(self) -> None:
        agent = CognitiveDialogueAgent(self.tokenizer, learn_tokenizer_from_user=False)
        agent._word_memory.record(  # noqa: SLF001
            "跑路",
            WordStateImprint(1, 1, 0, 0.1, 0.1, "群", "封禁"),
        )
        intent = agent._germinate_intent("跑路", 0.0)  # noqa: SLF001
        plan = agent._sparse_plan(intent)  # noqa: SLF001
        snap = agent._freeze_resource()  # noqa: SLF001

        def fake_efe(draft: str, **_kwargs: object) -> float:
            return -100.0 if "群" in draft else 100.0

        agent._expected_free_energy = fake_efe  # type: ignore[method-assign]  # noqa: SLF001
        _draft, _eps, branch, _replans, _efe, _a_tr, _a_pk = agent._efe_best_reply(  # noqa: SLF001
            intent=intent,
            plan=plan,
            pi_s=agent.pi_statement(),
            user_text="跑路",
            eps_in=0.0,
            listen_resource_snap=snap,
            max_surp=1.0,
            focus_tok="跑路",
            internal_monologue=True,
        )
        self.assertEqual(branch, "associative_probe")

    def test_internal_tick_can_select_associative_probe_from_associated_frontier(self) -> None:
        agent = CognitiveDialogueAgent(
            self.tokenizer,
            learn_tokenizer_from_user=False,
            internal_tension_threshold=0.0,
            internal_global_scan_k=4,
            internal_spontaneous_jitter=0.0,
            internal_max_efe_to_speak=500.0,
        )
        agent._word_memory.record(  # noqa: SLF001
            "群",
            WordStateImprint(1, 1, 0, 0.1, 0.1, "", "跑路"),
        )
        agent._word_memory.record(  # noqa: SLF001
            "跑路",
            WordStateImprint(1, 1, 0, 0.1, 0.1, "群", "封禁"),
        )
        agent._recent_attention.append("群")  # noqa: SLF001

        def fake_efe(draft: str, **_kwargs: object) -> float:
            return -100.0 if "封禁" in draft else 100.0

        agent._expected_free_energy = fake_efe  # type: ignore[method-assign]  # noqa: SLF001
        mono = agent.internal_tick()
        self.assertIsNotNone(mono)
        self.assertEqual(mono.trigger_token, "跑路")
        self.assertEqual(mono.output_branch, "associative_probe")

    def test_internal_monologue_efe_skips_social_alignment_penalty(self) -> None:
        t = PrecisionTokenizer()
        t.fit(["你好"])
        agent = CognitiveDialogueAgent(t, learn_tokenizer_from_user=False)
        agent.hear("你好")
        draft = "罕见词ZZZZ" * 12
        uc = agent.u_curiosity()
        # ε_social 取小使 κ·ε_social < ε_secondary，外部稿承担对齐超额项；内心独白该项归零。
        g_ext = agent._expected_free_energy(  # noqa: SLF001
            draft,
            eps_social=0.05,
            u_curiosity=uc,
            is_explore=False,
            imprint_surprise=0.0,
            internal_monologue=False,
        )
        g_int = agent._expected_free_energy(  # noqa: SLF001
            draft,
            eps_social=0.05,
            u_curiosity=uc,
            is_explore=False,
            imprint_surprise=0.0,
            internal_monologue=True,
        )
        self.assertLess(g_int, g_ext)

    def test_expected_free_energy_computes_secondary_once(self) -> None:
        t = PrecisionTokenizer()
        t.fit(["你好"])
        agent = CognitiveDialogueAgent(t, learn_tokenizer_from_user=False)
        calls = {"n": 0}

        def fake_eps(_draft: str) -> float:
            calls["n"] += 1
            return 0.5

        agent.epsilon_secondary_on_draft = fake_eps  # type: ignore[method-assign]
        _ = agent._expected_free_energy(  # noqa: SLF001
            "测试草稿",
            eps_social=0.2,
            u_curiosity=0.4,
            is_explore=True,
            imprint_surprise=0.1,
            internal_monologue=False,
        )
        self.assertEqual(calls["n"], 1)

    def test_arbitration_ticket_scales_with_epsilon_social(self) -> None:
        t = PrecisionTokenizer()
        t.fit(["你好"])
        agent = CognitiveDialogueAgent(t, learn_tokenizer_from_user=False)
        self.assertGreater(
            agent.social_arbitration_ticket(0.5),
            agent.social_arbitration_ticket(0.1),
        )

    def test_arbitration_external_first_on_tie_or_lower_internal(self) -> None:
        t = PrecisionTokenizer()
        t.fit(["你好"])
        agent = CognitiveDialogueAgent(t, learn_tokenizer_from_user=False)
        intent = CommunicativeIntent(
            speech_act="respond",
            anchor_slot="sigma_slot_1",
            tension_slot="sigma_slot_2",
            u_curiosity=0.5,
            u_task=0.3,
        )
        turn = DialogueTurn(
            user_text="x",
            reply="r",
            u_curiosity=0.5,
            u_task=0.3,
            pi_statement=0.5,
            epsilon_social_in=0.2,
            epsilon_secondary=0.1,
            intent=intent,
            replan_count=0,
            output_branch="conservative",
            best_efe=1.0,
        )
        mono_lo = InternalMonologue(
            reply="i",
            trigger_token="t",
            tension=1.0,
            best_efe=0.5,
            output_branch="conservative",
            epsilon_secondary=0.1,
            semantic_surprise_max=0.0,
            u_curiosity=0.5,
        )
        self.assertFalse(agent.arbitration_external_first(mono_lo, turn))
        mono_hi = InternalMonologue(
            reply="i",
            trigger_token="t",
            tension=1.0,
            best_efe=10.0,
            output_branch="conservative",
            epsilon_secondary=0.1,
            semantic_surprise_max=0.0,
            u_curiosity=0.5,
        )
        self.assertTrue(agent.arbitration_external_first(mono_hi, turn))

    def test_unknown_hedge_draft_gets_extra_efe_penalty(self) -> None:
        t = PrecisionTokenizer()
        t.fit(["你好"])
        agent = CognitiveDialogueAgent(t, learn_tokenizer_from_user=False)

        def fixed_eps(_draft: str) -> float:
            return 0.35

        agent.epsilon_secondary_on_draft = fixed_eps  # type: ignore[method-assign]
        kwargs = dict(
            eps_social=0.2,
            u_curiosity=0.4,
            is_explore=False,
            imprint_surprise=0.0,
            internal_monologue=False,
        )
        g_plain = agent._expected_free_energy("说明一下", **kwargs)  # noqa: SLF001
        g_unk = agent._expected_free_energy("我不知道，说明一下", **kwargs)  # noqa: SLF001
        self.assertGreater(g_unk - g_plain, float(agent.EFE_W_UNKNOWN_HEDGE) - 1e-6)

    def test_external_turn_decays_internal_pending(self) -> None:
        agent = CognitiveDialogueAgent(self.tokenizer, learn_tokenizer_from_user=False)
        agent._internal_pending["你好"] = 1.0  # noqa: SLF001
        agent.turn("今天天气")
        self.assertLess(agent._internal_pending.get("你好", 0.0), 1.0)  # noqa: SLF001

    def test_internal_tick_train_fast_accumulates_pending_without_emit(self) -> None:
        agent = CognitiveDialogueAgent(
            self.tokenizer,
            learn_tokenizer_from_user=False,
            internal_tension_threshold=0.0,
            internal_global_scan_k=4,
            internal_spontaneous_jitter=0.0,
        )
        agent._word_memory.record(  # noqa: SLF001
            "旧事",
            WordStateImprint(F_ema=0.1, R=0.1, m=0.0, u_curiosity=0.0, u_task=0.0),
        )
        self.assertIsNone(agent.internal_tick_train_fast())
        self.assertGreater(agent._internal_pending.get("旧事", 0.0), 0.0)  # noqa: SLF001

    def test_internal_tick_train_fast_decays_existing_pending(self) -> None:
        agent = CognitiveDialogueAgent(
            self.tokenizer,
            learn_tokenizer_from_user=False,
            internal_tension_threshold=10.0,
            internal_global_scan_k=0,
        )
        agent._internal_pending["旧事"] = 1.0  # noqa: SLF001
        self.assertIsNone(agent.internal_tick_train_fast())
        self.assertLess(agent._internal_pending.get("旧事", 0.0), 1.0)  # noqa: SLF001

    def test_internal_tick_train_fast_many_matches_repeated_calls(self) -> None:
        agent0 = CognitiveDialogueAgent(
            self.tokenizer,
            learn_tokenizer_from_user=False,
            internal_tension_threshold=0.0,
            internal_global_scan_k=4,
            internal_spontaneous_jitter=0.0,
        )
        agent1 = CognitiveDialogueAgent(
            PrecisionTokenizer.from_dict(self.tokenizer.to_dict()),
            learn_tokenizer_from_user=False,
            internal_tension_threshold=0.0,
            internal_global_scan_k=4,
            internal_spontaneous_jitter=0.0,
        )
        for agent in (agent0, agent1):
            agent._word_memory.record(  # noqa: SLF001
                "旧事",
                WordStateImprint(F_ema=0.1, R=0.1, m=0.0, u_curiosity=0.0, u_task=0.0),
            )
        for _ in range(3):
            self.assertIsNone(agent0.internal_tick_train_fast())
        self.assertIsNone(agent1.internal_tick_train_fast_many(3))
        self.assertEqual(agent0._internal_pending.keys(), agent1._internal_pending.keys())  # noqa: SLF001
        for key in agent0._internal_pending:  # noqa: SLF001
            self.assertAlmostEqual(
                agent0._internal_pending[key],  # noqa: SLF001
                agent1._internal_pending[key],  # noqa: SLF001
                places=6,
            )

    def test_recent_attention_evicts_when_over_capacity(self) -> None:
        agent = CognitiveDialogueAgent(self.tokenizer, recent_attention_capacity=3)
        agent.turn("你好")
        agent.turn("今天")
        agent.turn("天气")
        self.assertLessEqual(len(agent.recent_attention_words), 3)

    def test_turn_produces_non_empty_reply(self) -> None:
        turn = self.agent.turn("你好今天")
        self.assertGreater(len(turn.reply), 0)
        self.assertGreaterEqual(turn.curiosity_u, 0.0)
        self.assertLessEqual(turn.pi_statement, 1.0)

    def test_question_triggers_question_branch(self) -> None:
        turn = self.agent.turn("你觉得分词对吗？")
        self.assertTrue("关于" in turn.reply or "惊奇" in turn.reply)

    def test_hear_without_tokenizer_learn_skips_ingest(self) -> None:
        t = PrecisionTokenizer()
        t.fit(["你好"])
        agent = CognitiveDialogueAgent(t, learn_tokenizer_from_user=False)
        before = t.total_chars
        agent.hear("罕见字Z")
        self.assertEqual(t.total_chars, before)
        self.assertEqual(t.unigram.get("Z", 0), 0)

    def test_hear_with_learn_updates_unigram(self) -> None:
        t = PrecisionTokenizer()
        t.fit(["你好"])
        agent = CognitiveDialogueAgent(t, learn_tokenizer_from_user=True)
        self.assertEqual(t.unigram.get("Z", 0), 0)
        agent.hear("罕见字Z")
        self.assertGreater(t.unigram.get("Z", 0), 0)

    def test_dialogue_model_save_load_roundtrip_sigma(self) -> None:
        t = PrecisionTokenizer()
        t.fit(["hello world"])
        a0 = CognitiveDialogueAgent(t, learn_tokenizer_from_user=False)
        a0.turn("something surprising " * 5)
        sig_after = dict(a0._sigma)  # noqa: SLF001
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "d.dialogue.json"
            a0.save_dialogue_model(str(p))
            raw = json.loads(p.read_text(encoding="utf-8"))
            self.assertEqual(raw["format"], DIALOGUE_MODEL_FORMAT)
            t2 = PrecisionTokenizer()
            t2.fit(["hello world"])
            a1 = CognitiveDialogueAgent(t2, learn_tokenizer_from_user=True)
            a1.load_dialogue_model(str(p))
            self.assertEqual(a1._sigma, sig_after)  # noqa: SLF001

    def test_dialogue_model_compact_save_load_roundtrip(self) -> None:
        t = PrecisionTokenizer()
        t.fit(["hello"])
        a0 = CognitiveDialogueAgent(t, learn_tokenizer_from_user=False)
        a0.turn("something")
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "d.compact.dialogue.json"
            a0.save_dialogue_model(str(p), compact=True)
            raw = p.read_text(encoding="utf-8")
            self.assertNotIn("\n  ", raw)
            t2 = PrecisionTokenizer()
            t2.fit(["hello"])
            a1 = CognitiveDialogueAgent(t2, learn_tokenizer_from_user=False)
            a1.load_dialogue_model(str(p))
            self.assertEqual(a1.sigma_state(), a0.sigma_state())

    def test_dialogue_model_load_migrates_legacy_sigma_keys(self) -> None:
        """旧版 sigma 使用中文占位键名时应映射到当前 SLOT_NAMES。"""
        t = PrecisionTokenizer()
        t.fit(["hello"])
        agent = CognitiveDialogueAgent(t, learn_tokenizer_from_user=False)
        legacy = {
            "format": DIALOGUE_MODEL_FORMAT,
            "sigma": {"小明": 0.1, "游戏": 0.2, "类型": 0.3, "偏好": 0.4},
            "word_imprints": agent._word_memory.to_dict(),  # noqa: SLF001
            "preference_state": agent._preference_state_to_dict(),  # noqa: SLF001
        }
        agent.apply_dialogue_model_dict(cast(Dict[str, object], legacy))
        self.assertEqual(
            agent.sigma_state(),
            {
                "sigma_slot_1": 0.1,
                "sigma_slot_2": 0.2,
                "sigma_slot_3": 0.3,
                "sigma_slot_4": 0.4,
            },
        )

    def test_dialogue_model_to_dict_includes_format_version(self) -> None:
        t = PrecisionTokenizer()
        t.fit(["hello"])
        agent = CognitiveDialogueAgent(t, learn_tokenizer_from_user=False)
        d = agent.dialogue_model_to_dict()
        self.assertEqual(d.get("format_version"), int(DIALOGUE_MODEL_FORMAT_VERSION))

    def test_strict_load_without_preference_state_resets_preferences(self) -> None:
        """省略 preference_state 时显式默认空偏好（不保留内存旧偏置）。"""
        t = PrecisionTokenizer()
        t.fit(["a b"])
        agent = CognitiveDialogueAgent(t, learn_tokenizer_from_user=False)
        agent._branch_bias["conservative"] = 0.42
        agent._curiosity_injection_pool = 0.1
        agent.apply_dialogue_model_dict(
            cast(
                Dict[str, object],
                {
                    "format": DIALOGUE_MODEL_FORMAT,
                    "sigma": {k: 0.33 for k in agent.SLOT_NAMES},
                    "word_imprints": agent._word_memory.to_dict(),
                },
            )
        )
        self.assertEqual(agent._branch_bias, {})
        self.assertAlmostEqual(
            agent._curiosity_injection_pool, agent.curiosity_injection_pool_max
        )

    def test_patch_dialogue_model_dict_partial_sigma(self) -> None:
        t = PrecisionTokenizer()
        t.fit(["x y"])
        agent = CognitiveDialogueAgent(t, learn_tokenizer_from_user=False)
        for k in agent.SLOT_NAMES:
            agent._sigma[k] = 0.4
        agent.patch_dialogue_model_dict({"sigma": {"sigma_slot_1": 0.91}})
        self.assertAlmostEqual(agent._sigma["sigma_slot_1"], 0.91)
        self.assertAlmostEqual(agent._sigma["sigma_slot_2"], 0.4)

    def test_from_tokenizer_path_loads_dialogue_model(self) -> None:
        t = PrecisionTokenizer()
        t.fit(["你好"])
        a0 = CognitiveDialogueAgent(t, learn_tokenizer_from_user=False)
        a0.turn("今天天气如何？" * 3)
        with tempfile.TemporaryDirectory() as td:
            tp = Path(td) / "tok.json"
            dp = Path(td) / "tok.dialogue.json"
            t.save_model(str(tp))
            a0.save_dialogue_model(str(dp))
            a1 = CognitiveDialogueAgent.from_tokenizer_path(
                str(tp),
                learn_tokenizer_from_user=True,
                dialogue_model_path=str(dp),
            )
            self.assertEqual(a1._sigma, a0._sigma)  # noqa: SLF001

    def test_output_branch_explore_when_resources_and_curiosity_high(self) -> None:
        t = PrecisionTokenizer()
        t.fit(["hello world", "hello friend"])
        agent = CognitiveDialogueAgent(
            t,
            learn_tokenizer_from_user=False,
            lambda_explore_0=2.5,
            social_risk_alpha=0.02,
        )
        for k in agent._sigma:  # noqa: SLF001
            agent._sigma[k] = 1.0  # noqa: SLF001
        t.R = float(t.R_max)
        t.m = 0.0
        turn = agent.turn("hello")
        self.assertEqual(turn.output_branch, "explore_echo")
        self.assertIn("回声", turn.reply)

    def test_hedge_skips_unknown_when_dialogue_pi_above_threshold(self) -> None:
        """R_max 语料尺度大、R 中等时：原始 π 仍低，但 hedge 用对话尺度，保守稿可不以「我不知道」起头。"""
        t = PrecisionTokenizer()
        t.fit(["你好 世界", "你好"])
        t.R_max = 2.4
        t.R = 0.2
        t.m = 0.05
        agent = CognitiveDialogueAgent(t, learn_tokenizer_from_user=False)
        turn = agent.turn("你好")
        self.assertLess(turn.pi_statement, 0.32)
        self.assertFalse(
            turn.reply.startswith("我不知道"),
            msg=f"reply[:50]={turn.reply[:50]!r}",
        )

    def test_resource_factor_not_starved_when_R_max_large(self) -> None:
        """大 R_max 时仍应用对话尺度分母，避免 R/R_max 假阴性。"""
        t = PrecisionTokenizer()
        t.fit(["ab", "bc"])
        t.R_max = 2.5
        t.R = 0.2
        t.m = 0.05
        agent = CognitiveDialogueAgent(t, learn_tokenizer_from_user=False)
        rf = agent._resource_factor()  # noqa: SLF001
        self.assertGreater(rf, 0.35)

    def test_efe_invariant_to_prior_candidate_tokenize(self) -> None:
        """构造其它候选时的 tokenize(user) 不得残留 R/m，以免扭曲后续 EFE（倾听快照回归）。"""
        t = PrecisionTokenizer()
        t.fit(["你好 世界", "你好 朋友"])
        t.auto_rest_threshold = -1.0
        agent = CognitiveDialogueAgent(
            t,
            learn_tokenizer_from_user=False,
            lambda_explore_0=0.0,
            social_risk_alpha=30.0,
        )
        for k in agent._sigma:  # noqa: SLF001
            agent._sigma[k] = 1.0  # noqa: SLF001
        t.R = float(t.R_max) * 0.02
        t.m = 0.95
        eps = agent.hear("你好")
        intent = agent._germinate_intent("你好", eps)
        plan_d = agent._plan_at_depth(agent._sparse_plan(intent), 0)
        pi_s = agent.pi_statement()
        uc = float(intent.u_curiosity)
        snap = agent._freeze_resource()  # noqa: SLF001
        ex = agent._build_explore_echo_draft(
            intent, plan_d, pi_s, eps, "你好", 0, hedge_prefix=""
        )
        agent._restore_resource(snap)  # noqa: SLF001
        g0 = agent._expected_free_energy(
            ex,
            eps_social=float(eps),
            u_curiosity=uc,
            is_explore=True,
            imprint_surprise=0.0,
        )
        agent._restore_resource(snap)  # noqa: SLF001
        _ = agent._realize(
            intent, plan_d, pi_s, eps, "你好", 0, hedge_prefix=""
        )
        agent._restore_resource(snap)  # noqa: SLF001
        g1 = agent._expected_free_energy(
            ex,
            eps_social=float(eps),
            u_curiosity=uc,
            is_explore=True,
            imprint_surprise=0.0,
        )
        self.assertAlmostEqual(g0, g1, places=5)

    def test_output_branch_conservative_when_resources_depleted(self) -> None:
        t = PrecisionTokenizer()
        t.fit(["你好 世界", "你好 朋友"])
        # 避免 trace 内 auto_rest 把 R 抬上去，否则「枯竭」断言不稳定
        t.auto_rest_threshold = -1.0
        agent = CognitiveDialogueAgent(
            t,
            learn_tokenizer_from_user=False,
            lambda_explore_0=0.0,
            social_risk_alpha=30.0,
        )
        for k in agent._sigma:  # noqa: SLF001
            agent._sigma[k] = 1.0  # noqa: SLF001
        t.R = float(t.R_max) * 0.02
        t.m = 0.95
        turn = agent.turn("你好")
        self.assertEqual(turn.output_branch, "conservative")

    def test_train_step_batch_matches_repeated_train_step(self) -> None:
        texts = ["群主跑路了", "真的假的", "今晚考试"]
        tok_a = PrecisionTokenizer()
        tok_b = PrecisionTokenizer()
        tok_a.fit(["你好 世界", "今天 天气"])
        tok_b.fit(["你好 世界", "今天 天气"])
        agent_a = CognitiveDialogueAgent(tok_a, learn_tokenizer_from_user=False)
        agent_b = CognitiveDialogueAgent(tok_b, learn_tokenizer_from_user=False)

        expected = [agent_a.train_step(text) for text in texts]
        got = agent_b.train_step_batch(texts)

        self.assertEqual(len(got), len(expected))
        for lhs, rhs in zip(got, expected, strict=True):
            self.assertEqual(lhs["output_branch"], rhs["output_branch"])
            self.assertAlmostEqual(lhs["epsilon_social_in"], rhs["epsilon_social_in"], places=6)
            self.assertAlmostEqual(lhs["semantic_surprise_max"], rhs["semantic_surprise_max"], places=6)
            self.assertEqual(lhs["conflict_focus_token"], rhs["conflict_focus_token"])

        self.assertEqual(agent_a.sigma_state(), agent_b.sigma_state())
        self.assertEqual(agent_a.recent_attention_words, agent_b.recent_attention_words)
        self.assertEqual(agent_a._word_memory.to_dict(), agent_b._word_memory.to_dict())  # noqa: SLF001


if __name__ == "__main__":
    unittest.main()
