"""保守分支负反馈：共享预算下的 pending / 认知债务 / tension_slot σ 注入。"""
from __future__ import annotations

import unittest

from uva_model.dialogue import (
    CognitiveDialogueAgent,
    CommunicativeIntent,
    DialogueTurn,
)
from uva_model.tokenizer import PrecisionTokenizer
from uva_model.word_imprints import WordStateImprint, WordStateMemory


class DialogueCuriosityInjectionTests(unittest.TestCase):
    def test_conservative_bad_injects_pending_debt_sigma(self) -> None:
        tok = PrecisionTokenizer()
        tok.fit(["你好"])
        agent = CognitiveDialogueAgent(
            tok,
            learn_tokenizer_from_user=False,
            curiosity_injection_per_event_cap=0.9,
            curiosity_injection_pool_max=1.0,
            curiosity_injection_pool_refill_external=0.0,
            curiosity_injection_w_pending=1.0,
            curiosity_injection_w_debt=1.0,
            curiosity_injection_w_sigma=1.0,
        )
        intent = CommunicativeIntent(
            speech_act="clarify",
            anchor_slot="sigma_slot_1",
            tension_slot="sigma_slot_2",
            u_curiosity=0.5,
            u_task=0.3,
        )
        turn = DialogueTurn(
            user_text="u",
            reply="r",
            u_curiosity=0.5,
            u_task=0.3,
            pi_statement=0.5,
            epsilon_social_in=0.1,
            epsilon_secondary=0.2,
            intent=intent,
            replan_count=0,
            output_branch="conservative",
            conflict_focus_token="你好",
        )
        sig0 = float(agent._sigma["sigma_slot_2"])
        pool0 = float(agent._curiosity_injection_pool)
        agent.apply_dialogue_feedback(-1.0, turn)
        self.assertGreater(agent._internal_pending.get("你好", 0.0), 0.0)
        self.assertGreater(agent._word_memory._cognitive_debt.get("你好", 0.0), 0.0)
        self.assertGreater(float(agent._sigma["sigma_slot_2"]), sig0)
        self.assertLess(agent._curiosity_injection_pool, pool0)

    def test_non_conservative_bad_skips_injection(self) -> None:
        tok = PrecisionTokenizer()
        tok.fit(["你好"])
        agent = CognitiveDialogueAgent(
            tok,
            learn_tokenizer_from_user=False,
            curiosity_injection_per_event_cap=0.9,
        )
        intent = CommunicativeIntent(
            speech_act="explore",
            anchor_slot="sigma_slot_1",
            tension_slot="sigma_slot_2",
            u_curiosity=0.6,
            u_task=0.4,
        )
        turn = DialogueTurn(
            user_text="u",
            reply="r",
            u_curiosity=0.6,
            u_task=0.4,
            pi_statement=0.5,
            epsilon_social_in=0.1,
            epsilon_secondary=0.2,
            intent=intent,
            replan_count=0,
            output_branch="explore_echo",
            conflict_focus_token="你好",
        )
        agent.apply_dialogue_feedback(-1.0, turn)
        self.assertEqual(agent._internal_pending.get("你好", 0.0), 0.0)
        self.assertEqual(agent._word_memory._cognitive_debt.get("你好", 0.0), 0.0)

    def test_cognitive_debt_decay_on_tick_only_path(self) -> None:
        mem = WordStateMemory(capacity=10)
        mem.inject_cognitive_debt("词", 0.8)
        mem.decay_cognitive_debt_tick(0.5)
        self.assertAlmostEqual(mem._cognitive_debt.get("词", 0.0), 0.4, places=5)

    def test_semantic_surprise_scales_with_debt(self) -> None:
        mem = WordStateMemory(capacity=10, cognitive_debt_surprise_gamma=1.0)
        mem.record(
            "tok",
            WordStateImprint(1.0, 0.5, 0.1, 0.2, 0.2, "a", "b"),
        )
        cur = WordStateImprint(0.5, 0.6, 0.2, 0.3, 0.1, "x", "y")
        base = mem.semantic_surprise_for_token("tok", cur)
        self.assertGreater(base, 0.0)
        mem.inject_cognitive_debt("tok", 1.0)
        boosted = mem.semantic_surprise_for_token("tok", cur)
        self.assertGreater(boosted, base)
        self.assertAlmostEqual(boosted, base * 2.0, places=5)


if __name__ == "__main__":
    unittest.main()
