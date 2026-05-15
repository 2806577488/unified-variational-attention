"""C 方案自动反馈：信号推断与 ``CognitiveDialogueAgent.turn`` 集成。"""

from __future__ import annotations

import unittest

from uva_model.auto_feedback import (
    AutoFeedbackConfig,
    combine_rewards,
    infer_acceptance_feedback,
    infer_auto_feedback,
    infer_avoidance_feedback,
    infer_continuation_feedback,
)
from uva_model.dialogue import CognitiveDialogueAgent
from uva_model.tokenizer import PrecisionTokenizer
from uva_model.word_imprints import WordStateImprint


class AutoFeedbackSignalTests(unittest.TestCase):
    def test_acceptance_on_focus_token(self) -> None:
        r = infer_acceptance_feedback("跑路", "", "你说跑路是什么意思", reward_strength=0.5)
        self.assertEqual(r, 0.5)

    def test_avoidance_short_reply_after_probe(self) -> None:
        r = infer_avoidance_feedback(
            "conflict_probe",
            "跑路",
            "嗯",
            evasion_max_length=8,
            reward_strength=-0.2,
        )
        self.assertEqual(r, -0.2)

    def test_avoidance_not_triggered_when_focus_mentioned(self) -> None:
        r = infer_avoidance_feedback(
            "conflict_probe",
            "跑路",
            "跑路？",
            evasion_max_length=8,
            reward_strength=-0.2,
        )
        self.assertIsNone(r)

    def test_continuation_saturates(self) -> None:
        r3 = infer_continuation_feedback(3, min_turns=3, base_reward=0.3)
        r8 = infer_continuation_feedback(8, min_turns=3, base_reward=0.3)
        self.assertIsNotNone(r3)
        self.assertIsNotNone(r8)
        assert r3 is not None and r8 is not None
        self.assertAlmostEqual(r8, 0.3, places=6)
        self.assertLess(r3, r8)

    def test_combine_mean(self) -> None:
        out = combine_rewards([0.5, -0.2], mode="mean")
        self.assertAlmostEqual(out or 0.0, 0.15, places=6)


class AutoFeedbackAgentIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        tok = PrecisionTokenizer()
        tok.fit(["你好 世界", "跑路 游戏", "你好 朋友"])
        cfg = AutoFeedbackConfig(
            enabled=True,
            acceptance_reward=0.4,
            avoidance_reward=-0.2,
            continuation_min_turns=100,
            continuation_reward_base=0.3,
        )
        self.agent = CognitiveDialogueAgent(tok, learn_tokenizer_from_user=False, auto_feedback_config=cfg)

    def test_turn_applies_acceptance_to_previous_branch_bias(self) -> None:
        from uva_model.dialogue import CommunicativeIntent, DialogueTurn

        intent = CommunicativeIntent(
            speech_act="respond",
            anchor_slot="sigma_slot_1",
            tension_slot="sigma_slot_2",
            u_curiosity=0.2,
            u_task=0.1,
        )
        self.agent._last_turn = DialogueTurn(  # noqa: SLF001
            user_text="",
            reply="",
            u_curiosity=0.2,
            u_task=0.1,
            pi_statement=0.5,
            epsilon_social_in=0.1,
            epsilon_secondary=0.1,
            intent=intent,
            replan_count=0,
            output_branch="conflict_probe",
            conflict_focus_token="跑路",
        )
        bias_before = float(self.agent._branch_bias.get("conflict_probe", 0.0))  # noqa: SLF001
        self.agent.turn("跑路怎么回事")
        bias_after = float(self.agent._branch_bias.get("conflict_probe", 0.0))  # noqa: SLF001
        self.assertGreater(bias_after, bias_before)

    def test_produce_turn_does_not_run_auto_feedback(self) -> None:
        self.agent._last_turn = None  # noqa: SLF001
        bb0 = dict(self.agent._branch_bias)  # noqa: SLF001
        self.agent.produce_turn("你好")
        self.agent.produce_turn("跑路")
        self.assertEqual(self.agent._branch_bias, bb0)  # noqa: SLF001

    def test_preference_state_roundtrip_session(self) -> None:
        self.agent.turn("第一句")
        snap = self.agent.dialogue_model_to_dict()  # noqa: SLF001
        ps = snap["preference_state"]
        self.assertIsInstance(ps, dict)
        afs = ps.get("auto_feedback_session")  # type: ignore[union-attr]
        self.assertIsInstance(afs, dict)
        self.assertEqual(afs.get("consecutive_dialogue_turns"), 1)  # type: ignore[union-attr]

        agent2 = CognitiveDialogueAgent(
            self.agent.tokenizer,
            learn_tokenizer_from_user=False,
        )
        agent2.apply_dialogue_model_dict(snap)  # type: ignore[arg-type]
        self.assertEqual(agent2._consecutive_dialogue_turns, 1)  # noqa: SLF001
        self.assertIsNotNone(agent2._last_turn)  # noqa: SLF001

    def test_reset_auto_feedback_session(self) -> None:
        self.agent.turn("a")
        self.agent.reset_auto_feedback_session()
        self.assertIsNone(self.agent._last_turn)  # noqa: SLF001
        self.assertEqual(self.agent._consecutive_dialogue_turns, 0)  # noqa: SLF001


class AutoFeedbackInferWrapperTests(unittest.TestCase):
    def test_infer_auto_feedback_disabled(self) -> None:
        cfg = AutoFeedbackConfig(enabled=False)
        self.assertIsNone(
            infer_auto_feedback(
                "conflict_probe",
                "跑路",
                "",
                "嗯",
                5,
                cfg,
            )
        )


if __name__ == "__main__":
    unittest.main()
