import json
import tempfile
import unittest
from pathlib import Path
from typing import cast

from uva_model.dialogue import (
    CognitiveDialogueAgent,
    CommunicativeIntent,
    DialogueTurn,
    InternalMonologue,
    PREFERENCE_STATE_FORMAT,
)
from uva_model.tokenizer import PrecisionTokenizer
from uva_model.word_imprints import WordStateImprint


class DialoguePreferenceTests(unittest.TestCase):
    def test_apply_dialogue_feedback_updates_branch_token_and_assoc(self) -> None:
        tok = PrecisionTokenizer()
        tok.fit(["你好 世界"])
        agent = CognitiveDialogueAgent(tok)
        intent = CommunicativeIntent(
            speech_act="explore",
            anchor_slot="sigma_slot_1",
            tension_slot="sigma_slot_2",
            u_curiosity=0.5,
            u_task=0.3,
        )
        turn_echo = DialogueTurn(
            user_text="hi",
            reply="x",
            u_curiosity=0.5,
            u_task=0.3,
            pi_statement=0.5,
            epsilon_social_in=0.1,
            epsilon_secondary=0.2,
            intent=intent,
            replan_count=0,
            output_branch="explore_echo",
            conflict_focus_token="焦点",
        )
        agent.apply_dialogue_feedback(1.0, turn_echo)
        self.assertGreater(agent._branch_bias.get("explore_echo", 0.0), 0.0)
        self.assertGreater(agent._token_value.get("焦点", 0.0), 0.0)

        turn_assoc = DialogueTurn(
            user_text="hi",
            reply="y",
            u_curiosity=0.5,
            u_task=0.3,
            pi_statement=0.5,
            epsilon_social_in=0.1,
            epsilon_secondary=0.2,
            intent=intent,
            replan_count=0,
            output_branch="associative_probe",
            association_trigger="跑路",
            association_pick="群",
        )
        agent.apply_dialogue_feedback(1.0, turn_assoc)
        self.assertGreater(agent._association_value["跑路"]["群"], 0.0)

        agent.apply_dialogue_feedback(-1.0, turn_echo)
        self.assertLess(agent._branch_bias.get("explore_echo", 0.0), 0.0)

    def test_preference_values_stay_within_clamp_after_many_updates(self) -> None:
        tok = PrecisionTokenizer()
        tok.fit(["你好"])
        agent = CognitiveDialogueAgent(tok)
        intent = CommunicativeIntent(
            speech_act="clarify",
            anchor_slot="sigma_slot_1",
            tension_slot="sigma_slot_2",
            u_curiosity=0.5,
            u_task=0.3,
        )
        turn = DialogueTurn(
            user_text="x",
            reply="y",
            u_curiosity=0.5,
            u_task=0.3,
            pi_statement=0.5,
            epsilon_social_in=0.1,
            epsilon_secondary=0.2,
            intent=intent,
            replan_count=0,
            output_branch="conservative",
        )
        vmax = agent.preference_value_max
        for _ in range(80):
            agent.apply_dialogue_feedback(1.0, turn)
        bb = agent._branch_bias.get("conservative", 0.0)
        self.assertLessEqual(bb, vmax + 1e-6)
        self.assertGreaterEqual(bb, -vmax - 1e-6)

    def test_internal_monologue_feedback_updates_trigger_token(self) -> None:
        tok = PrecisionTokenizer()
        tok.fit(["你好"])
        agent = CognitiveDialogueAgent(tok)
        mono = InternalMonologue(
            reply="…",
            trigger_token="念头词",
            tension=1.0,
            best_efe=1.0,
            output_branch="conflict_probe",
            epsilon_secondary=0.1,
            semantic_surprise_max=0.2,
            u_curiosity=0.5,
        )
        agent.apply_internal_monologue_feedback(1.0, mono)
        self.assertGreater(agent._token_value.get("念头词", 0.0), 0.0)

    def test_dialogue_model_roundtrip_preserves_preference_state(self) -> None:
        tok = PrecisionTokenizer()
        tok.fit(["你好"])
        agent = CognitiveDialogueAgent(tok)
        intent = CommunicativeIntent(
            speech_act="respond",
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
            epsilon_social_in=0.2,
            epsilon_secondary=0.3,
            intent=intent,
            replan_count=0,
            output_branch="associative_probe",
            association_trigger="甲",
            association_pick="乙",
        )
        agent.apply_dialogue_feedback(1.0, turn)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "d.json"
            agent.save_dialogue_model(str(path))
            raw_saved = json.loads(path.read_text(encoding="utf-8"))
            agent2 = CognitiveDialogueAgent(tok)
            agent2.load_dialogue_model(str(path))

        ps = raw_saved.get("preference_state")
        self.assertIsInstance(ps, dict)
        ps_d = cast(dict, ps)
        self.assertEqual(ps_d.get("format"), PREFERENCE_STATE_FORMAT)
        bb = cast(dict, ps_d["branch_bias"])
        self.assertGreater(float(bb["associative_probe"]), 0.0)
        self.assertGreater(agent2._association_value["甲"]["乙"], 0.0)

    def test_curiosity_injection_pool_roundtrip(self) -> None:
        tok = PrecisionTokenizer()
        tok.fit(["你好"])
        agent = CognitiveDialogueAgent(
            tok,
            learn_tokenizer_from_user=False,
            curiosity_injection_pool_refill_external=0.0,
            curiosity_injection_per_event_cap=0.5,
        )
        intent = CommunicativeIntent(
            speech_act="clarify",
            anchor_slot="sigma_slot_1",
            tension_slot="sigma_slot_3",
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
        agent.apply_dialogue_feedback(-1.0, turn)
        expected = float(agent._curiosity_injection_pool)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "d.json"
            agent.save_dialogue_model(str(path))
            raw_saved = json.loads(path.read_text(encoding="utf-8"))
            agent2 = CognitiveDialogueAgent(tok, learn_tokenizer_from_user=False)
            agent2.load_dialogue_model(str(path))
        ps = raw_saved.get("preference_state")
        self.assertIsInstance(ps, dict)
        self.assertIn("curiosity_injection_pool", cast(dict, ps))
        self.assertAlmostEqual(float(agent2._curiosity_injection_pool), expected, places=5)

    def test_biased_associated_tokens_prefers_learned_boost(self) -> None:
        tok = PrecisionTokenizer()
        tok.fit(["你好"])
        agent = CognitiveDialogueAgent(tok)
        stub_b = WordStateImprint(1.0, 0.5, 0.1, 0.2, 0.2, "", "乙")
        stub_c = WordStateImprint(1.0, 0.5, 0.1, 0.2, 0.2, "", "丙")
        agent._word_memory.record("甲", stub_b)
        agent._word_memory.record("甲", stub_c)
        raw_order = agent._word_memory.associated_tokens("甲", k=5)
        self.assertEqual(len(raw_order), 2)
        agent._association_value.setdefault("甲", {})["丙"] = agent.preference_value_max
        biased = agent._biased_associated_tokens("甲", k=2)
        self.assertEqual([x[0] for x in biased], ["丙", "乙"])


if __name__ == "__main__":
    unittest.main()
