"""example_run 对话退出时写回分词模型与对话模型。"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from example_run import run_interactive_dialogue
from uva_model.dialogue import CognitiveDialogueAgent, DIALOGUE_MODEL_FORMAT
from uva_model.tokenizer import PrecisionTokenizer


class DialogueSaveOnExitTests(unittest.TestCase):
    def test_interactive_exit_writes_tokenizer_and_dialogue_json(self) -> None:
        tok = PrecisionTokenizer()
        tok.fit(["hello world", "good morning"])
        agent = CognitiveDialogueAgent(tok, learn_tokenizer_from_user=False)
        agent.turn("hello there")
        with tempfile.TemporaryDirectory() as td:
            out_tok = Path(td) / "tok.json"
            out_dlg = Path(td) / "tok.dialogue.json"

            def input_exit(_prompt: str = "") -> str:
                return "exit"

            run_interactive_dialogue(
                agent,
                input_fn=input_exit,
                save_tokenizer_path=str(out_tok),
                save_dialogue_path=str(out_dlg),
            )
            self.assertTrue(out_tok.is_file())
            self.assertTrue(out_dlg.is_file())
            loaded = PrecisionTokenizer.load_model(str(out_tok))
            self.assertTrue(loaded.fitted)
            d = json.loads(out_dlg.read_text(encoding="utf-8"))
            self.assertEqual(d.get("format"), DIALOGUE_MODEL_FORMAT)
            self.assertIn("sigma", d)

    def test_interactive_exit_can_skip_tokenizer_save(self) -> None:
        tok = PrecisionTokenizer()
        tok.fit(["hello world"])
        agent = CognitiveDialogueAgent(tok, learn_tokenizer_from_user=False)
        agent.turn("hello")
        with tempfile.TemporaryDirectory() as td:
            out_tok = Path(td) / "tok.json"
            out_dlg = Path(td) / "tok.dialogue.json"

            def input_exit(_prompt: str = "") -> str:
                return "exit"

            run_interactive_dialogue(
                agent,
                input_fn=input_exit,
                save_tokenizer_path="",
                save_dialogue_path=str(out_dlg),
            )
            self.assertFalse(out_tok.exists())
            self.assertTrue(out_dlg.is_file())


if __name__ == "__main__":
    unittest.main()
