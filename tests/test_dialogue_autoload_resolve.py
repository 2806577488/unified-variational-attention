"""resolve_dialogue_model_load_path：默认 sibling .dialogue.json 自动加载。"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from example_run import default_dialogue_model_path, resolve_dialogue_model_load_path
from uva_model.dialogue import CognitiveDialogueAgent
from uva_model.tokenizer import PrecisionTokenizer


class DialogueAutoloadResolveTests(unittest.TestCase):
    def test_no_file_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tp = Path(td) / "only_tok.json"
            tp.write_text("{}")
            self.assertEqual(resolve_dialogue_model_load_path(str(tp), ""), "")

    def test_sibling_dialogue_json_is_autoload_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tp = Path(td) / "tok.json"
            dp = Path(td) / "tok.dialogue.json"
            t = PrecisionTokenizer()
            t.fit(["ab"])
            t.save_model(str(tp))
            CognitiveDialogueAgent(t, learn_tokenizer_from_user=False).save_dialogue_model(str(dp))
            got = resolve_dialogue_model_load_path(str(tp), "")
            self.assertEqual(got, str(dp))
            self.assertEqual(default_dialogue_model_path(str(tp)), str(dp))

    def test_explicit_arg_wins_over_missing_sibling(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tp = Path(td) / "tok.json"
            explicit = Path(td) / "other.dialogue.json"
            t = PrecisionTokenizer()
            t.fit(["ab"])
            t.save_model(str(tp))
            CognitiveDialogueAgent(t, learn_tokenizer_from_user=False).save_dialogue_model(str(explicit))
            got = resolve_dialogue_model_load_path(str(tp), str(explicit))
            self.assertEqual(got, str(explicit))

    def test_dialogue_fresh_returns_empty_even_if_sibling_exists(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tp = Path(td) / "tok.json"
            dp = Path(td) / "tok.dialogue.json"
            t = PrecisionTokenizer()
            t.fit(["ab"])
            t.save_model(str(tp))
            CognitiveDialogueAgent(t, learn_tokenizer_from_user=False).save_dialogue_model(str(dp))
            self.assertEqual(
                resolve_dialogue_model_load_path(str(tp), "", dialogue_fresh=True),
                "",
            )


if __name__ == "__main__":
    unittest.main()
