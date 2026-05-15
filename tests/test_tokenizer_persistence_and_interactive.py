import tempfile
import unittest
from pathlib import Path

from example_run import run_interactive_tokenizer
from uva_model.tokenizer import PrecisionTokenizer


class TokenizerPersistenceAndInteractiveTests(unittest.TestCase):
    def test_save_and_load_tokenizer_keeps_segmentation(self) -> None:
        tokenizer = PrecisionTokenizer()
        tokenizer.fit(["attention model", "model learning", "attention mechanism"])
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "tokenizer.json"
            tokenizer.save_model(str(model_path))
            loaded = PrecisionTokenizer.load_model(str(model_path))
            tokens = loaded.tokenize("attentionmodel")
            self.assertEqual(tokens, ["attention", "model"])

    def test_interactive_tokenizer_accepts_text_and_exit(self) -> None:
        tokenizer = PrecisionTokenizer()
        tokenizer.fit(["attention model", "model learning", "attention mechanism"])

        inputs = iter(["attentionmodel", "exit"])
        outputs: list[str] = []

        def fake_input(_prompt: str) -> str:
            return next(inputs)

        def fake_print(message: str) -> None:
            outputs.append(message)

        run_interactive_tokenizer(tokenizer, input_fn=fake_input, output_fn=fake_print)
        content = "\n".join(outputs)
        self.assertIn("进入交互式分词模式", content)
        self.assertIn("分词结果: attention | model", content)
        self.assertIn("已退出交互式分词模式", content)


if __name__ == "__main__":
    unittest.main()
