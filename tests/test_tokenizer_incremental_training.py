import tempfile
import unittest
from pathlib import Path

from uva_model.tokenizer import PrecisionTokenizer


class TokenizerIncrementalTrainingTests(unittest.TestCase):
    def test_partial_fit_stream_accumulates_counts(self) -> None:
        tokenizer = PrecisionTokenizer()
        tokenizer.fit(["hello world"])
        before_h = tokenizer.unigram.get("h", 0)

        tokenizer.partial_fit_stream(iter(["hello there", "hello model"]))
        after_h = tokenizer.unigram.get("h", 0)

        self.assertGreater(after_h, before_h)
        self.assertTrue(tokenizer.fitted)

    def test_load_then_continue_training_persists(self) -> None:
        tokenizer = PrecisionTokenizer()
        tokenizer.fit(["attention model"])

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "tok.json"
            tokenizer.save_model(str(model_path))

            loaded = PrecisionTokenizer.load_model(str(model_path))
            loaded.partial_fit_stream(iter(["attention mechanism"]))
            loaded.save_model(str(model_path))

            reloaded = PrecisionTokenizer.load_model(str(model_path))
            self.assertGreater(reloaded.unigram.get("n", 0), tokenizer.unigram.get("n", 0))


if __name__ == "__main__":
    unittest.main()
