import unittest

from uva_model.tokenizer import PrecisionTokenizer


class AutoRestTokenizationTests(unittest.TestCase):
    def test_low_resource_auto_rest_then_continue(self) -> None:
        tokenizer = PrecisionTokenizer(auto_rest_threshold=0.2, auto_resume_threshold=0.55)
        tokenizer.fit(["你好今天心情如何", "今天心情不错", "你好你好"])
        tokenizer.R = 0.1

        trace = tokenizer.trace_tokenize("你好今天心情如何")
        self.assertGreaterEqual(trace["auto_rest_count"], 1)
        self.assertGreater(trace["resource_state"]["R"], 0.0)
        self.assertGreaterEqual(len(trace["tokens"]), 1)


if __name__ == "__main__":
    unittest.main()
