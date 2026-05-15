import unittest

from uva_model.tokenizer import PrecisionTokenizer


class TokenizerChineseNoSpaceTests(unittest.TestCase):
    def test_no_explicit_punctuation_boundary_rule(self) -> None:
        tokenizer = PrecisionTokenizer()
        tokenizer.fit(["你好今天天气如何", "今天心情不错", "你好吗"])
        trace = tokenizer.trace_tokenize("你好，今天天气如何？")
        punct_indices = [i for i, ch in enumerate("你好，今天天气如何？") if ch in "，？"]
        # 边界来自动力学事件，不应等同于“标点即边界”的硬规则。
        self.assertNotEqual(trace["boundary_indices"], punct_indices)

    def test_no_space_corpus_can_still_split_by_surprise_jump(self) -> None:
        tokenizer = PrecisionTokenizer()
        tokenizer.fit(
            [
                "你好今天天气如何",
                "你好今天心情如何",
                "晚安今天早点休息",
                "晚安明天继续努力",
            ]
        )
        tokens = tokenizer.tokenize("你好今天心情如何")
        self.assertGreaterEqual(len(tokens), 2)


if __name__ == "__main__":
    unittest.main()
