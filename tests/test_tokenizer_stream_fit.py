import unittest

from uva_model.tokenizer import PrecisionTokenizer


class TokenizerStreamFitTests(unittest.TestCase):
    def test_fit_stream_matches_fit_stats(self) -> None:
        corpus = ["attention model", "model learning", "attention mechanism"]
        a = PrecisionTokenizer()
        b = PrecisionTokenizer()
        a.fit(corpus)
        seen = b.fit_stream(iter(corpus))

        self.assertEqual(seen, len(corpus))
        self.assertEqual(a.unigram, b.unigram)
        self.assertEqual(a.bigram, b.bigram)
        self.assertEqual(a.follow_counts, b.follow_counts)


if __name__ == "__main__":
    unittest.main()
