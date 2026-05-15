import unittest
from unittest.mock import patch

from uva_model import tokenizer as tokenizer_module
from uva_model.tokenizer import PrecisionTokenizer, TokenizationTrace


class PrecisionTokenizerTests(unittest.TestCase):
    def test_tokenize_emergent_boundary_without_spaces(self) -> None:
        tokenizer = PrecisionTokenizer()
        tokenizer.fit(["attention model", "attention mechanism", "model learning"])
        tokens = tokenizer.tokenize("attentionmodel")
        self.assertEqual(tokens, ["attention", "model"])

    def test_boundary_has_precision_spike(self) -> None:
        tokenizer = PrecisionTokenizer()
        tokenizer.fit(["hello world", "hello there", "world model"])
        trace = tokenizer.trace_tokenize("helloworld")
        self.assertGreaterEqual(len(trace["boundary_indices"]), 1)
        b = trace["boundary_indices"][0]
        self.assertGreater(b, 0)
        self.assertGreater(trace["precision"][b], trace["precision"][b - 1])

    def test_ingest_interaction_line_updates_counts_without_fit_loop(self) -> None:
        t = PrecisionTokenizer()
        t.fit(["ab"])
        n0 = t.total_chars
        t.ingest_interaction_line("xy")
        self.assertGreater(t.total_chars, n0)
        self.assertGreaterEqual(t.unigram.get("x", 0), 1)
        self.assertGreaterEqual(t.bigram.get("^", {}).get("x", 0), 1)

    def test_ingest_repeated_surface_lowers_next_surprise(self) -> None:
        def mean_surprise(tr: dict) -> float:
            sur = tr.get("surprise", [])
            if not sur:
                return 0.0
            return sum(float(x) for x in sur) / len(sur)

        t = PrecisionTokenizer()
        t.fit(["你好世界"])
        s0 = mean_surprise(t.trace_tokenize("罕见短语"))
        for _ in range(12):
            t.ingest_interaction_line("罕见短语")
        s1 = mean_surprise(t.trace_tokenize("罕见短语"))
        self.assertLess(s1, s0)

    def test_mean_surprise_matches_trace_average(self) -> None:
        t = PrecisionTokenizer()
        t.fit(["hello world", "hello there", "world model"])
        trace = t.trace_tokenize("helloworld")
        sur = trace.get("surprise", [])
        expected = sum(float(x) for x in sur) / len(sur)
        got = t.mean_surprise("helloworld")
        self.assertAlmostEqual(got, expected, places=6)

    def test_mean_surprise_advances_resource_state_like_trace(self) -> None:
        t0 = PrecisionTokenizer()
        t0.fit(["hello world", "hello there", "world model"])
        t1 = PrecisionTokenizer.from_dict(t0.to_dict())
        t2 = PrecisionTokenizer.from_dict(t0.to_dict())

        trace = t1.trace_tokenize("helloworld")
        got = t2.mean_surprise("helloworld")

        sur = trace.get("surprise", [])
        expected = sum(float(x) for x in sur) / len(sur)
        self.assertAlmostEqual(got, expected, places=6)
        s1 = t1.resource_state()
        s2 = t2.resource_state()
        self.assertAlmostEqual(s1["R"], s2["R"], places=6)
        self.assertAlmostEqual(s1["m"], s2["m"], places=6)
        self.assertAlmostEqual(s1["R_max"], s2["R_max"], places=6)
        self.assertAlmostEqual(s1["F_ema"], s2["F_ema"], places=6)

    def test_mean_surprise_batch_matches_sequential_calls_and_resource_state(self) -> None:
        texts = ["helloworld", "worldhello", "hellohello"]
        t0 = PrecisionTokenizer()
        t0.fit(["hello world", "hello there", "world model"])
        t1 = PrecisionTokenizer.from_dict(t0.to_dict())
        t2 = PrecisionTokenizer.from_dict(t0.to_dict())

        expected = [t1.mean_surprise(text) for text in texts]
        got = t2.mean_surprise_batch(texts)

        self.assertEqual(len(got), len(expected))
        for lhs, rhs in zip(got, expected, strict=True):
            self.assertAlmostEqual(lhs, rhs, places=6)

        s1 = t1.resource_state()
        s2 = t2.resource_state()
        self.assertAlmostEqual(s1["R"], s2["R"], places=6)
        self.assertAlmostEqual(s1["m"], s2["m"], places=6)
        self.assertAlmostEqual(s1["R_max"], s2["R_max"], places=6)
        self.assertAlmostEqual(s1["F_ema"], s2["F_ema"], places=6)

    def test_trace_can_delegate_to_optional_accelerated_impl(self) -> None:
        t = PrecisionTokenizer()
        t.fit(["hello world", "hello there", "world model"])

        def fake_trace_impl(tokenizer: PrecisionTokenizer, text: str, **_kwargs: object) -> TokenizationTrace:
            tokenizer.R = 0.42
            tokenizer.m = 0.24
            tokenizer.F_ema = 1.23
            return TokenizationTrace(
                tokens=[f"accel:{text}"],
                precision=[],
                surprise=[],
                boundary_indices=[],
                resource=[],
                mind_wander=[],
                auto_rest_count=0,
                mean_surprise=0.75,
            )

        with patch.object(tokenizer_module, "_TRACE_ACCEL_IMPL", fake_trace_impl):
            trace = t._trace("helloworld")  # noqa: SLF001

        self.assertEqual(trace.tokens, ["accel:helloworld"])
        self.assertAlmostEqual(trace.mean_surprise, 0.75, places=6)
        state = t.resource_state()
        self.assertAlmostEqual(state["R"], 0.42, places=6)
        self.assertAlmostEqual(state["m"], 0.24, places=6)
        self.assertAlmostEqual(state["F_ema"], 1.23, places=6)

    def test_mean_surprise_batch_can_delegate_to_optional_accelerated_impl(self) -> None:
        t = PrecisionTokenizer()
        t.fit(["hello world", "hello there", "world model"])

        def fake_batch_impl(tokenizer: PrecisionTokenizer, texts: list[str]) -> list[float]:
            tokenizer.R = 0.33
            return [float(len(text)) for text in texts]

        with patch.object(tokenizer_module, "_TRACE_ACCEL_BATCH_IMPL", fake_batch_impl):
            got = t.mean_surprise_batch(["ab", "abcd"])

        self.assertEqual(got, [2.0, 4.0])
        self.assertAlmostEqual(t.resource_state()["R"], 0.33, places=6)

    def test_compiled_accelerator_matches_python_trace_when_available(self) -> None:
        try:
            from uva_model import _tokenizer_accel
        except ImportError:
            self.skipTest("compiled tokenizer accelerator is not available")

        t0 = PrecisionTokenizer()
        t0.fit(["hello world", "hello there", "world model"])
        t1 = PrecisionTokenizer.from_dict(t0.to_dict())
        t2 = PrecisionTokenizer.from_dict(t0.to_dict())

        expected = t1._trace_python("helloworld")  # noqa: SLF001
        got = _tokenizer_accel.trace_tokenize(t2, "helloworld")

        self.assertEqual(got.tokens, expected.tokens)
        self.assertEqual(got.boundary_indices, expected.boundary_indices)
        self.assertAlmostEqual(got.mean_surprise, expected.mean_surprise, places=6)
        for lhs, rhs in zip(got.surprise, expected.surprise, strict=True):
            self.assertAlmostEqual(lhs, rhs, places=6)
        self.assertAlmostEqual(t2.R, t1.R, places=6)
        self.assertAlmostEqual(t2.m, t1.m, places=6)
        self.assertAlmostEqual(t2.R_max, t1.R_max, places=6)
        self.assertAlmostEqual(t2.F_ema, t1.F_ema, places=6)


if __name__ == "__main__":
    unittest.main()
