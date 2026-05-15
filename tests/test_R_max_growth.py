import unittest

from uva_model.tokenizer import PrecisionTokenizer


class RMaxGrowthTests(unittest.TestCase):
    def test_R_max_grows_when_long_term_uncertainty_falls(self) -> None:
        tok = PrecisionTokenizer(
            tau_grow=80.0,
            eta_learn=0.06,
            lambda_grow=0.001,
            F_ema_beta=0.08,
            R_max_cap=2.2,
            R_base=0.45,
            R_max=1.0,
        )
        tok.fit(["相同句子重复", "相同句子重复", "相同句子重复"])
        r0 = tok.R_max
        for _ in range(400):
            tok.partial_fit(["相同句子重复"])
        self.assertGreater(tok.R_max, r0)
        self.assertLessEqual(tok.R_max, tok.R_max_cap)
        self.assertGreaterEqual(tok.R_max, tok.R_base)

    def test_inference_also_updates_R_max_trace(self) -> None:
        tok = PrecisionTokenizer(tau_grow=50.0, eta_learn=0.05, F_ema_beta=0.1)
        tok.fit(["ab cd", "ab ef"])
        r0 = tok.R_max
        tok.trace_tokenize("abcd")
        self.assertIsNotNone(tok.resource_state()["F_ema"])
        self.assertGreaterEqual(tok.R_max, r0)


if __name__ == "__main__":
    unittest.main()
