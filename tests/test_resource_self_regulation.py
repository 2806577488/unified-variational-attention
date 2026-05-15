import unittest

from uva_model.tokenizer import PrecisionTokenizer


class ResourceSelfRegulationTests(unittest.TestCase):
    def test_resource_depletes_and_recovers_during_interaction(self) -> None:
        tokenizer = PrecisionTokenizer()
        tokenizer.fit(["attention model", "model learning", "attention mechanism"])
        tokenizer.idle(steps=80)

        start_r = tokenizer.resource_state()["R"]
        tokenizer.trace_tokenize("attentionmodelattentionmodelattentionmodel")
        low_r = tokenizer.resource_state()["R"]
        self.assertLess(low_r, start_r)

        tokenizer.idle(steps=80)
        recovered_r = tokenizer.resource_state()["R"]
        self.assertGreater(recovered_r, low_r)

    def test_mind_wandering_activates_under_high_load(self) -> None:
        tokenizer = PrecisionTokenizer(
            theta_F=0.6,
            R_crit=0.85,
            lambda_deplete=0.08,
            rho=0.14,
        )
        tokenizer.fit(["a b c", "b c d", "c d e"])
        tokenizer.trace_tokenize("zzzzzzzzzzzzzzzzzzzz")
        state = tokenizer.resource_state()
        self.assertGreater(state["m"], 0.05)

    def test_stream_training_consumes_resource_then_idles_back(self) -> None:
        tokenizer = PrecisionTokenizer()
        lines = ("星域能赊账吗" for _ in range(600))
        tokenizer.fit_stream(lines)
        after_train = tokenizer.resource_state()["R"]
        self.assertLess(after_train, tokenizer.R_max)

        tokenizer.idle(steps=120)
        after_idle = tokenizer.resource_state()["R"]
        self.assertGreater(after_idle, after_train)


if __name__ == "__main__":
    unittest.main()
