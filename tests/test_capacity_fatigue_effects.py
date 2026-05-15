import unittest

from uva_model.arithmetic import ArithmeticPredictiveCoder, ArithmeticProblem
from uva_model.curriculum import CurriculumA, Evaluator


class CapacityFatigueEffectsTests(unittest.TestCase):
    def test_low_capacity_increases_estimation_bias(self) -> None:
        dataset = CurriculumA().sample(count=40, ops=["+", "*"])

        high_capacity = ArithmeticPredictiveCoder(seed=3, c_max=3.0)
        low_capacity = ArithmeticPredictiveCoder(seed=3, c_max=0.5)

        high_metrics = Evaluator(high_capacity).evaluate(dataset)
        low_metrics = Evaluator(low_capacity).evaluate(dataset)
        self.assertGreater(low_metrics["mean_abs_error"], high_metrics["mean_abs_error"])

    def test_fatigue_increases_carry_related_errors(self) -> None:
        coder = ArithmeticPredictiveCoder(seed=13)
        hard_set = [ArithmeticProblem(op="+", a=9, b=8), ArithmeticProblem(op="+", a=7, b=8)] * 30
        low_fatigue_errors = coder.carry_error_rate(hard_set[:10])

        for problem in hard_set:
            coder.learn(problem)

        high_fatigue_errors = coder.carry_error_rate(hard_set[:10], induce_fatigue=True)
        self.assertGreater(high_fatigue_errors, low_fatigue_errors)


if __name__ == "__main__":
    unittest.main()
