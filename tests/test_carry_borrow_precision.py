import unittest

from uva_model.arithmetic import ArithmeticPredictiveCoder, ArithmeticProblem


class CarryBorrowPrecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.coder = ArithmeticPredictiveCoder(seed=9)

    def test_carry_event_boosts_carry_path_precision(self) -> None:
        problem = ArithmeticProblem(op="+", a=8, b=7)
        trace = self.coder.trace(problem)
        self.assertGreater(trace["carry_precision_peak"], trace["base_precision_peak"])

    def test_borrow_event_boosts_borrow_path_precision(self) -> None:
        problem = ArithmeticProblem(op="-", a=11, b=7)
        trace = self.coder.trace(problem)
        self.assertGreater(trace["borrow_precision_peak"], trace["base_precision_peak"])


if __name__ == "__main__":
    unittest.main()
