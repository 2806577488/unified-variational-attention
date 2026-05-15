import unittest

from uva_model.arithmetic import ArithmeticPredictiveCoder, ArithmeticProblem


class HebbianMemoryTests(unittest.TestCase):
    def test_repeated_exposure_reduces_average_steps(self) -> None:
        coder = ArithmeticPredictiveCoder(seed=11)
        problems = [ArithmeticProblem(op="+", a=4, b=5), ArithmeticProblem(op="+", a=6, b=3)]

        before = [coder.solve(problem).steps for problem in problems]

        for _ in range(25):
            for problem in problems:
                coder.learn(problem)

        after = [coder.solve(problem).steps for problem in problems]
        self.assertLess(sum(after) / len(after), sum(before) / len(before))


if __name__ == "__main__":
    unittest.main()
