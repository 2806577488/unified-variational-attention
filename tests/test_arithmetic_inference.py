import unittest

from uva_model.arithmetic import ArithmeticPredictiveCoder, ArithmeticProblem


class ArithmeticInferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.coder = ArithmeticPredictiveCoder(seed=7)

    def test_addition_problem_converges_to_correct_answer(self) -> None:
        problem = ArithmeticProblem(op="+", a=3, b=4)
        result = self.coder.solve(problem)
        self.assertEqual(result.answer, 7)
        self.assertLess(result.steps, self.coder.max_steps + 1)

    def test_subtraction_problem_converges_to_correct_answer(self) -> None:
        problem = ArithmeticProblem(op="-", a=9, b=2)
        result = self.coder.solve(problem)
        self.assertEqual(result.answer, 7)

    def test_multiplication_problem_converges_to_correct_answer(self) -> None:
        problem = ArithmeticProblem(op="*", a=6, b=7)
        result = self.coder.solve(problem)
        self.assertEqual(result.answer, 42)

    def test_integer_division_problem_converges_to_correct_answer(self) -> None:
        problem = ArithmeticProblem(op="/", a=8, b=2)
        result = self.coder.solve(problem)
        self.assertEqual(result.answer, 4)


if __name__ == "__main__":
    unittest.main()
