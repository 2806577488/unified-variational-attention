import tempfile
from pathlib import Path
import unittest

from uva_model.arithmetic import ArithmeticPredictiveCoder, ArithmeticProblem


class PersistenceAndCalculatorTests(unittest.TestCase):
    def test_save_and_load_preserves_memory_prediction(self) -> None:
        coder = ArithmeticPredictiveCoder(seed=21, dt=0.05)
        problem = ArithmeticProblem(op="+", a=8, b=7)
        for _ in range(12):
            coder.learn(problem)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "model.json"
            coder.save_model(str(model_path))
            loaded = ArithmeticPredictiveCoder.load_model(str(model_path), seed=21)
            result = loaded.solve(problem)
            self.assertEqual(result.answer, problem.target)

    def test_calculator_expression_interface(self) -> None:
        coder = ArithmeticPredictiveCoder(seed=3, dt=0.05)
        result = coder.solve_expression("6*7")
        self.assertEqual(result.answer, 42)


if __name__ == "__main__":
    unittest.main()
