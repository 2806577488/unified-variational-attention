import unittest

from example_run import run_interactive_calculator
from uva_model.arithmetic import ArithmeticPredictiveCoder


class InteractiveCalculatorTests(unittest.TestCase):
    def test_interactive_mode_handles_expression_and_exit(self) -> None:
        coder = ArithmeticPredictiveCoder(seed=1, dt=0.05)
        inputs = iter(["6*7", "exit"])
        outputs: list[str] = []

        def fake_input(_prompt: str) -> str:
            return next(inputs)

        def fake_print(message: str) -> None:
            outputs.append(message)

        run_interactive_calculator(coder, input_fn=fake_input, output_fn=fake_print)

        combined = "\n".join(outputs)
        self.assertIn("进入交互式计算器模式", combined)
        self.assertIn("答案: 42", combined)
        self.assertIn("已退出交互式计算器模式", combined)


if __name__ == "__main__":
    unittest.main()
