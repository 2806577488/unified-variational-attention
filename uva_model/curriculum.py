from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, Iterable, List

from .arithmetic import ArithmeticPredictiveCoder, ArithmeticProblem


@dataclass
class CurriculumA:
    seed: int = 0

    def sample(self, count: int, ops: List[str] | None = None) -> List[ArithmeticProblem]:
        rng = random.Random(self.seed)
        chosen_ops = ops if ops is not None else ["+", "-", "*", "/"]
        dataset: List[ArithmeticProblem] = []

        while len(dataset) < count:
            op = chosen_ops[len(dataset) % len(chosen_ops)]
            a = rng.randint(0, 9)
            b = rng.randint(0, 9)

            if op == "/":
                b = rng.randint(1, 9)
                q = rng.randint(0, 9)
                a = b * q
            elif op == "-" and a < b:
                a, b = b, a

            dataset.append(ArithmeticProblem(op=op, a=a, b=b))
        return dataset


class Evaluator:
    def __init__(self, coder: ArithmeticPredictiveCoder) -> None:
        self.coder = coder

    def evaluate(self, dataset: Iterable[ArithmeticProblem]) -> Dict[str, float]:
        total = 0
        correct = 0
        sum_steps = 0.0
        sum_abs_error = 0.0
        sum_free_energy_drop = 0.0
        carry_total = 0
        carry_wrong = 0

        for problem in dataset:
            total += 1
            result = self.coder.solve(problem)
            target = problem.target

            if result.answer == target:
                correct += 1
            sum_steps += result.steps
            sum_abs_error += abs(result.answer - target)

            if result.free_energy_trace:
                sum_free_energy_drop += result.free_energy_trace[0] - result.free_energy_trace[-1]

            if problem.op == "+" and (problem.a % 10 + problem.b % 10) >= 10:
                carry_total += 1
                if result.answer != target:
                    carry_wrong += 1

        if total == 0:
            return {
                "accuracy": 0.0,
                "mean_steps": 0.0,
                "mean_abs_error": 0.0,
                "mean_free_energy_drop": 0.0,
                "carry_error_rate": 0.0,
            }

        return {
            "accuracy": correct / total,
            "mean_steps": sum_steps / total,
            "mean_abs_error": sum_abs_error / total,
            "mean_free_energy_drop": sum_free_energy_drop / total,
            "carry_error_rate": carry_wrong / max(1, carry_total),
        }
