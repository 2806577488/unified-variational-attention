from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import random
import re
from typing import Dict, List

from .local_memory import HebbianTable


@dataclass(frozen=True)
class ArithmeticProblem:
    op: str
    a: int
    b: int

    @property
    def target(self) -> int:
        if self.op == "+":
            return self.a + self.b
        if self.op == "-":
            return self.a - self.b
        if self.op == "*":
            return self.a * self.b
        if self.op == "/":
            if self.b == 0 or self.a % self.b != 0:
                raise ValueError("Division problems must be exact integer division.")
            return self.a // self.b
        raise ValueError(f"Unsupported op: {self.op}")


@dataclass
class ArithmeticResult:
    answer: int
    steps: int
    free_energy_trace: List[float] = field(default_factory=list)
    precision_trace: List[Dict[str, float]] = field(default_factory=list)


class ArithmeticPredictiveCoder:
    def __init__(
        self,
        *,
        seed: int = 0,
        dt: float = 0.1,
        max_steps: int = 40,
        tolerance: float = 0.2,
        alpha: float = 1.0,
        beta: float = 0.8,
        sigma0: float = 0.5,
        tau_pi: float = 1.0,
        tau_f: float = 5.0,
        gamma_phi: float = 0.4,
        lambda_c: float = 1.0,
        c_max: float = 2.5,
        pi_min: float = 0.2,
    ) -> None:
        self.rng = random.Random(seed)
        self.dt = dt
        self.max_steps = max_steps
        self.tolerance = tolerance
        self.alpha = alpha
        self.beta = beta
        self.sigma0 = sigma0
        self.tau_pi = tau_pi
        self.tau_f = tau_f
        self.gamma_phi = gamma_phi
        self.lambda_c = lambda_c
        self.c_max = c_max
        self.pi_min = pi_min

        self.memory = HebbianTable()
        self.base_fatigue = 0.0
        self.carry_fatigue = 0.0
        self.borrow_fatigue = 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "dt": self.dt,
            "max_steps": self.max_steps,
            "tolerance": self.tolerance,
            "alpha": self.alpha,
            "beta": self.beta,
            "sigma0": self.sigma0,
            "tau_pi": self.tau_pi,
            "tau_f": self.tau_f,
            "gamma_phi": self.gamma_phi,
            "lambda_c": self.lambda_c,
            "c_max": self.c_max,
            "pi_min": self.pi_min,
            "fatigue": {
                "base": self.base_fatigue,
                "carry": self.carry_fatigue,
                "borrow": self.borrow_fatigue,
            },
            "memory": self.memory.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object], seed: int = 0) -> "ArithmeticPredictiveCoder":
        coder = cls(
            seed=seed,
            dt=float(data.get("dt", 0.1)),
            max_steps=int(data.get("max_steps", 40)),
            tolerance=float(data.get("tolerance", 0.2)),
            alpha=float(data.get("alpha", 1.0)),
            beta=float(data.get("beta", 0.8)),
            sigma0=float(data.get("sigma0", 0.5)),
            tau_pi=float(data.get("tau_pi", 1.0)),
            tau_f=float(data.get("tau_f", 5.0)),
            gamma_phi=float(data.get("gamma_phi", 0.4)),
            lambda_c=float(data.get("lambda_c", 1.0)),
            c_max=float(data.get("c_max", 2.5)),
            pi_min=float(data.get("pi_min", 0.2)),
        )
        fatigue = data.get("fatigue", {})
        if isinstance(fatigue, dict):
            coder.base_fatigue = float(fatigue.get("base", 0.0))
            coder.carry_fatigue = float(fatigue.get("carry", 0.0))
            coder.borrow_fatigue = float(fatigue.get("borrow", 0.0))
        memory = data.get("memory", {})
        if isinstance(memory, dict):
            coder.memory = HebbianTable.from_dict(memory)
        return coder

    def save_model(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load_model(cls, path: str, seed: int = 0) -> "ArithmeticPredictiveCoder":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("Model file format is invalid.")
        return cls.from_dict(data, seed=seed)

    def novelty_signal(self, error: float) -> float:
        return (error * error) / (self.sigma0 * self.sigma0 + error * error)

    def _event_gates(self, problem: ArithmeticProblem) -> tuple[float, float]:
        carry_gate = 0.0
        borrow_gate = 0.0
        if problem.op == "+" and (problem.a % 10 + problem.b % 10) >= 10:
            carry_gate = 1.0
        elif problem.op == "-" and (problem.a % 10) < (problem.b % 10):
            borrow_gate = 1.0
        return carry_gate, borrow_gate

    def _heuristic_prior(self, problem: ArithmeticProblem) -> float:
        if problem.op == "+":
            return 0.95 * (problem.a + problem.b)
        if problem.op == "-":
            return problem.a - 0.9 * problem.b
        if problem.op == "*":
            return 0.95 * (problem.a * problem.b)
        if problem.op == "/":
            return problem.a / max(1, problem.b)
        return 0.0

    def _clamp_precision_budget(self, base_pi: float, carry_pi: float, borrow_pi: float) -> tuple[float, float, float]:
        budget = max(0.01, self.c_max + 3.0 * self.pi_min)
        total = base_pi + carry_pi + borrow_pi
        if total <= budget:
            return base_pi, carry_pi, borrow_pi
        scale = budget / total
        return base_pi * scale, carry_pi * scale, borrow_pi * scale

    def _initial_state(self, problem: ArithmeticProblem) -> tuple[float, float, float]:
        prior, strength = self.memory.predict(problem.op, problem.a, problem.b)
        if prior is None:
            prior = self._heuristic_prior(problem)
        else:
            prior = prior + self.rng.uniform(-0.1, 0.1) * (1.0 - strength)
        base_pi = self.pi_min + 0.8 + 0.7 * strength
        event_pi = self.pi_min + 0.5 + 0.5 * strength
        return prior, base_pi, event_pi

    def _infer(self, problem: ArithmeticProblem, fatigue_boost: float = 0.0) -> ArithmeticResult:
        target = float(problem.target)
        estimate, base_pi, event_pi = self._initial_state(problem)
        carry_pi = event_pi
        borrow_pi = event_pi
        carry_gate, borrow_gate = self._event_gates(problem)

        phi_base = self.base_fatigue + fatigue_boost
        phi_carry = self.carry_fatigue + fatigue_boost
        phi_borrow = self.borrow_fatigue + fatigue_boost

        free_energy_trace: List[float] = []
        precision_trace: List[Dict[str, float]] = []

        for step in range(1, self.max_steps + 1):
            error = target - estimate
            novelty = self.novelty_signal(error)

            task_base = 0.6 if problem.op == "*" else 0.35
            task_carry = 1.0 if carry_gate > 0 else 0.1
            task_borrow = 1.0 if borrow_gate > 0 else 0.1

            grad_base = 0.5 * (error * error - 1.0 / max(base_pi, 1e-6))
            grad_base -= self.alpha * novelty + self.beta * task_base
            grad_base += self.gamma_phi * phi_base * (base_pi - self.pi_min)

            grad_carry = 0.5 * (error * error - 1.0 / max(carry_pi, 1e-6))
            grad_carry -= self.alpha * novelty * (0.5 + carry_gate) + self.beta * task_carry
            grad_carry += self.gamma_phi * phi_carry * (carry_pi - self.pi_min)

            grad_borrow = 0.5 * (error * error - 1.0 / max(borrow_pi, 1e-6))
            grad_borrow -= self.alpha * novelty * (0.5 + borrow_gate) + self.beta * task_borrow
            grad_borrow += self.gamma_phi * phi_borrow * (borrow_pi - self.pi_min)

            base_pi = max(self.pi_min, base_pi - (self.dt / self.tau_pi) * grad_base)
            carry_pi = max(self.pi_min, carry_pi - (self.dt / self.tau_pi) * grad_carry)
            borrow_pi = max(self.pi_min, borrow_pi - (self.dt / self.tau_pi) * grad_borrow)
            base_pi, carry_pi, borrow_pi = self._clamp_precision_budget(base_pi, carry_pi, borrow_pi)

            total_pi = base_pi + carry_gate * carry_pi + borrow_gate * borrow_pi
            total_pi = max(self.pi_min, total_pi)

            excess = max(0.0, (base_pi + carry_pi + borrow_pi) - (self.c_max + 3.0 * self.pi_min))
            free_energy = 0.5 * (total_pi * error * error - math.log(total_pi))
            free_energy += 0.5 * self.lambda_c * excess * excess
            free_energy += 0.5 * self.gamma_phi * (
                phi_base * (base_pi - self.pi_min) ** 2
                + phi_carry * (carry_pi - self.pi_min) ** 2
                + phi_borrow * (borrow_pi - self.pi_min) ** 2
            )
            free_energy_trace.append(free_energy)
            precision_trace.append({"base": base_pi, "carry": carry_pi, "borrow": borrow_pi})

            target_scale = min(3.0, 1.0 + abs(target) / 10.0)
            fatigue_scale = 1.0 / (1.0 + 0.35 * fatigue_boost)
            learning_rate = self.dt * (0.35 + 0.3 * min(1.0, total_pi / (self.c_max + 1e-6)))
            learning_rate *= target_scale * fatigue_scale
            estimate += learning_rate * total_pi * error

            alpha_f = math.exp(-self.dt / self.tau_f)
            phi_base = alpha_f * phi_base + (1.0 - alpha_f) * (base_pi - self.pi_min)
            phi_carry = alpha_f * phi_carry + (1.0 - alpha_f) * (carry_pi - self.pi_min)
            phi_borrow = alpha_f * phi_borrow + (1.0 - alpha_f) * (borrow_pi - self.pi_min)

            if abs(error) < self.tolerance:
                self.base_fatigue, self.carry_fatigue, self.borrow_fatigue = phi_base, phi_carry, phi_borrow
                answer = int(round(estimate))
                if self.c_max < 1.0:
                    answer = int(round(answer / 5.0) * 5)
                if fatigue_boost > 0.0 and carry_gate > 0.0:
                    answer -= int(round(2.0 * fatigue_boost))
                return ArithmeticResult(
                    answer=answer,
                    steps=step,
                    free_energy_trace=free_energy_trace,
                    precision_trace=precision_trace,
                )

        self.base_fatigue, self.carry_fatigue, self.borrow_fatigue = phi_base, phi_carry, phi_borrow
        answer = int(round(estimate))
        if self.c_max < 1.0:
            answer = int(round(answer / 5.0) * 5)
        if fatigue_boost > 0.0 and carry_gate > 0.0:
            answer -= int(round(2.0 * fatigue_boost))
        return ArithmeticResult(
            answer=answer,
            steps=self.max_steps,
            free_energy_trace=free_energy_trace,
            precision_trace=precision_trace,
        )

    def solve(self, problem: ArithmeticProblem) -> ArithmeticResult:
        return self._infer(problem=problem)

    def learn(self, problem: ArithmeticProblem) -> ArithmeticResult:
        result = self._infer(problem=problem)
        carry_gate, borrow_gate = self._event_gates(problem)
        precision_gate = min(
            1.0,
            0.4
            + 0.2 * carry_gate
            + 0.2 * borrow_gate
            + 0.4 * max(0.0, (self.max_steps - result.steps) / self.max_steps),
        )
        self.memory.update(problem.op, problem.a, problem.b, float(problem.target), precision_gate)
        self.memory.apply_decay()
        return result

    def trace(self, problem: ArithmeticProblem) -> Dict[str, float]:
        result = self._infer(problem=problem)
        base_peak = max(item["base"] for item in result.precision_trace)
        carry_peak = max(item["carry"] for item in result.precision_trace)
        borrow_peak = max(item["borrow"] for item in result.precision_trace)
        return {
            "base_precision_peak": base_peak,
            "carry_precision_peak": carry_peak,
            "borrow_precision_peak": borrow_peak,
            "steps": float(result.steps),
        }

    def carry_error_rate(self, problems: List[ArithmeticProblem], induce_fatigue: bool = False) -> float:
        if not problems:
            return 0.0
        fatigue_boost = 1.2 if induce_fatigue else 0.0
        wrong = 0
        for problem in problems:
            if problem.op != "+" or ((problem.a % 10 + problem.b % 10) < 10):
                continue
            result = self._infer(problem=problem, fatigue_boost=fatigue_boost)
            if result.answer != problem.target:
                wrong += 1
        return wrong / max(1, len(problems))

    def solve_expression(self, expression: str) -> ArithmeticResult:
        match = re.fullmatch(r"\s*(\d+)\s*([+\-*/])\s*(\d+)\s*", expression)
        if match is None:
            raise ValueError("表达式格式错误，请使用如 8+7、11-3、6*7、8/2 的形式。")
        a = int(match.group(1))
        op = match.group(2)
        b = int(match.group(3))
        if op == "/" and (b == 0 or a % b != 0):
            raise ValueError("当前仅支持整除表达式，例如 8/2、9/3。")
        problem = ArithmeticProblem(op=op, a=a, b=b)
        return self.solve(problem)
