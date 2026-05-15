from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable, List, Optional


def _default_g(mu_i: float) -> float:
    return mu_i


@dataclass
class UnifiedVariationalAttentionModel:
    observation: List[float]
    mu: List[float]
    pi: List[float]
    pi_min: float
    c_max: float
    alpha: float
    beta: float
    gamma_phi: float
    lambda_c: float
    sigma0: float
    tau_pi: float
    dt: float
    task_input: List[float]
    delta_s: List[float]
    pi_s: List[float] | None = None
    prior_mean: List[float] | None = None
    tau_f: float = 2.0
    g_fn: Callable[[float], float] = _default_g
    g_prime: Callable[[float], float] = lambda _x: 1.0
    phi: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        dim = len(self.mu)
        if not (len(self.observation) == len(self.pi) == len(self.task_input) == len(self.delta_s) == dim):
            raise ValueError("All input vectors must share the same length.")
        if self.pi_s is None:
            self.pi_s = [1.0] * dim
        if self.prior_mean is None:
            self.prior_mean = [0.0] * dim
        if self.phi and len(self.phi) != dim:
            raise ValueError("phi length must match state dimension.")
        if not self.phi:
            self.phi = [0.0] * dim
        self.pi = [max(p, self.pi_min) for p in self.pi]

    def sensory_error(self) -> List[float]:
        return [o - self.g_fn(m) for o, m in zip(self.observation, self.mu)]

    def prior_error(self) -> List[float]:
        assert self.prior_mean is not None
        return [m - p for m, p in zip(self.mu, self.prior_mean)]

    def novelty_signal(self, eps: float) -> float:
        return (eps * eps) / (self.sigma0 * self.sigma0 + eps * eps)

    def effective_drive(self, eps_o: Optional[List[float]] = None) -> List[float]:
        if eps_o is None:
            eps_o = self.sensory_error()
        return [
            self.alpha * self.novelty_signal(e) + self.beta * u
            for e, u in zip(eps_o, self.task_input)
        ]

    def capacity_excess(self) -> float:
        return max(0.0, sum(p - self.pi_min for p in self.pi) - self.c_max)

    def capacity_penalty(self) -> float:
        excess = self.capacity_excess()
        return 0.5 * self.lambda_c * excess * excess

    def free_energy(self) -> float:
        eps_o = self.sensory_error()
        eps_s = self.prior_error()
        assert self.pi_s is not None

        sensory_term = 0.5 * sum(p * (e * e) - math.log(p) for p, e in zip(self.pi, eps_o))
        prior_term = 0.5 * sum(ps * (e * e) for ps, e in zip(self.pi_s, eps_s))

        drive = self.effective_drive(eps_o)
        drive_term = -sum(g * (p - self.pi_min) for g, p in zip(drive, self.pi))
        fatigue_term = 0.5 * self.gamma_phi * sum(ph * (p - self.pi_min) ** 2 for ph, p in zip(self.phi, self.pi))

        return sensory_term + prior_term + self.capacity_penalty() + drive_term + fatigue_term

    def gradients(self) -> tuple[List[float], List[float]]:
        eps_o = self.sensory_error()
        eps_s = self.prior_error()
        assert self.pi_s is not None

        grad_mu = []
        for e_o, e_s, p, ps, m in zip(eps_o, eps_s, self.pi, self.pi_s, self.mu):
            grad = -(p * e_o * self.g_prime(m)) + ps * e_s
            grad_mu.append(grad)

        drive = self.effective_drive(eps_o)
        excess = self.capacity_excess()
        grad_pi = []
        for e_o, p, g, ph in zip(eps_o, self.pi, drive, self.phi):
            grad = 0.5 * (e_o * e_o - 1.0 / p)
            grad += self.lambda_c * excess
            grad -= g
            grad += self.gamma_phi * ph * (p - self.pi_min)
            grad_pi.append(grad)
        return grad_mu, grad_pi

    def _update_fatigue(self) -> None:
        alpha_f = math.exp(-self.dt / self.tau_f)
        self.phi = [alpha_f * ph + (1.0 - alpha_f) * (p - self.pi_min) for ph, p in zip(self.phi, self.pi)]

    def step(self) -> None:
        grad_mu, grad_pi = self.gradients()

        self.mu = [m - self.dt * g for m, g in zip(self.mu, grad_mu)]
        self.pi = [max(self.pi_min, p - (self.dt / self.tau_pi) * g) for p, g in zip(self.pi, grad_pi)]

        self._update_fatigue()

    def gamma_power_proxy(self) -> float:
        eps_o = self.sensory_error()
        return sum(p * (e * e) for p, e in zip(self.pi, eps_o))

    def alpha_power_proxy(self, index: int) -> float:
        if index < 0 or index >= len(self.pi):
            raise IndexError("index out of range")
        return self.pi_min / self.pi[index]

    def ddm_drift(self, kappa: float) -> float:
        return kappa * sum(math.sqrt(p) * ds for p, ds in zip(self.pi, self.delta_s))
