import math
import unittest

from uva_model.model import UnifiedVariationalAttentionModel


class UnifiedVariationalAttentionModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = UnifiedVariationalAttentionModel(
            observation=[1.2, -0.4],
            mu=[0.3, -0.1],
            pi=[1.5, 1.2],
            pi_min=0.5,
            c_max=2.0,
            alpha=1.2,
            beta=0.7,
            gamma_phi=0.9,
            lambda_c=1.1,
            sigma0=0.3,
            tau_pi=0.8,
            dt=0.01,
            task_input=[0.2, 0.6],
            delta_s=[0.8, 0.3],
        )

    def test_free_energy_is_finite(self) -> None:
        value = self.model.free_energy()
        self.assertTrue(math.isfinite(value))

    def test_step_updates_mu_and_pi(self) -> None:
        old_mu = list(self.model.mu)
        old_pi = list(self.model.pi)
        self.model.step()
        self.assertNotEqual(old_mu, self.model.mu)
        self.assertNotEqual(old_pi, self.model.pi)
        self.assertTrue(all(p >= self.model.pi_min for p in self.model.pi))

    def test_capacity_penalty_active_when_exceeding_budget(self) -> None:
        self.model.pi = [2.2, 1.9]
        penalty = self.model.capacity_penalty()
        self.assertGreater(penalty, 0.0)

    def test_fatigue_tracks_precision_history(self) -> None:
        start_phi = list(self.model.phi)
        for _ in range(5):
            self.model.step()
        self.assertNotEqual(start_phi, self.model.phi)
        self.assertTrue(all(v >= 0 for v in self.model.phi))

    def test_behavior_interface_outputs_ddm_drift(self) -> None:
        drift = self.model.ddm_drift(kappa=0.75)
        expected = 0.75 * sum(math.sqrt(p) * ds for p, ds in zip(self.model.pi, self.model.delta_s))
        self.assertAlmostEqual(drift, expected, places=10)


if __name__ == "__main__":
    unittest.main()
