"""Patched Gaussian DP mechanism + extended privacy accounting.

A drop-in replacement for ``presidio_fl.dp`` that builds on the baseline
accountant in two ways:

  1. It composes delta as well as epsilon. Under basic composition of n rounds
     the guarantee is (n * eps_round, n * delta_round) -- both parameters
     accumulate, so both are tracked here.

  2. It adds an RDP ("moments") accountant (Abadi et al. 2016; Mironov 2017)
     alongside basic composition, giving the tighter Gaussian composition that
     motivates DP-SGD's use of Gaussian noise.

Slide map:
  - GaussianMechanism           -> slide "Applying DP in Practice: DP-SGD"
  - BasicComposition (eps+delta)-> slide "(eps, delta) cards" / composition
  - RdpAccountant               -> slide "DP-SGD composes better (n vs sqrt n)"
"""

from __future__ import annotations

import math

import numpy as np


class PrivacyBudgetExhausted(Exception):
    """Raised when the total privacy budget has been consumed."""


# ---------------------------------------------------------------------------
# Mechanism (unchanged from the original -- it was already correct)
# ---------------------------------------------------------------------------


class GaussianMechanism:
    """Add calibrated Gaussian noise to model weights.

        sigma = sensitivity * sqrt(2 * ln(1.25 / delta)) / epsilon
    """

    def __init__(self, epsilon: float, delta: float, sensitivity: float = 1.0) -> None:
        if epsilon <= 0:
            raise ValueError("epsilon must be > 0")
        if not (0 < delta < 1):
            raise ValueError("delta must be in (0, 1)")
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity

    @property
    def sigma(self) -> float:
        return self.sensitivity * math.sqrt(2.0 * math.log(1.25 / self.delta)) / self.epsilon

    @property
    def noise_multiplier(self) -> float:
        """z = sigma / sensitivity -- the scale-free noise level used by RDP."""
        return self.sigma / self.sensitivity

    def add_noise(
        self,
        weights: dict[str, np.ndarray],
        rng: np.random.Generator | None = None,
    ) -> dict[str, np.ndarray]:
        if rng is None:
            rng = np.random.default_rng()
        return {k: v + rng.normal(0.0, self.sigma, size=v.shape) for k, v in weights.items()}


# ---------------------------------------------------------------------------
# FIX 1 -- basic composition that ACTUALLY composes delta
# ---------------------------------------------------------------------------


class BasicComposition:
    """Sequential composition of n identical (eps, delta) rounds.

    Under basic composition, n applications of an (eps_r, delta_r)-DP
    mechanism yield (n * eps_r, n * delta_r)-DP. This tracks both parameters,
    extending the baseline (which tracked epsilon) to also compose delta.
    """

    def __init__(self, epsilon_per_round: float, delta_per_round: float, max_rounds: int) -> None:
        self.epsilon_per_round = epsilon_per_round
        self.delta_per_round = delta_per_round
        self.max_rounds = max_rounds
        self.spent_rounds = 0

    def spend_round(self) -> None:
        if self.spent_rounds >= self.max_rounds:
            raise PrivacyBudgetExhausted(
                f"Budget exhausted after {self.max_rounds} rounds "
                f"(total eps={self.total_epsilon():.4f}, delta={self.total_delta():.2e})"
            )
        self.spent_rounds += 1

    def total_epsilon(self) -> float:
        return self.max_rounds * self.epsilon_per_round

    def total_delta(self) -> float:
        return self.max_rounds * self.delta_per_round  # delta composed alongside epsilon

    def spent_epsilon(self) -> float:
        return self.spent_rounds * self.epsilon_per_round

    def spent_delta(self) -> float:
        return self.spent_rounds * self.delta_per_round


# ---------------------------------------------------------------------------
# FIX 2 -- RDP (moments) accountant: tight Gaussian composition
# ---------------------------------------------------------------------------

# Standard RDP orders to search over (as in TF-Privacy / Opacus).
_DEFAULT_ORDERS: tuple[float, ...] = (
    1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0,
    10.0, 12.0, 16.0, 20.0, 32.0, 48.0, 64.0, 128.0, 256.0,
)


class RdpAccountant:
    """Renyi-DP accountant for the (unsubsampled) Gaussian mechanism.

    The Gaussian mechanism with noise multiplier ``z = sigma/sensitivity``
    satisfies (alpha, alpha / (2 z^2))-RDP for every order alpha > 1.
    RDP composes by ADDITION, so n rounds give (alpha, n * alpha / (2 z^2))-RDP.

    Conversion to (eps, delta)-DP (Mironov 2017):
        eps(delta) = min over alpha of [ rdp(alpha) + ln(1/delta) / (alpha - 1) ]

    This is the accounting that makes eps grow ~sqrt(n) instead of ~n, i.e.
    the reason DP-SGD picks Gaussian -- provided here alongside basic comp.
    """

    def __init__(
        self,
        noise_multiplier: float,
        orders: tuple[float, ...] = _DEFAULT_ORDERS,
    ) -> None:
        if noise_multiplier <= 0:
            raise ValueError("noise_multiplier must be > 0")
        self.z = noise_multiplier
        self.orders = orders
        self.steps = 0

    def step(self, n: int = 1) -> None:
        """Account for ``n`` further Gaussian applications."""
        self.steps += n

    def _rdp_at(self, alpha: float) -> float:
        # per-step RDP of the Gaussian mechanism, times number of steps
        return self.steps * alpha / (2.0 * self.z**2)

    def get_epsilon(self, delta: float) -> float:
        """Tightest (eps) for the target delta over all searched orders."""
        if self.steps == 0:
            return 0.0
        best = math.inf
        for alpha in self.orders:
            eps = self._rdp_at(alpha) + math.log(1.0 / delta) / (alpha - 1.0)
            best = min(best, eps)
        return best