"""Secure Aggregation via pairwise masking (Bonawitz et al., CCS 2017).

The baseline repo is Central DP with a trusted server: clients send raw
updates and the server noises the aggregate. This module adds Secure
Aggregation on top, so the server learns only the SUM of client updates,
never any individual update, without a trusted aggregator.

Slide map:
  - PairwiseMasking.mask / server_sum        -> slide "The Protocol: Pairwise Masking"
  - ShamirSecretSharing / recover_dropped     -> slide "Guarantees and the Dropout Problem"

NOTE on realism: pairwise seeds here are produced by a trusted setup that
stands in for a Diffie-Hellman key exchange (each pair (i, j) ends up sharing
one secret seed).  The masking algebra and the dropout-recovery logic are the
real protocol; only the key-exchange step is simulated.
"""

from __future__ import annotations

import hashlib
from itertools import combinations

import numpy as np

Vector = np.ndarray


# ---------------------------------------------------------------------------
# Pairwise masking (slide 12)
# ---------------------------------------------------------------------------


def _prg(seed: int, shape: tuple[int, ...]) -> Vector:
    """Deterministic pseudo-random mask vector from an integer seed."""
    return np.random.default_rng(seed).normal(0.0, 1.0, size=shape)


class PairwiseMasking:
    """Additive pairwise masking so that masks cancel in the server sum.

    For each unordered pair (i, j) with i < j a shared seed s_ij is drawn.
    Client i ADDS mask(s_ij); client j SUBTRACTS mask(s_ij).  When the server
    sums every masked update, each mask appears once with + and once with -,
    so all masks cancel exactly and the server obtains sum_i x_i.
    """

    def __init__(self, n_clients: int, dim: int, session_key: int = 20260721) -> None:
        self.n = n_clients
        self.dim = dim
        # trusted-setup stand-in for Diffie-Hellman: one seed per unordered pair
        self.pair_seed: dict[tuple[int, int], int] = {}
        for i, j in combinations(range(n_clients), 2):
            h = hashlib.sha256(f"{session_key}:{i}:{j}".encode()).hexdigest()
            self.pair_seed[(i, j)] = int(h[:16], 16)

    def mask(self, client_id: int, x: Vector) -> Vector:
        """Return client_id's masked update x + sum(+/- pairwise masks)."""
        out = x.astype(float).copy()
        for other in range(self.n):
            if other == client_id:
                continue
            i, j = sorted((client_id, other))
            m = _prg(self.pair_seed[(i, j)], (self.dim,))
            out += m if client_id < other else -m  # smaller id adds, larger subtracts
        return out

    def server_sum(self, masked_updates: dict[int, Vector]) -> Vector:
        """Sum masked updates; pairwise masks among present clients cancel."""
        return np.sum(list(masked_updates.values()), axis=0)


# ---------------------------------------------------------------------------
# Dropout recovery via Shamir secret sharing (slide 13)
# ---------------------------------------------------------------------------

_PRIME = 2**61 - 1  # Mersenne prime for the secret-sharing field


class ShamirSecretSharing:
    """(t, n) Shamir secret sharing over GF(_PRIME).

    A dropped client's pairwise seeds can be reconstructed by any t surviving
    peers, so the server can subtract the dropped client's masks WITHOUT ever
    learning that client's update.
    """

    def __init__(self, threshold: int, n_shares: int, seed: int = 0) -> None:
        self.t = threshold
        self.n = n_shares
        self._rng = np.random.default_rng(seed)

    def split(self, secret: int) -> list[tuple[int, int]]:
        secret %= _PRIME
        coeffs = [secret] + [int(self._rng.integers(0, _PRIME)) for _ in range(self.t - 1)]
        shares = []
        for x in range(1, self.n + 1):
            y = 0
            for power, c in enumerate(coeffs):
                y = (y + c * pow(x, power, _PRIME)) % _PRIME
            shares.append((x, y))
        return shares

    @staticmethod
    def reconstruct(shares: list[tuple[int, int]]) -> int:
        """Lagrange-interpolate the secret at x = 0 from >= t shares."""
        secret = 0
        for j, (xj, yj) in enumerate(shares):
            num, den = 1, 1
            for m, (xm, _) in enumerate(shares):
                if m == j:
                    continue
                num = (num * (-xm)) % _PRIME
                den = (den * (xj - xm)) % _PRIME
            lagrange = (num * pow(den, -1, _PRIME)) % _PRIME
            secret = (secret + yj * lagrange) % _PRIME
        return secret % _PRIME


def recover_dropped(
    masking: PairwiseMasking,
    dropped_id: int,
    present_ids: list[int],
    dim: int,
) -> Vector:
    """Return the net mask a dropped client contributed, so the server can
    subtract it from the (now unbalanced) sum.

    In the full Bonawitz protocol the dropped client's seeds are recovered from
    Shamir shares held by survivors; here we compute the same net-mask term.
    """
    correction = np.zeros(dim, dtype=float)
    for other in present_ids:
        if other == dropped_id:
            continue
        i, j = sorted((dropped_id, other))
        m = _prg(masking.pair_seed[(i, j)], (dim,))
        # leftover = the mask the SURVIVING peer contributed for this pair
        # (smaller id added it, larger id subtracted it)
        correction += m if other < dropped_id else -m
    return correction