"""Verification checker: does the REPORTED privacy budget match the TRUTH?

This makes slide 18 ("Verifying privacy claims in untrusted infrastructure")
concrete for THIS repo. The baseline tool reports a privacy budget of
`n_rounds * epsilon`. This checker recomputes the guarantee three ways so the
reported budget can be compared with fuller (basic and RDP) accounting.

Usage:
    python -m presidio_fl.verify --epsilon 1.0 --delta 1e-5 --rounds 10
    python -m presidio_fl.verify --log logs/security.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from presidio_fl.dp_patched import BasicComposition, GaussianMechanism, RdpAccountant


def audit(epsilon_round: float, delta_round: float, rounds: int) -> dict:
    """Return reported vs. true (basic) vs. true (RDP) accounting."""
    # what the baseline tool reports
    reported_eps = rounds * epsilon_round
    reported_delta = delta_round  # baseline reports per-round delta

    # true basic composition (eps AND delta accumulate)
    basic = BasicComposition(epsilon_round, delta_round, rounds)
    for _ in range(rounds):
        basic.spend_round()

    # true tight (RDP) accounting for the same Gaussian noise
    mech = GaussianMechanism(epsilon_round, delta_round, sensitivity=1.0)
    z = mech.noise_multiplier
    rdp = RdpAccountant(noise_multiplier=z)
    rdp.step(rounds)
    # report RDP eps at the composed delta (fair comparison to basic)
    rdp_eps = rdp.get_epsilon(delta=basic.total_delta())

    return {
        "rounds": rounds,
        "noise_multiplier_z": z,
        "reported": {"epsilon": reported_eps, "delta": reported_delta},
        "true_basic": {"epsilon": basic.total_epsilon(), "delta": basic.total_delta()},
        "true_rdp": {"epsilon": rdp_eps, "delta": basic.total_delta()},
    }


def print_report(a: dict) -> None:
    print(f"\nPrivacy audit over {a['rounds']} rounds  (noise multiplier z = {a['noise_multiplier_z']:.3f})")
    print("-" * 64)
    r, b, t = a["reported"], a["true_basic"], a["true_rdp"]
    print(f"  REPORTED by tool   : eps = {r['epsilon']:.4f}   delta = {r['delta']:.2e}")
    print(f"  TRUE (basic comp.) : eps = {b['epsilon']:.4f}   delta = {b['delta']:.2e}")
    print(f"  TRUE (RDP / tight) : eps = {t['epsilon']:.4f}   delta = {t['delta']:.2e}")
    print("-" * 64)

    flags = []
    if not math.isclose(r["delta"], b["delta"]):
        factor = b["delta"] / r["delta"] if r["delta"] else float("inf")
        flags.append(
            f"delta: tool reports {r['delta']:.2e}; composing delta gives "
            f"{b['delta']:.2e} ({factor:.0f}x) over {a['rounds']} rounds."
        )
    if t["epsilon"] < b["epsilon"] - 1e-9:
        flags.append(
            f"eps: basic composition gives {b['epsilon']:.3f}; RDP tightens it to "
            f"{t['epsilon']:.3f} for the same noise ({b['epsilon'] / t['epsilon']:.1f}x)."
        )
    if flags:
        print("  [NOTE] fuller accounting:")
        for f in flags:
            print(f"    - {f}")
    else:
        print("  [OK] reported budget matches the fuller accounting.")
    print()


def _from_log(path: Path) -> tuple[float, float, int]:
    eps = delta = None
    rounds = 0
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        if rec.get("event") == "training_start":
            eps, delta = rec.get("epsilon"), rec.get("delta")
        if rec.get("event") == "privacy_budget_spent":
            rounds += 1
    if eps is None or delta is None:
        raise ValueError("log has no DP training_start event with epsilon/delta")
    return float(eps), float(delta), rounds


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Audit reported vs. true DP budget.")
    p.add_argument("--epsilon", type=float, help="epsilon per round")
    p.add_argument("--delta", type=float, default=1e-5, help="delta per round")
    p.add_argument("--rounds", type=int, help="number of rounds")
    p.add_argument("--log", type=str, help="parse logs/security.jsonl instead")
    args = p.parse_args(argv)

    if args.log:
        eps, delta, rounds = _from_log(Path(args.log))
    else:
        if args.epsilon is None or args.rounds is None:
            p.error("provide --epsilon and --rounds, or --log")
        eps, delta, rounds = args.epsilon, args.delta, args.rounds

    print_report(audit(eps, delta, rounds))


if __name__ == "__main__":
    main()