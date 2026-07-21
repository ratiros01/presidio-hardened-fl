# presidio-hardened-fl — coursework extension

A fork of [`presidio-v/presidio-hardened-fl`](https://github.com/presidio-v/presidio-hardened-fl)
for **PRES-EDU-CS-101 (Cloud Solutions with Cybersecurity & ML)**. It builds on
the baseline federated-learning + differential-privacy simulation and adds three
primitives from our group presentation on privacy-preserving ML in the cloud.

The upstream repo is MIT-licensed; these files extend it and do not modify the
original modules. All work lives on the `improve-accounting-secagg` branch.

## What this adds

The baseline is a clean, minimal FL + Gaussian DP simulation — a great starting
point. This fork builds on it in three directions:

- **Fuller privacy accounting.** The baseline tracks ε per round; this extends it
  to also compose δ, and adds an RDP ("moments") accountant for tighter Gaussian
  composition.
- **Secure Aggregation.** The baseline uses a trusted aggregator; this adds
  pairwise masking so the server only ever sees the sum of client updates.
- **Verifiable privacy claims.** A checker recomputes the (ε, δ) guarantee and
  reports it alongside the tool's number.

## New / patched files

| File | What it adds | Presentation slide |
|---|---|---|
| `src/presidio_fl/dp_patched.py` | `BasicComposition` composes **ε and δ**; new `RdpAccountant` for tight Gaussian composition | (ε,δ) definition · "DP-SGD composes better (n vs √n)" |
| `src/presidio_fl/secure_aggregation.py` | Pairwise masking (server sees only the sum) + Shamir dropout recovery | "The Protocol: Pairwise Masking" · "Guarantees and the Dropout Problem" |
| `src/presidio_fl/verify.py` | Recomputes (ε, δ) under basic + RDP accounting | "Verifying Privacy Claims in Untrusted Infrastructure" |
| `demo.py` | End-to-end FL + secure aggregation + DP + audit | — |

## Quick start

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# baseline training (unchanged upstream tool)
python generate_data.py --nodes 3 --samples-per-node 1000 --seed 42 --output data/
python train_federated.py --nodes 3 --rounds 10 --epsilon 1.0 --delta 1e-5 --model-out models/fl_strong.pkl

# recompute the privacy budget under fuller accounting
python -m presidio_fl.verify --epsilon 1.0 --delta 1e-5 --rounds 10

# full pipeline: FL + secure aggregation + DP + audit
python demo.py
```

## Example: the accounting audit

For ε = 1.0 / round, δ = 1e-5, over 10 rounds:

```
  REPORTED by tool   : eps = 10.0000   delta = 1.00e-05
  TRUE (basic comp.) : eps = 10.0000   delta = 1.00e-04   <- delta composed
  TRUE (RDP / tight) : eps =  3.0199   delta = 1.00e-04   <- RDP tightens eps 3.3x
```

Same Gaussian noise, three views of the budget: the per-round report, basic
composition (δ now accumulates), and the tighter RDP bound.

## Verified behaviour

- Pairwise masks cancel exactly in the server sum; individual updates stay hidden.
- A dropped client's masks are recovered so surviving clients still aggregate correctly.
- Shamir (t, n) secret sharing round-trips (split → reconstruct).
- RDP ε < basic ε for the same noise (the √n benefit); δ composes across rounds.

## Notes

- Secure aggregation *simulates* the Diffie–Hellman key exchange with a
  trusted-setup seed. The masking algebra and dropout recovery are the real
  protocol; only the key-exchange step is stubbed for the simulation.
- Synthetic IID data is used throughout, so reported accuracies illustrate the
  pipeline rather than a benchmark result.