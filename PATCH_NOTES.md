# presidio-hardened-fl — patch (three fixes)

A fork/patch of `presidio-v/presidio-hardened-fl` that builds on the baseline
and adds three primitives from our presentation. Each patch maps to a slide.
Original repo is MIT-licensed; these files extend it and do not overwrite the
upstream.

## Extending the baseline
The original repo is a clean, minimal FL + Gaussian DP simulation — a great
starting point. This patch builds on it in three directions:

- **Fuller privacy accounting.** The baseline tracks ε per round; we extend it
  to also compose δ, and add an RDP ("moments") accountant for tighter Gaussian
  composition.
- **Adding Secure Aggregation.** The baseline uses a trusted aggregator; we add
  pairwise masking so the server only ever sees the sum.
- **Making the privacy claim verifiable.** We add a checker that recomputes the
  true (ε, δ) and reports it alongside the tool's number.

## The three additions

| File | Fix | Slide |
|---|---|---|
| `dp_patched.py` | `BasicComposition` now composes **ε and δ**; new `RdpAccountant` gives tight Gaussian composition | "(ε,δ) cards", "DP-SGD composes better (n vs √n)" |
| `secure_aggregation.py` | Pairwise masking (server sees only the sum) + Shamir dropout recovery | "The Protocol: Pairwise Masking", "Guarantees and the Dropout Problem" |
| `verify.py` | Recomputes true (ε,δ) three ways and flags claim ≠ delivered | "Verifying Privacy Claims in Untrusted Infrastructure" |

## Verifier output (ε=1.0/round, δ=1e-5, 10 rounds)

```
  REPORTED by tool   : eps = 10.0000   delta = 1.00e-05
  TRUE (basic comp.) : eps = 10.0000   delta = 1.00e-04   <- delta now composed
  TRUE (RDP / tight) : eps =  3.0199   delta = 1.00e-04   <- RDP tightens eps 3.3x
```

Run:
```
python -m presidio_fl.verify --epsilon 1.0 --delta 1e-5 --rounds 10
```

## Verified behaviour
- Pairwise masks cancel exactly in the server sum; individual updates stay hidden.
- A dropped client's masks are recovered so surviving clients still aggregate correctly.
- Shamir (t,n) split/reconstruct round-trips.
- RDP ε < basic ε for the same noise (the √n win), and δ now composes.