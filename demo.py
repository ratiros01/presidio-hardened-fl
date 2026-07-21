"""End-to-end demo of the patch: FL + Secure Aggregation + fixed accounting.

Runs a small federated logistic-regression training (mirroring the repo's
setup) and shows, side by side:

  * that Secure Aggregation reproduces the exact FedAvg result while hiding
    each client's individual update, and
  * the difference between the tool's REPORTED privacy budget and the TRUE
    budget under correct (basic and RDP) accounting.

Run:  python demo.py
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier

from presidio_fl.dp_patched import GaussianMechanism
from presidio_fl.secure_aggregation import PairwiseMasking
from presidio_fl.verify import audit, print_report

N_NODES = 3
ROUNDS = 10
EPS_PER_ROUND = 1.0
DELTA = 1e-5
CLIP = 1.0
SEED = 42


def make_nodes():
    X, y = make_classification(
        n_samples=3000, n_features=20, n_informative=10,
        n_redundant=2, class_sep=2.0, flip_y=0.01, random_state=SEED,
    )
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]
    parts = np.array_split(np.arange(len(y)), N_NODES)
    nodes = [(X[p], y[p]) for p in parts]
    return nodes, X[:600], y[:600]


def local_update(Xn, yn, global_w):
    clf = SGDClassifier(loss="log_loss", max_iter=100, warm_start=True, random_state=SEED, tol=1e-4)
    clf.classes_ = np.array([0, 1])
    clf.coef_ = global_w["coef"].copy()
    clf.intercept_ = global_w["intercept"].copy()
    clf.fit(Xn, yn)
    return {"coef": clf.coef_.copy(), "intercept": clf.intercept_.copy()}


def flatten(w):
    return np.concatenate([w["coef"].ravel(), w["intercept"].ravel()])


def unflatten(vec, template):
    c = template["coef"].size
    return {"coef": vec[:c].reshape(template["coef"].shape),
            "intercept": vec[c:].reshape(template["intercept"].shape)}


def main() -> None:
    nodes, X_test, y_test = make_nodes()
    dim = nodes[0][0].shape[1] + 1  # coef + intercept
    global_w = {"coef": np.zeros((1, nodes[0][0].shape[1])), "intercept": np.zeros(1)}

    mech = GaussianMechanism(EPS_PER_ROUND, DELTA, sensitivity=CLIP)
    masking = PairwiseMasking(N_NODES, dim)

    print(f"Federated demo: {N_NODES} nodes, {ROUNDS} rounds, eps/round={EPS_PER_ROUND}, delta={DELTA}\n")

    for r in range(1, ROUNDS + 1):
        old = flatten(global_w)
        updates = [flatten(local_update(Xn, yn, global_w)) - old for Xn, yn in nodes]

        # clip each update to sensitivity
        updates = [u * min(1.0, CLIP / (np.linalg.norm(u) + 1e-8)) for u in updates]

        # --- Secure Aggregation: server sees only the sum, not each update ---
        masked = {i: masking.mask(i, u) for i, u in enumerate(updates)}
        agg = masking.server_sum(masked) / N_NODES

        # sanity: secure sum == plain FedAvg average
        plain = np.mean(updates, axis=0)
        assert np.allclose(agg, plain), "secure aggregation must match FedAvg"

        # --- DP noise on the aggregate ---
        noised = mech.add_noise({"u": agg}, rng=np.random.default_rng(SEED + r))["u"]
        global_w = unflatten(old + noised, global_w)

    # evaluate
    clf = SGDClassifier(loss="log_loss")
    clf.classes_ = np.array([0, 1])
    clf.coef_, clf.intercept_ = global_w["coef"], global_w["intercept"]
    acc = clf.score(X_test, y_test)
    print(f"Final accuracy: {acc:.3f}")
    print("Secure aggregation matched FedAvg every round (individual updates never exposed).")

    # --- budget: reported vs true ---
    a = audit(EPS_PER_ROUND, DELTA, ROUNDS)
    print_report(a)

    r, b, t = a["reported"], a["true_basic"], a["true_rdp"]
    print("What this shows")
    print("-" * 64)
    print(f"  1. delta composes too: reported {r['delta']:.0e} per round becomes")
    print(f"     {b['delta']:.0e} over {ROUNDS} rounds ({b['delta']/r['delta']:.0f}x) once composed.")
    print(f"  2. tighter accounting: same Gaussian noise, linear comp gives")
    print(f"     eps={b['epsilon']:.1f}, but RDP gives eps={t['epsilon']:.1f} "
          f"({b['epsilon']/t['epsilon']:.1f}x lower).")
    print(f"  3. secure aggregation: server saw only the sum every round,")
    print(f"     never an individual client's update.")
    print("-" * 64)
    print("  => real privacy is BETTER than the loose eps, and delta must be composed.")
    print()


if __name__ == "__main__":
    main()