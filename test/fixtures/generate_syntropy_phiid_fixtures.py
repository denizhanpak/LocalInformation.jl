"""Generate integrated information decomposition (PhiID) cross-check fixtures from syntropy.

Usage (run from a directory other than the syntropy checkout):

    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    uv run --no-project --with numpy --with scipy --with networkx \
        python -I generate_syntropy_phiid_fixtures.py SYNTROPY_DIR SYNTROPY_COMMIT OUT.json

SYNTROPY_DIR is the root of a syntropy checkout (the directory that contains the
`syntropy` package). Three kinds of cases are written:

- "discrete": a joint distribution over (past_1..past_n, future_1..future_n) given as
  distinct states with integer weights (the Julia tests repeat state k weights[k] times,
  so the empirical distribution of the samples is the distribution given to syntropy);
- "discrete_series": raw integer series x, y; the lagged samples
  (x_t, y_t, x_{t+tau}, y_{t+tau}), t = 1..N-tau, are counted into distinct states and
  counts (sorted lexicographically), and that empirical distribution is given to syntropy;
- "gaussian": raw real series x, y; the lagged sample matrix with rows
  (x_t, y_t, x_{t+tau}, y_{t+tau}), t = 1..N-tau, is passed to syntropy's Gaussian PhiID,
  which estimates the covariance as np.cov(data, ddof=0) (only MMI, average atoms).

syntropy conventions (as of the recorded commit):
- `integrated_information_decomposition(inputs, target, joint_distribution, rf)` (discrete)
  returns (pointwise, average); pointwise is None for "mmi"; for "isx"/"ipm" pointwise
  atoms are keyed by (input state, target state);
- the Gaussian version returns only the average dict for "mmi";
- atoms are (past antichain, future antichain), each a tuple of tuples of 0-based
  positions in `inputs` / `target`; the fixture stores them as lists of lists of 1-based
  variable indices (past and future both indexed 1..n);
- discrete values are in bits (log2); Gaussian values are in nats (natural log) in
  syntropy and are converted to bits here (divided by ln 2) so every value in the
  fixture is in bits.
"""

import json
import math
import sys
from fractions import Fraction

import numpy as np

def antichain_name(antichain):
    """r/x/y/s letter of a 2-variable antichain given as a tuple of 0-based tuples."""
    table = {((0,), (1,)): "r", ((0,),): "x", ((1,),): "y", ((0, 1),): "s"}
    return table[tuple(tuple(a) for a in antichain)]


def wb_leq(alpha, beta):
    """Williams-Beer order on antichains: alpha <= beta iff every B in beta contains some A in alpha."""
    return all(any(set(a) <= set(b) for a in alpha) for b in beta)


def main(syntropy_dir: str, commit: str, out_path: str) -> None:
    sys.path.insert(0, syntropy_dir)
    # syntropy/__init__.py reads its version from installed package metadata, which a
    # plain checkout does not have; answer that one lookup with the commit instead.
    import importlib.metadata as md

    real_version = md.version
    md.version = lambda name: commit if name == "syntropyx" else real_version(name)
    import syntropy  # noqa: E402
    from syntropy.discrete.decompositions import (  # noqa: E402
        integrated_information_decomposition as phiid_discrete,
    )
    from syntropy.discrete.shannon import mutual_information as mi_discrete  # noqa: E402
    from syntropy.gaussian.decompositions import (  # noqa: E402
        integrated_information_decomposition as phiid_gaussian,
    )
    from syntropy.gaussian.shannon import mutual_information as mi_gaussian  # noqa: E402
    from syntropy.lattices import load_lattice  # noqa: E402

    # The fixture's atom order is the node order of syntropy's pickled product lattice.
    atom_orders = {}
    for n in (2, 3):
        lattice = load_lattice(num_inputs=n, num_target=n)
        nodes = list(lattice.nodes)
        # Sanity: the pickled order is the product of Williams-Beer orders, and the
        # Mobius inversion visits every strict descendant before the node itself.
        for p in nodes:
            below = {q for q in nodes if q != p and wb_leq(q[0], p[0]) and wb_leq(q[1], p[1])}
            assert below == set(lattice.nodes[p]["descendants"]), p
            d_p = lattice.nodes[p]["distance_from_top"]
            assert all(lattice.nodes[q]["distance_from_top"] > d_p for q in below), p
        atom_orders[n] = nodes

    def atom_json(atom):
        past, future = atom
        return {
            "past": [[i + 1 for i in a] for a in past],
            "future": [[i + 1 for i in a] for a in future],
        }

    def case_header(n):
        entry = {"atom_order": [atom_json(a) for a in atom_orders[n]]}
        if n == 2:
            entry["atom_names"] = [
                antichain_name(a[0]) + "t" + antichain_name(a[1]) for a in atom_orders[n]
            ]
        return entry

    rng = np.random.default_rng(20261114)

    # ------------------------------------------------------------------ discrete
    discrete = {}
    bits = [(a, b) for a in (0, 1) for b in (0, 1)]
    # copy: x' = x, y' = y (independent uniform bits)
    discrete["copy"] = ([[x, y, x, y] for x, y in bits], [1] * 4)
    # swap: x' = y, y' = x
    discrete["swap"] = ([[x, y, y, x] for x, y in bits], [1] * 4)
    # parity-preserving: x' fresh uniform bit, y' chosen so that x' xor y' = x xor y
    st = [[x, y, xf, xf ^ x ^ y] for x, y in bits for xf in (0, 1)]
    discrete["parity"] = (st, [1] * len(st))
    # AND / noisy copy: x' = x AND y; y' = y kept with weight 3, flipped with weight 1
    st, wt = [], []
    for x, y in bits:
        for yf, w in ((y, 3), (1 - y, 1)):
            st.append([x, y, x & y, yf])
            wt.append(w)
    discrete["and_noisy"] = (st, wt)
    # random distribution over ternary variables, integer weights 0..4 (0 = left out)
    st, wt = [], []
    for s in range(81):
        v = [(s // 3**k) % 3 for k in (3, 2, 1, 0)]
        w = int(rng.integers(0, 5))
        if w > 0:
            st.append(v)
            wt.append(w)
    discrete["random_ternary"] = (st, wt)
    # 3 variables per side: three-bit copy x_i' = x_i
    st = [[a, b, c, a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)]
    discrete["copy3"] = (st, [1] * 8)
    # 3 variables per side: random binary distribution on a random 36-state support,
    # integer weights 1..4
    support = sorted(rng.choice(64, size=36, replace=False).tolist())
    st = [[(s >> k) & 1 for k in (5, 4, 3, 2, 1, 0)] for s in support]
    wt = [int(w) for w in rng.integers(1, 5, size=len(st))]
    discrete["random3_binary"] = (st, wt)

    # ------------------------------------------------------------------ discrete series
    def lagged_counts(x, y, tau):
        rows = np.column_stack([x[:-tau], y[:-tau], x[tau:], y[tau:]])
        uniq, counts = np.unique(rows, axis=0, return_counts=True)
        return [[int(v) for v in r] for r in uniq], [int(c) for c in counts]

    def binary_chain(n_samples, rng_):
        # x_{t+1} = y_t w.p. 0.8 (else a fresh fair bit);
        # y_{t+1} = x_t xor y_t w.p. 0.7 (else a fresh fair bit)
        x = np.zeros(n_samples, dtype=int)
        y = np.zeros(n_samples, dtype=int)
        x[0], y[0] = rng_.integers(0, 2, size=2)
        for t in range(n_samples - 1):
            x[t + 1] = y[t] if rng_.random() < 0.8 else rng_.integers(0, 2)
            y[t + 1] = (x[t] ^ y[t]) if rng_.random() < 0.7 else rng_.integers(0, 2)
        return x, y

    def ternary_chain(n_samples, rng_):
        # x_{t+1} = (x_t + y_t) mod 3 w.p. 0.7; y_{t+1} = x_t w.p. 0.6; else uniform
        x = np.zeros(n_samples, dtype=int)
        y = np.zeros(n_samples, dtype=int)
        x[0], y[0] = rng_.integers(0, 3, size=2)
        for t in range(n_samples - 1):
            x[t + 1] = (x[t] + y[t]) % 3 if rng_.random() < 0.7 else rng_.integers(0, 3)
            y[t + 1] = x[t] if rng_.random() < 0.6 else rng_.integers(0, 3)
        return x, y

    series_cases = {}
    x, y = binary_chain(400, rng)
    series_cases["markov_binary_tau1"] = (x, y, 1)
    x, y = binary_chain(500, rng)
    series_cases["markov_binary_tau2"] = (x, y, 2)
    x, y = ternary_chain(300, rng)
    series_cases["markov_ternary_tau1"] = (x, y, 1)

    # ------------------------------------------------------------------ run syntropy (discrete)
    out_cases = {}

    def run_discrete(states, weights):
        n = len(states[0]) // 2
        total = sum(weights)
        joint = {tuple(s): float(Fraction(w, total)) for s, w in zip(states, weights)}
        inputs = tuple(range(n))
        target = tuple(range(n, 2 * n))
        nodes = atom_orders[n]
        atoms, pointwise = {}, {}
        _, avg = phiid_discrete(
            inputs=inputs, target=target, joint_distribution=joint, redundancy_function="mmi"
        )
        atoms["mmi"] = [float(avg[a]) for a in nodes]
        ptw_mi, avg_mi = mi_discrete(idxs_x=inputs, idxs_y=target, joint_distribution=joint)
        assert abs(sum(atoms["mmi"]) - avg_mi) < 1e-9
        for rf in ("isx", "ipm"):
            ptw, avg = phiid_discrete(
                inputs=inputs, target=target, joint_distribution=joint, redundancy_function=rf
            )
            atoms[rf] = [float(avg[a]) for a in nodes]
            rows = []
            for s in states:
                key = (tuple(s[:n]), tuple(s[n:]))
                row = [float(ptw[key][a]) for a in nodes]
                # every pointwise decomposition sums to the local I(past; future)
                assert abs(sum(row) - ptw_mi[key]) < 1e-9, (rf, s)
                rows.append(row)
            pointwise[rf] = rows
            assert abs(sum(atoms[rf]) - avg_mi) < 1e-9
        return n, atoms, pointwise

    for name, (states, weights) in discrete.items():
        n, atoms, pointwise = run_discrete(states, weights)
        entry = {"kind": "discrete", "n": n, "states": states, "weights": weights}
        entry.update(case_header(n))
        entry["atoms"] = atoms
        entry["pointwise"] = pointwise
        out_cases[name] = entry

    for name, (x, y, tau) in series_cases.items():
        states, weights = lagged_counts(x, y, tau)
        assert sum(weights) == len(x) - tau
        n, atoms, pointwise = run_discrete(states, weights)
        entry = {
            "kind": "discrete_series",
            "n": n,
            "tau": tau,
            "series": [[int(v) for v in x], [int(v) for v in y]],
            "states": states,
            "weights": weights,
        }
        entry.update(case_header(n))
        entry["atoms"] = atoms
        entry["pointwise"] = pointwise
        out_cases[name] = entry

    # ------------------------------------------------------------------ gaussian
    def var1(n_samples, a_mat, noise_cov, rng_, burn=500):
        chol = np.linalg.cholesky(noise_cov)
        z = np.zeros(2)
        out = np.empty((n_samples, 2))
        for t in range(burn + n_samples):
            z = a_mat @ z + chol @ rng_.standard_normal(2)
            if t >= burn:
                out[t - burn] = z
        return out[:, 0].copy(), out[:, 1].copy()

    gaussian_cases = {}
    x, y = var1(2000, np.array([[0.5, 0.3], [0.0, 0.6]]), np.array([[1.0, 0.3], [0.3, 1.0]]), rng)
    gaussian_cases["var1_tau1"] = (x, y, 1)
    x, y = var1(2000, np.array([[0.4, -0.3], [0.5, 0.3]]), np.array([[1.0, -0.2], [-0.2, 0.5]]), rng)
    gaussian_cases["var1_tau2"] = (x, y, 2)

    ln2 = math.log(2.0)
    for name, (x, y, tau) in gaussian_cases.items():
        data = np.vstack([x[:-tau], y[:-tau], x[tau:], y[tau:]])  # channels x samples
        nodes = atom_orders[2]
        avg = phiid_gaussian(
            inputs=(0, 1), target=(2, 3), data=data, cov=None, redundancy_function="mmi"
        )
        mmi_bits = [float(avg[a]) / ln2 for a in nodes]
        total = mi_gaussian(idxs_x=(0, 1), idxs_y=(2, 3), cov=np.cov(data, ddof=0)) / ln2
        assert abs(sum(mmi_bits) - total) < 1e-9
        entry = {
            "kind": "gaussian",
            "n": 2,
            "tau": tau,
            "series": [[float(v) for v in x], [float(v) for v in y]],
        }
        entry.update(case_header(2))
        entry["atoms"] = {"mmi": mmi_bits}
        out_cases[name] = entry

    out = {
        "meta": {
            "syntropy_commit": commit,
            "unit": "bits",
            "generator": "test/fixtures/generate_syntropy_phiid_fixtures.py",
            "syntropy_version": getattr(syntropy, "__version__", "unknown"),
            "numpy": np.__version__,
            "python": sys.version.split()[0],
            "layout": (
                "states are [past_1..past_n, future_1..future_n]; atom_order entries are "
                "(past antichain, future antichain) with 1-based variable indices; "
                "atoms/pointwise values are in atom_order; pointwise row k <-> states[k]; "
                "atom_names (n = 2 only) are the phyid-style names, past letter + 't' + "
                "future letter, r = {1}{2}, x = {1}, y = {2}, s = {12}"
            ),
            "series_lagging": (
                "series cases use samples (x_t, y_t, x_{t+tau}, y_{t+tau}) for "
                "t = 1..N-tau (1-based); discrete_series states/weights are the distinct "
                "lagged states (lexicographically sorted) and their counts"
            ),
            "gaussian": (
                "syntropy.gaussian.decompositions.integrated_information_decomposition("
                "inputs=(0,1), target=(2,3), data=4 x (N-tau) lagged matrix, cov=None, "
                "redundancy_function='mmi'); covariance = np.cov(data, ddof=0) (mean-"
                "centred, divided by N-tau); MI = 0.5*(logdet + logdet - logdet) via "
                "slogdet in nats; values here are divided by ln 2 to give bits"
            ),
        },
        "cases": out_cases,
    }
    with open(out_path, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main(*sys.argv[1:4])
