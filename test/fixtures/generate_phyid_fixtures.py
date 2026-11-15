"""Generate local MMI PhiID cross-check fixtures from phyid (Imperial-MIND-lab).

Usage (run from a directory other than the phyid checkout):

    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    uv run --no-project --with numpy --with scipy \
        python -I generate_phyid_fixtures.py PHYID_DIR PHYID_COMMIT OUT.json

PHYID_DIR is the root of a checkout of https://github.com/Imperial-MIND-lab/integrated-info-decomp
(the directory that contains the `phyid` package). phyid is imported without running its
`__init__.py` (which only asks versioneer/git for a version string), by registering an
empty `phyid` package whose path is the checkout's `phyid/` directory.

For each case the fixture stores the raw series, tau, phyid's 16 atom names in order,
phyid's local atoms (one row per lagged sample, `calc_PhiID(x, y, tau, kind, "MMI")`) and
their sample means (phyid returns only local atoms; its tests and the MATLAB toolbox
average them). phyid's lagging is `x[:-tau]`, `x[tau:]`, so row k is the sample whose past
is at 1-based time t = k (k = 1..N-tau).

Units: phyid's Gaussian values are in nats (natural log) and its discrete values in bits
(log2). Every value in the fixture is in bits: Gaussian values are divided by ln 2.

Gaussian preprocessing in phyid: X = [x_t, y_t, x_{t+tau}, y_{t+tau}] (4 x (N-tau)); each
row divided by its own sample std (ddof=1), not centred; for every variable subset S the
local entropy is -ln N(X_S; mean(X_S), np.cov(X)[S,S]) with np.cov's default ddof=1.
Discrete preprocessing: each of the 4 lagged vectors is binarised as (v > mean(v)); local
entropies are -log2 of empirical frequencies over the N-tau lagged samples. The series
used here are 0/1 with both values present, so binarisation leaves them unchanged.

The generator also reruns the Gaussian cases with the maximum-likelihood covariance
(np.cov(X, ddof=0)) substituted inside phyid and checks the exact relation

    local_ml = average + N'/(N'-1) * (local_phyid - average),   N' = N - tau,

which is how a local-Gaussian implementation using the ML covariance (e.g. syntropy, or
LocalInformation.jl's Gaussian local entropies) relates to phyid's local atoms.
"""

import json
import math
import os
import sys
import types

import numpy as np


def main(phyid_dir: str, commit: str, out_path: str) -> None:
    pkg = types.ModuleType("phyid")
    pkg.__path__ = [os.path.join(phyid_dir, "phyid")]
    sys.modules["phyid"] = pkg
    import phyid.calculate as pcalc  # noqa: E402
    from phyid.calculate import calc_PhiID  # noqa: E402
    from phyid.utils import PhiID_atoms_abbr  # noqa: E402

    names = list(PhiID_atoms_abbr)
    rng = np.random.default_rng(20261116)
    ln2 = math.log(2.0)

    def var1(n_samples, a_mat, noise_cov, rng_, burn=500):
        chol = np.linalg.cholesky(noise_cov)
        z = np.zeros(2)
        out = np.empty((n_samples, 2))
        for t in range(burn + n_samples):
            z = a_mat @ z + chol @ rng_.standard_normal(2)
            if t >= burn:
                out[t - burn] = z
        return out[:, 0].copy(), out[:, 1].copy()

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

    cases = {}
    x, y = var1(1000, np.array([[0.5, 0.3], [0.0, 0.6]]), np.array([[1.0, 0.3], [0.3, 1.0]]), rng)
    cases["var1_tau1"] = ("gaussian", x, y, 1)
    x, y = var1(1000, np.array([[0.4, -0.3], [0.5, 0.3]]), np.array([[1.0, -0.2], [-0.2, 0.5]]), rng)
    cases["var1_tau2"] = ("gaussian", x, y, 2)
    x, y = var1(1000, np.array([[0.2, 0.7], [0.6, -0.1]]), np.array([[1.0, 0.6], [0.6, 2.0]]), rng)
    cases["var1_coupled_tau1"] = ("gaussian", x, y, 1)
    x, y = binary_chain(500, rng)
    cases["markov_binary_tau1"] = ("discrete", x, y, 1)
    x, y = binary_chain(500, rng)
    cases["markov_binary_tau2"] = ("discrete", x, y, 2)

    class _NumpyMLCov:
        """numpy stand-in for phyid.calculate whose cov uses the ML (ddof=0) normalisation."""

        def __getattr__(self, attr):
            return getattr(np, attr)

        @staticmethod
        def cov(m):
            return np.cov(m, ddof=0)

    max_ml_dev = 0.0
    out_cases = {}
    for name, (kind, x, y, tau) in cases.items():
        atoms_res, calc_res = calc_PhiID(x, y, tau, kind=kind, redundancy="MMI")
        local = np.column_stack([atoms_res[a] for a in names])  # (N - tau) x 16
        n_rows = len(x) - tau
        assert local.shape == (n_rows, 16) and np.all(np.isfinite(local))
        # local atoms sum to the local I(past; future) of each sample
        assert np.allclose(local.sum(axis=1), calc_res["I_res"]["I_xytab"], atol=1e-10)
        average = local.mean(axis=0)
        if kind == "discrete":
            for v in (x[:-tau], y[:-tau], x[tau:], y[tau:]):
                assert set(np.unique(v)) == {0, 1}  # binarisation is the identity
        else:
            pcalc.np = _NumpyMLCov()
            try:
                atoms_ml, _ = calc_PhiID(x, y, tau, kind=kind, redundancy="MMI")
            finally:
                pcalc.np = np
            local_ml = np.column_stack([atoms_ml[a] for a in names])
            predicted = average + n_rows / (n_rows - 1) * (local - average)
            max_ml_dev = max(max_ml_dev, float(np.max(np.abs(local_ml - predicted))))
            assert np.allclose(local_ml.mean(axis=0), average, atol=1e-10)
        scale = 1.0 / ln2 if kind == "gaussian" else 1.0
        out_cases[name] = {
            "kind": "gaussian" if kind == "gaussian" else "discrete_series",
            "tau": tau,
            "series": (
                [[float(v) for v in x], [float(v) for v in y]]
                if kind == "gaussian"
                else [[int(v) for v in x], [int(v) for v in y]]
            ),
            "redundancy": "MMI",
            "atom_names": names,
            "average": [float(v) * scale for v in average],
            "local": [[float(v) * scale for v in row] for row in local],
            "local_sample_indices": list(range(1, n_rows + 1)),
        }
    assert max_ml_dev < 1e-9, max_ml_dev

    out = {
        "meta": {
            "phyid_source": "https://github.com/Imperial-MIND-lab/integrated-info-decomp (not on PyPI)",
            "phyid_commit": commit,
            "unit": "bits",
            "unit_note": (
                "phyid's Gaussian values are in nats; they are divided by ln 2 here. "
                "phyid's discrete values are already in bits."
            ),
            "generator": "test/fixtures/generate_phyid_fixtures.py",
            "numpy": np.__version__,
            "python": sys.version.split()[0],
            "call": "phyid.calculate.calc_PhiID(x, y, tau, kind='gaussian'|'discrete', redundancy='MMI')",
            "preprocessing": (
                "Rows: past = x[:-tau], y[:-tau]; future = x[tau:], y[tau:], i.e. samples "
                "(x_t, y_t, x_{t+tau}, y_{t+tau}) for t = 1..N-tau (1-based); local row k "
                "has past time t = k. Gaussian: each of the 4 rows is divided by its sample "
                "std (ddof=1, no centring); local entropy of subset S = -ln N(X_S; sample "
                "mean, np.cov ddof=1); local MI from local entropies. Discrete: each lagged "
                "vector binarised as v > mean(v) (identity for these 0/1 series); local "
                "entropy = -log2 empirical frequency over the N-tau samples. 'average' = "
                "mean of the local rows (phyid returns only local atoms)."
            ),
            "local_mmi": (
                "Each redundancy (6 single-target PIDs, 1 double redundancy) is the local MI "
                "of the source/target pair with the smallest sample-mean local MI; ties: "
                "single-target picks the second argument (strict <), double redundancy picks "
                "the first of [I(x_p;x_f), I(x_p;y_f), I(y_p;x_f), I(y_p;y_f)] (np.argmin). "
                "Local atoms = per-sample solve of the 16x16 Mobius system."
            ),
            "ml_covariance_relation": (
                "With the ML covariance (ddof=0) the local atoms are "
                "average + N'/(N'-1) * (local - average), N' = N - tau; checked here with "
                f"max abs deviation {max_ml_dev:.3e} (nats)"
            ),
        },
        "cases": out_cases,
    }
    with open(out_path, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main(*sys.argv[1:4])
