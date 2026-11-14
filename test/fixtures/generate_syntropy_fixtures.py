"""Generate cross-check fixtures from syntropy (Python reference implementation).

Usage (run from a directory other than the syntropy checkout):

    uv run --no-project --with numpy --with scipy --with networkx \
        python -I generate_syntropy_fixtures.py SYNTROPY_DIR SYNTROPY_COMMIT OUT.json

SYNTROPY_DIR is the root of a syntropy checkout (the directory that contains the
`syntropy` package). The fixture stores the input data and syntropy's pointwise values,
so the Julia tests do not need Python.

syntropy conventions (as of the recorded commit):
- data are (n_variables, n_samples) arrays;
- Gaussian local entropies are -log N(x; mean, cov) with cov = np.cov(data, ddof=0)
  (maximum-likelihood covariance), in nats;
- knn.differential_entropy returns (pointwise (1, N), mean) with
  h_i = -psi(k) + psi(N) + d * log(2 * rho_i), rho_i the distance to the k-th neighbour
  (excluding the point itself) in the p-norm, in nats.
"""

import json
import sys

import numpy as np


def main(syntropy_dir: str, commit: str, out_path: str) -> None:
    sys.path.insert(0, syntropy_dir)
    # syntropy/__init__.py reads its version from installed package metadata, which a
    # plain checkout does not have; answer that one lookup with the commit instead.
    import importlib.metadata as md

    real_version = md.version
    md.version = lambda name: commit if name == "syntropyx" else real_version(name)
    import syntropy  # noqa: E402
    from syntropy import gaussian, knn  # noqa: E402

    rng = np.random.default_rng(20261114)

    # Gaussian: 4 correlated variables.
    n_vars, n = 4, 300
    mixing = rng.standard_normal((n_vars, n_vars)) + 0.5 * np.eye(n_vars)
    g = mixing @ rng.standard_normal((n_vars, n)) + rng.standard_normal((n_vars, 1))

    def ptw(a):
        return np.asarray(a, dtype=float).reshape(-1).tolist()

    gaussian_fixture = {
        "data": g.tolist(),
        "h_all": ptw(gaussian.local_differential_entropy(g)),
        "h_01": ptw(gaussian.local_differential_entropy(g, idxs=(0, 1))),
        "h_2": ptw(gaussian.local_differential_entropy(g, idxs=(2,))),
        "mi_0_12": ptw(gaussian.local_mutual_information((0,), (1, 2), g)),
        "tc": ptw(gaussian.local_total_correlation(g)),
        "dtc": ptw(gaussian.local_dual_total_correlation(g)),
        "oinfo": ptw(gaussian.local_o_information(g)),
        "sinfo": ptw(gaussian.local_s_information(g)),
    }

    # k-NN (Kozachenko-Leonenko / Kraskov form) differential entropy.
    k_data = rng.standard_normal((2, 250))
    k_data[1] += 0.7 * k_data[0]
    k = 3
    knn_fixture = {"data": k_data.tolist(), "k": k}
    for name, p in (("pinf", np.inf), ("p2", 2.0)):
        local, avg = knn.differential_entropy(k_data, k=k, p=p)
        knn_fixture[f"h_{name}"] = ptw(local)
        knn_fixture[f"mean_{name}"] = float(avg)

    out = {
        "meta": {
            "syntropy_commit": commit,
            "syntropy_version": getattr(syntropy, "__version__", "unknown"),
            "numpy": np.__version__,
            "unit": "nats",
            "layout": "data are (n_variables, n_samples)",
        },
        "gaussian": gaussian_fixture,
        "knn": knn_fixture,
    }
    with open(out_path, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main(*sys.argv[1:4])
