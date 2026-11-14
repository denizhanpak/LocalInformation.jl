"""Generate KSG / Frenzel-Pompe cross-check fixtures from syntropy.

Usage (run from a directory other than the syntropy checkout):

    uv run --no-project --with numpy --with scipy --with networkx \
        python -I generate_syntropy_ksg_fixtures.py SYNTROPY_DIR SYNTROPY_COMMIT OUT.json

Companion of generate_syntropy_fixtures.py (same version-lookup stub). Stores the input
data and syntropy's pointwise values in nats, so the Julia tests do not need Python.

syntropy conventions (as of the recorded commit), all with p = inf (Chebyshev):
- data are (n_variables, n_samples) arrays;
- eps_i = distance to the k-th neighbour in the joint space (cKDTree.query, k + 1, self
  excluded by position);
- knn.mutual_information(..., algorithm=1): psi(k) + psi(N) - psi(n_x + 1) - psi(n_y + 1),
  n = #{j != i : d < eps_i} (strict, via np.nextafter);
- knn.mutual_information(..., algorithm=2): psi(k) - 1/k + psi(N) - psi(n_x) - psi(n_y),
  n = #{j != i : d <= eps_x(i)}, eps_x(i) = max over the k joint neighbours of the marginal
  distance;
- knn.conditional_mutual_information: psi(k) + psi(n_z + 1) - psi(n_xz + 1) - psi(n_yz + 1),
  strict counts as for algorithm 1.
"""

import json
import sys

import numpy as np


def main(syntropy_dir: str, commit: str, out_path: str) -> None:
    sys.path.insert(0, syntropy_dir)
    import importlib.metadata as md

    real_version = md.version
    md.version = lambda name: commit if name == "syntropyx" else real_version(name)
    import syntropy  # noqa: E402
    from syntropy import knn  # noqa: E402

    rng = np.random.default_rng(20261115)
    n_vars, n = 4, 400
    mixing = rng.standard_normal((n_vars, n_vars)) + np.eye(n_vars)
    data = mixing @ rng.standard_normal((n_vars, n))

    def ptw(a):
        return np.asarray(a, dtype=float).reshape(-1).tolist()

    x, y = (0, 1), (2,)  # 2-D x, 1-D y
    k = 3
    out = {
        "meta": {
            "syntropy_commit": commit,
            "syntropy_version": getattr(syntropy, "__version__", "unknown"),
            "numpy": np.__version__,
            "unit": "nats",
            "layout": "data are (n_variables, n_samples)",
            "p": "inf",
        },
        "data": data.tolist(),
        "k": k,
        "mi_x": list(x),
        "mi_y": list(y),
        "ksg1": ptw(knn.mutual_information(x, y, k, data, algorithm=1, p=np.inf)[0]),
        "ksg2_k1": ptw(knn.mutual_information(x, y, 1, data, algorithm=2, p=np.inf)[0]),
        "ksg2": ptw(knn.mutual_information(x, y, k, data, algorithm=2, p=np.inf)[0]),
        # I(x0; x2 | x1, x3)
        "cmi_x": [0],
        "cmi_y": [2],
        "cmi_z": [1, 3],
        "cmi": ptw(knn.conditional_mutual_information((0,), (2,), (1, 3), k, data, p=np.inf)[0]),
    }
    with open(out_path, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main(*sys.argv[1:4])
