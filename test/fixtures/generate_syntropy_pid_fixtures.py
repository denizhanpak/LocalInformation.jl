"""Generate PID cross-check fixtures from syntropy (Python reference implementation).

Usage (run from a directory other than the syntropy checkout):

    uv run --no-project --with numpy --with scipy --with networkx \
        python -I generate_syntropy_pid_fixtures.py SYNTROPY_DIR SYNTROPY_COMMIT OUT.json

SYNTROPY_DIR is the root of a syntropy checkout (the directory that contains the
`syntropy` package). For each test distribution, the fixture stores the states (the last
variable is the target), integer weights (the Julia tests repeat state k weights[k]
times, so the empirical distribution of the samples is the distribution given to
syntropy), and syntropy's average and pointwise PID atoms for the redundancy functions
i_sx ("isx") and i_pm ("ipm").

syntropy conventions (as of the recorded commit):
- `partial_information_decomposition(inputs, target, joint, redundancy_function)` returns
  (pointwise, average); pointwise atoms are keyed by (input state, target state);
- atoms are tuples of tuples of 0-based positions in `inputs`; the fixture stores them as
  lists of lists of 1-based source indices;
- values are in bits (log2).
"""

import json
import sys
from fractions import Fraction

import numpy as np


def main(syntropy_dir: str, commit: str, out_path: str) -> None:
    sys.path.insert(0, syntropy_dir)
    # syntropy/__init__.py reads its version from installed package metadata, which a
    # plain checkout does not have; answer that one lookup with the commit instead.
    import importlib.metadata as md

    real_version = md.version
    md.version = lambda name: commit if name == "syntropyx" else real_version(name)
    import syntropy  # noqa: E402
    from syntropy.discrete.decompositions import (  # noqa: E402
        partial_information_decomposition,
    )

    rng = np.random.default_rng(20261115)

    def uniform(states):
        return [list(s) for s in states], [1] * len(states)

    cases = {}
    bits = [(a, b) for a in (0, 1) for b in (0, 1)]
    cases["xor"] = uniform([(a, b, a ^ b) for a, b in bits])
    cases["and"] = uniform([(a, b, a & b) for a, b in bits])
    # Two-bit copy: the target is the pair (X1, X2), encoded as 2 X1 + X2.
    cases["copy"] = uniform([(a, b, 2 * a + b) for a, b in bits])
    cases["rdn"] = uniform([(a, a, a) for a in (0, 1)])

    # Random 3-source distribution: binary sources, ternary target, integer weights
    # 0..5 (states with weight 0 are left out).
    states, weights = [], []
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                for t in (0, 1, 2):
                    w = int(rng.integers(0, 6))
                    if w > 0:
                        states.append([a, b, c, t])
                        weights.append(w)
    cases["random3"] = (states, weights)

    # Random 4-source distribution: binary sources and target, weights 0..3.
    states, weights = [], []
    for s in range(32):
        v = [(s >> k) & 1 for k in range(5)]
        w = int(rng.integers(0, 4))
        if w > 0:
            states.append(v)
            weights.append(w)
    cases["random4"] = (states, weights)

    def atom_key(atom):
        return [[i + 1 for i in a] for a in atom]

    out_cases = {}
    for name, (states, weights) in cases.items():
        n_vars = len(states[0])
        total = sum(weights)
        joint = {tuple(s): float(Fraction(w, total)) for s, w in zip(states, weights)}
        inputs = tuple(range(n_vars - 1))
        target = (n_vars - 1,)
        entry = {"states": states, "weights": weights}
        for rf in ("isx", "ipm"):
            ptw, avg = partial_information_decomposition(
                inputs=inputs, target=target, joint_distribution=joint, redundancy_function=rf
            )
            entry[rf] = {
                "avg": [[atom_key(a), float(v)] for a, v in avg.items()],
                "ptw": [
                    [
                        [atom_key(a), float(v)]
                        for a, v in ptw[(tuple(s[:-1]), (s[-1],))].items()
                    ]
                    for s in states
                ],
            }
        out_cases[name] = entry

    out = {
        "meta": {
            "syntropy_commit": commit,
            "syntropy_version": getattr(syntropy, "__version__", "unknown"),
            "numpy": np.__version__,
            "unit": "bits",
            "layout": "states are [source_1, ..., source_n, target]",
        },
        "cases": out_cases,
    }
    with open(out_path, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main(*sys.argv[1:4])
