# Supported estimators

| Estimator (Associations / ComplexityMeasures) | Measures | CPU | GPU |
|:--|:--|:-:|:-:|
| `JointProbabilities(def, CodifyVariables(o))` (plug-in) | `MIShannon`, `CMIShannon`, higher-order | yes | computed on the CPU |
| `EntropyDecomposition(def, PlugIn(Shannon()), CodifyVariables(o))` | `MIShannon`, `CMIShannon`, higher-order | yes | computed on the CPU |
| `EntropyDecomposition(def, Kraskov(; k, w))` | `MIShannon`, `CMIShannon`, higher-order | yes | yes |
| `EntropyDecomposition(def, KozachenkoLeonenko(; w))` | `MIShannon`, `CMIShannon`, higher-order | yes | yes |
| `KSG1(; k, w, metric_marginals)`, `KSG2(; k, w, metric_marginals)` | `MIShannon` | yes | yes |
| `FPVP(; k, w)` | `CMIShannon` | yes | yes |
| `GaussianDecomposition(def; copula)` | higher-order | yes | yes |
| `GaussianMI`, `GaussianCMI` | `MIShannon`, `CMIShannon` | yes | yes |
| `local_entropy(Kraskov(...), x)`, `local_entropy(KozachenkoLeonenko(...), x)` | Shannon differential entropy | yes | yes |
| `local_entropy(PlugIn(Shannon()), o, x)` | Shannon entropy | yes | computed on the CPU |

"Higher-order" is `TotalCorrelation`, `DualTotalCorrelation`, `OInformation` and
`SInformation`. Anything else throws an `ArgumentError` that names the estimator and lists
the supported ones. Bias-corrected entropy estimators (`MillerMadow`, `ChaoShen`,
`Jackknife`, `Schuermann`, ...) are not supported: their estimates are not sample means of
per-sample terms.

Notes:

- **Discrete** local entropies are `-log(n(ω_t)/N)` from the counts of the joint outcomes,
  with exactly the encoding Associations uses. With an embedding outcome space (e.g.
  `OrdinalPatterns`) there is one local value per encoded symbol.
- **Gaussian** local entropies are `½(d log 2π + log det Σ + N/(N-1) z'z)`, with Σ the
  `N - 1`-normalised covariance that Associations uses and `z = L \ (x - μ)`. The factor
  `N/(N-1)` makes the mean exactly Associations' closed form. This equals
  `-log 𝒩(x; μ, Σ̂) + (d/2) log(N/(N-1))` with the maximum-likelihood covariance Σ̂; the
  constant cancels in every mutual information and higher-order measure.
- **k-NN** local entropies are the per-sample terms of ComplexityMeasures' `Kraskov`
  (`-ψ(k) + ψ(N) + log V_D + D log ρ_t`) and `KozachenkoLeonenko`, with the same Euclidean
  k-d tree search and Theiler window.
- **KSG1 / KSG2 / FPVP** local values are the per-sample terms of Associations' estimators,
  e.g. `ψ(k) + ψ(N) - ψ(n_x(t)) - ψ(n_y(t))` for `KSG1`, with Associations' conventions:
  Chebyshev joint k-th neighbour outside the Theiler window, marginal counts `≤ ε_t`
  including the point itself and without Theiler window. (Kraskov et al. count strictly
  inside ε, as does syntropy; the difference is exactly accounted for in the tests.)
