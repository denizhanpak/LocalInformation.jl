# LocalInformation.jl

[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://denizhanpak.github.io/LocalInformation.jl/dev/)

Local (per-sample, pointwise) information measures, as a companion to
[Associations.jl](https://github.com/kahaaga/Associations.jl).

Associations.jl estimates a measure such as the mutual information or the O-information as
one number. Most of these estimates are sample means of a per-sample quantity, e.g. the
local mutual information `i(x_t; y_t) = log p(x_t, y_t) / (p(x_t) p(y_t))`. LocalInformation.jl
returns those per-sample values, for the same estimator objects:

```julia
using Associations, LocalInformation, Statistics

x = randn(10_000); y = x .+ randn(10_000); z = y .+ randn(10_000)

est = GaussianDecomposition(OInformation())
ω = local_association(est, x, y, z)       # Vector, one value per sample, in bits
mean(ω) ≈ association(est, x, y, z)        # true
```

## Features

- Local mutual information, conditional mutual information and higher-order measures
  (total correlation, dual total correlation, O- and S-information), for discrete, k-NN,
  KSG and Gaussian estimators.
- Local Shannon entropies (`local_entropy`).
- Per-sample PID atoms with the pointwise redundancy functions `ISx` and `IPM`.
- Integrated information decomposition (ΦID) of 2 or 3 variables, average and per-sample
  atoms. It supports the MMI double redundancy for every estimator family, and `ISx` and
  `IPM` for discrete data. Two-variable results also give the six information-dynamics
  modes.
- GPU backends through KernelAbstractions.jl.

## The invariant

For every supported estimator `est`,

```julia
mean(local_association(est, x...)) ≈ association(est, x...)
mean(local_entropy(est, x)) ≈ information(est, x)            # differential
mean(local_entropy(est, o, x)) ≈ information(est, o, x)      # discrete
```

to floating-point rounding (and to `Float32` rounding on a GPU). The test suite checks this
for every method, in bases 2 and ℯ, with vector and multi-column `StateSpaceSet` inputs.
Values are in the unit `est.definition.base` (bits by default). The decompositions (PID
and ΦID) return per-sample atoms, and the invariant holds for every atom `a`:
`mean(atom(local_association(est, x...), a)) ≈ atom(association(est, x...), a)`.

## Supported estimators

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

## GPU

`backend` takes a [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl)
backend. CUDA support is a package extension, loaded by `using CUDA`:

```julia
using Associations, LocalInformation, CUDA

X = [randn(10^6) for _ in 1:6]
est = GaussianDecomposition(OInformation())
ω = local_association(est, X...; backend = CUDABackend())                       # Vector{Float32}
ω64 = local_association(est, X...; backend = CUDABackend(), precision = Float64)
```

`precision = nothing` (the default) computes in `Float64` on the CPU and `Float32` on a GPU.
The result is always a host `Vector`. On the GPU, the Gaussian path computes the covariance,
its Cholesky factors and log-determinants on the CPU in `Float64`, and the triangular solves
and reductions over samples on the device. Discrete estimators count on the CPU whatever
the backend. k-NN searches and range counts on a GPU use an exact brute-force
KernelAbstractions kernel (O(N²) work, O(N) memory; metrics `Chebyshev`, `Euclidean`,
`Cityblock`). In `Float64` the neighbour counts equal the CPU's on continuous data; in
`Float32` a distance that rounds across a radius can shift a count by one (≲ 0.1% of samples
in the tests), which changes that sample's local value by up to `ψ(n+1) - ψ(n) = 1/n`.

## Partial information decomposition

Two redundancy functions with a pointwise form plug into Associations' PID
(`PartialInformationDecomposition`, through its `Associations.pid_redundancies` hook), for
discrete data (`JointProbabilities` with `CodifyVariables`, plug-in probabilities). For a
sample with target outcome `t` and joint outcome `a_j` of each source set `A_j` of a lattice
node `α = {A_1, …, A_k}`:

- `ISx()`, the shared-exclusion redundancy of Makkeh, Gutknecht & Wibral (2021):
  `i_sx(α; t) = log P(t ∩ (a_1 ∪ … ∪ a_k)) / (P(t) P(a_1 ∪ … ∪ a_k))`. Union probabilities
  are computed exactly from counts by inclusion–exclusion over subsets of `α`.
- `IPM()`, the `i_±` of Finn & Lizier (2018):
  `i_±(α; t) = min_j h(a_j) - min_j h(a_j | t)`.

The redundancy of a node is the sample mean of its local redundancies, so `association`
returns an ordinary `PIDResult`, and `local_association` returns per-sample atoms (a
`LocalPIDResult`) whose means are the atoms of `association`:

```julia
using Associations, LocalInformation, Statistics

x, y = rand(0:1, 10_000), rand(0:1, 10_000); t = x .⊻ y
est = JointProbabilities(PartialInformationDecomposition(ISx()), CodifyVariables(UniqueElements()))
res = association(est, x, y, t)           # PIDResult: synergy(res) ≈ log2(4/3)
loc = local_association(est, x, y, t)     # LocalPIDResult, N × 4 atoms
mean(synergy(loc)) ≈ synergy(res)         # true; also redundancy, unique_information, atom, atoms
sum(loc.atoms; dims = 2)                  # local mutual information i(x, y; t) of each sample
```

Per-sample atoms are the Möbius inversion of the local redundancies, done for all samples
at once as a triangular solve `A = R / Z` with the zeta matrix `Z` of the lattice (on the
`backend`, e.g. CUBLAS with `backend = CUDABackend()`). Local redundancies depend on a
sample only through its joint state, so `R` has one row per distinct state, computed on the
CPU. For 5 sources (7579 atoms) select the atoms you need, e.g.
`atoms = [[[1, 2, 3, 4, 5]], [[1], [2], [3], [4], [5]]]`: then only the needed columns of
`Z⁻¹` (the lattice's Möbius function, computed exactly) and the redundancies of the few
nodes where it is nonzero are computed. `atoms = :all` throws an `ArgumentError` before
allocating if it would need more than `memory_limit` bytes (default 2 GiB). A `PIDAtom`
estimator gives the per-sample values of one atom as a `Vector`.

`IMin` and `IMMI` have no per-sample form (their redundancies are minima of averages), so
`local_association` throws an `ArgumentError` for them. `test/fixtures/syntropy_pid_fixtures.json`
(from `test/fixtures/generate_syntropy_pid_fixtures.py`) holds syntropy's average and
pointwise `isx`/`ipm` atoms, which the tests match to 1e-10.

## Integrated information decomposition (ΦID)

`IntegratedInformationDecomposition(redundancy = IMMI(); τ = 1, base = 2)` decomposes the
time-delayed mutual information `I(X_t; X_{t+τ})` of 2 or 3 variables into the atoms of
the ΦID product lattice of Mediano, Rosas et al.: 16 atoms (`:rtr` … `:sts`) for two
variables and 324 for three. `association` returns the average atoms (a `PhiIDResult`) and
`local_association` the atoms of every lagged sample `t = 1, …, N - τ` (a
`LocalPhiIDResult`), whose means are the average atoms:

```julia
using Associations, LocalInformation, Statistics

x = randn(10_001); y = randn(10_001)
y[2:end] .+= 0.8 .* x[1:end-1]                   # x drives y

est = GaussianDecomposition(IntegratedInformationDecomposition())
res = association(est, x, y)                      # PhiIDResult: 16 atoms, in bits
information_dynamics(res).transfer                # xty + ytx
loc = local_association(est, x, y)                # LocalPhiIDResult: one row per sample
mean(atom(loc, :xty)) ≈ atom(res, :xty)           # true
```

- **Double redundancies.** `IMMI()` is the default. Its average atoms work with
  `JointProbabilities`, `GaussianDecomposition` and `EntropyDecomposition` with any
  Shannon entropy estimator. Its per-sample atoms work with plug-in, `Kraskov`,
  `KozachenkoLeonenko` and Gaussian entropies. `ISx()` (Varley's shared-exclusion double
  redundancy) and `IPM()` work only with `JointProbabilities` and a `CodifyVariables`
  discretization, for average and per-sample atoms.
- **Per-sample MMI atoms** follow the convention of `phyid`: each local double redundancy
  is the local mutual information of the pair of variable sets with the smallest mean.
  They run on the CPU or a GPU.
- **Encoding and lagging.** Discrete series are encoded once over their full length and
  then lagged.
- **Information dynamics.** `information_dynamics(res)` gives the six modes of Mediano et
  al.: storage, copy, transfer, erasure, downward and upward causation.

The [ΦID page](https://denizhanpak.github.io/LocalInformation.jl/dev/phiid/) of the
documentation lists the estimators, the conventions and recipes such as transfer entropy,
active information storage and Φ_WMS as sums of atoms. The tests compare the atoms with
syntropy (average MMI, ISx and IPM atoms; pointwise ISx and IPM) and with phyid (average
and local MMI atoms): `test/fixtures/syntropy_phiid_fixtures.json` and
`test/fixtures/phyid_fixtures.json`, described in `test/fixtures/phiid_reference_notes.md`.

## Relation to Associations.jl

LocalInformation.jl owns its functions (`local_association`, `local_entropy`) and dispatches
on Associations' estimator types. It builds on Associations' documented extension API
(`Associations.subset_entropies`, `Associations.entropy_combination`; see the "Extending
Associations" docs page): local higher-order measures are
`Associations.entropy_combination(est.definition, H, 1:n)` with a callable `H` that returns
a vector of local subset entropies.

## Internal interfaces

- `LocalInformation.local_subset_entropies(est, x...; backend, precision) → H`, with
  `H(S)` a `Vector` of local joint entropies of `x[S]` (cached). One method per estimator
  family; every local measure is built from it.
- `LocalInformation.knn_distances(points, k, w; metric, backend, precision)`: distances to
  the `k`-th nearest neighbour outside a Theiler window, dispatched on the backend.
- `LocalInformation.resolve_precision`, `to_device`, `to_host`: backend and precision
  handling.
- `LocalInformation.phiid_redundancies(r, est, lattice, past, future)`: the extension hook
  for ΦID double redundancies (an `m × m` matrix, past nodes as rows).

## Installation

LocalInformation.jl is not registered yet. It depends on extension hooks that are not in a
released version of Associations.jl; until they are, `Project.toml` takes Associations from
the `local-information-hooks` branch of
[denizhanpak/Associations.jl](https://github.com/denizhanpak/Associations.jl) through a
`[sources]` entry, which needs Julia ≥ 1.11.

```julia
using Pkg
Pkg.add(url = "https://github.com/denizhanpak/LocalInformation.jl")
```

For GPU support, also install CUDA.jl (`Pkg.add("CUDA")`) and load it with `using CUDA`.

## License

MIT; see [LICENSE](LICENSE).

## Tests

```julia
julia --project=test test/runtests.jl
```

The GPU tests run only if `CUDA.functional()`. `test/fixtures/syntropy_fixtures.json` holds
reference pointwise values from the Python package syntropy, produced by
`test/fixtures/generate_syntropy_fixtures.py`.
