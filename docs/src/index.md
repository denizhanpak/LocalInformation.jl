# LocalInformation.jl

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
  KSG and Gaussian estimators: [Supported estimators](@ref).
- Local Shannon entropies with [`local_entropy`](@ref).
- Per-sample PID atoms with the pointwise redundancy functions `ISx` and `IPM`:
  [Partial information decomposition](@ref).
- Integrated information decomposition (ΦID) of 2 or 3 variables, average and per-sample
  atoms. It supports the MMI double redundancy for every estimator family, and `ISx` and
  `IPM` for discrete data. Two-variable results also give the six information-dynamics
  modes: [Integrated information decomposition (ΦID)](@ref).
- GPU backends through KernelAbstractions.jl: [GPU](@ref).

For example, the ΦID of two coupled Gaussian series:

```julia
x = randn(10_001); y = randn(10_001)
y[2:end] .+= 0.8 .* x[1:end-1]                   # x drives y

est = GaussianDecomposition(IntegratedInformationDecomposition())
res = association(est, x, y)                      # PhiIDResult: 16 atoms, in bits
information_dynamics(res).transfer                # xty + ytx
loc = local_association(est, x, y)                # LocalPhiIDResult: one row per sample
mean(atom(loc, :xty)) ≈ atom(res, :xty)           # true
```

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

## Relation to Associations.jl

LocalInformation.jl owns its functions (`local_association`, `local_entropy`) and dispatches
on Associations' estimator types. It builds on Associations' documented extension API
(`Associations.subset_entropies`, `Associations.entropy_combination`; see the "Extending
Associations" docs page): local higher-order measures are
`Associations.entropy_combination(est.definition, H, 1:n)` with a callable `H` that returns
a vector of local subset entropies.

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

## Tests

```julia
julia --project=test test/runtests.jl
```

The GPU tests run only if `CUDA.functional()`. `test/fixtures/syntropy_fixtures.json` holds
reference pointwise values from the Python package syntropy, produced by
`test/fixtures/generate_syntropy_fixtures.py`.
