# Public entry points, the internal subset-entropy interface, and the errors for
# unsupported estimators. Methods for specific estimators live in the other files.

"""
    local_association(est, x...; backend = CPU(), precision = nothing) → Vector

Return the local (per-sample, pointwise) values of the measure `est.definition`, estimated
with the Associations.jl estimator `est` from the inputs `x...`, in sample order and in the
unit given by `est.definition.base` (bits for the default base 2).

Each input is an `AbstractVector` (one variable) or a `StateSpaceSet` (a multivariate
variable). For every supported estimator,

```julia
mean(local_association(est, x...)) ≈ association(est, x...)
```

up to floating-point rounding (and up to the `precision` used on a GPU).

## Supported estimators

| Estimator | Definitions | GPU |
|:--|:--|:--|
| `JointProbabilities(def, CodifyVariables(o))` | `MIShannon`, `CMIShannon`, higher-order | no (computed on the CPU) |
| `EntropyDecomposition(def, PlugIn(Shannon()), d)` | `MIShannon`, `CMIShannon`, higher-order | no (computed on the CPU) |
| `EntropyDecomposition(def, Kraskov(...))`, `EntropyDecomposition(def, KozachenkoLeonenko(...))` | `MIShannon`, `CMIShannon`, higher-order | yes |
| `KSG1`, `KSG2` (`KraskovStögbauerGrassberger1/2`) | `MIShannon` | yes |
| `FPVP` | `CMIShannon` | yes |
| `GaussianDecomposition(def; copula)` | higher-order | yes |
| `GaussianMI`, `GaussianCMI` | `MIShannon`, `CMIShannon` | yes |

"Higher-order" means `TotalCorrelation`, `DualTotalCorrelation`, `OInformation` and
`SInformation`. Other estimators throw an `ArgumentError`. Bias-corrected entropy
estimators (`MillerMadow`, `ChaoShen`, `Jackknife`, `Schuermann`, ...) are not supported,
because their estimates are not means of per-sample terms.

## Keywords

- `backend`: a KernelAbstractions.jl backend, `CPU()` (default) or a GPU backend such as
    `CUDABackend()` (requires `using CUDA`). Discrete (counting) estimators always compute
    on the CPU, whatever the backend: hash-based counting gains nothing on a GPU.
- `precision`: the floating-point type to compute in and of the result. `nothing` (default)
    means `Float64` on the CPU and `Float32` on a GPU; `precision = Float64` is allowed on
    a GPU.

## Discretized inputs

With an outcome space that embeds the data (e.g. `OrdinalPatterns`), there is one local
value per encoded symbol, i.e. per embedding vector, not per raw sample.
"""
function local_association end

"""
    local_entropy(est::DifferentialInfoEstimator, x; backend = CPU(), precision = nothing) → Vector
    local_entropy(est::PlugIn, o::OutcomeSpace, x; backend = CPU(), precision = nothing) → Vector
    local_entropy(o::OutcomeSpace, x; kwargs...) → Vector

Local (per-sample) Shannon entropies ``h_i = -\\log \\hat p(x_i)`` of the single variable `x`
(an `AbstractVector` or a `StateSpaceSet`), in the unit `est.definition.base`, with

```julia
mean(local_entropy(est, x)) ≈ information(est, x)          # differential
mean(local_entropy(est, o, x)) ≈ information(est, o, x)    # discrete
```

Supported estimators (from ComplexityMeasures.jl):

- `Kraskov(Shannon(); k, w)`: ``h_i = -\\psi(k) + \\psi(N) + \\log V_D + D \\log \\rho_i``,
    with ``\\rho_i`` the Euclidean distance to the `k`-th nearest neighbour outside the
    Theiler window `w`, and ``V_D`` the volume of the Euclidean unit ball.
- `KozachenkoLeonenko(Shannon(); w)`:
    ``h_i = D \\log \\rho_i + \\log V_D + \\gamma + \\log(N - 1)``, with ``\\rho_i`` the
    distance to the nearest neighbour.
- `PlugIn(Shannon())` with a counting-based outcome space `o`:
    ``h_i = -\\log(n(\\omega_i) / N)``, where ``\\omega_i`` is the outcome of the `i`-th
    encoded symbol, ``n(\\omega)`` its count and ``N`` the number of symbols.
    `local_entropy(o, x)` uses `PlugIn(Shannon())` (base 2).
    This needs `probabilities(o, x)` to be the relative counts of the outcomes that
    `codify(o, x)` returns. It throws an `ArgumentError` otherwise: for outcome spaces that
    are not counting-based (`WeightedOrdinalPatterns`, `AmplitudeAwareOrdinalPatterns`,
    `PowerSpectrum`, …) and for `Dispersion`, whose `codify` gives per-sample symbols while
    its probabilities count dispersion patterns.

`backend` and `precision` are as for [`local_association`](@ref). The k-NN estimators run
on the CPU (k-d tree) or on a GPU (exact brute-force kernel); the discrete estimator always
computes on the CPU.
"""
function local_entropy end

"""
    LocalInformation.local_subset_entropies(est, x...; backend = CPU(), precision = nothing) → H

The internal core of the package. Return a callable `H` such that `H(S)` is a host
`Vector{T}` of length `N` (the number of samples or encoded symbols) holding the local
joint Shannon entropies ``h(x_S)_i = -\\log \\hat p(x_S)_i`` of the inputs `x[S]`, in the unit
`est.definition.base`, with `T = LocalInformation.resolve_precision(backend, precision)`.
`mean(H(S))` equals the estimate of the joint entropy of `x[S]` that Associations.jl uses
for `est` (e.g. `Associations.subset_entropies(est, x...)(S)`).

`H` is directly usable as the entropy callable of `Associations.entropy_combination`
(local higher-order measures) and to build local mutual informations, e.g.
`H((1,)) + H((2,)) - H((1, 2))`.

## The callable

- `S` is a `Tuple`, `Vector` or range of distinct indices in `1:length(x)`; order does
    not matter. `H(S)` for an empty `S` returns `zeros(T, N)`.
- Results are cached per subset (sorted index vector), so repeated calls are cheap.
    The returned vectors are the cached objects: do not mutate them. `H` is not safe to
    call concurrently from several tasks.
- The concrete type is [`LocalInformation.LocalSubsetEntropies`](@ref).

## Methods

- `JointProbabilities` with `CodifyVariables` and the default `RelativeAmount`
    probabilities: plug-in local entropies from counts of the joint outcomes of `x[S]`,
    with the same encoding as `Associations.counts(est.discretization, x...)`. CPU.
- `EntropyDecomposition` with `PlugIn(Shannon())`: the same, with the encoding of
    `Associations.codified_marginals(est.discretization, x...)`. CPU.
- `EntropyDecomposition` with `Kraskov` or `KozachenkoLeonenko`: k-NN local entropies of
    `StateSpaceSet(x[S]...)`, using [`LocalInformation.knn_distances`](@ref).
- `GaussianDecomposition`, `GaussianMI`, `GaussianCMI`: Gaussian local entropies (see
    [`LocalInformation.local_gaussian_entropies`](@ref)). CPU or GPU.

## Extending

A new estimator family adds one method

```julia
function LocalInformation.local_subset_entropies(est::MyEstimator, x...; backend = CPU(), precision = nothing)
    T = LocalInformation.resolve_precision(backend, precision)
    N = ...                                   # number of local values
    h(S::Vector{Int}) = ...                   # Vector{T} of local entropies, unit est.definition.base
    return LocalInformation.LocalSubsetEntropies{T}(h, N)
end
```

after which `local_association` works for its higher-order definitions, `MIShannon` and
`CMIShannon` (see `higher_order.jl` and `bivariate.jl`).
"""
function local_subset_entropies end

# ----------------------------------------------------------------------------------------
# Errors for unsupported estimators. The style follows Associations' `pid_redundancies`
# fallback for `IMin`: what is supported, an example, and what was received.
# ----------------------------------------------------------------------------------------
const SUPPORTED_ESTIMATORS = """
Supported estimators: \
`JointProbabilities(def, CodifyVariables(o))` (plug-in probabilities), \
`EntropyDecomposition(def, PlugIn(Shannon()), discretization)`, \
`EntropyDecomposition(def, Kraskov(; k))`, `EntropyDecomposition(def, KozachenkoLeonenko())`, \
`KSG1(MIShannon())`, `KSG2(MIShannon())`, `FPVP(CMIShannon())`, \
`GaussianDecomposition(def)` (higher-order `def` only), `GaussianMI()` and `GaussianCMI()`, \
where `def` is `MIShannon()`, `CMIShannon()` or a higher-order measure \
(`TotalCorrelation`, `DualTotalCorrelation`, `OInformation`, `SInformation`). \
For example, `local_association(JointProbabilities(MIShannon(), \
CodifyVariables(ValueBinning(RectangularBinning(4)))), x, y)`."""

describe(est) = string(nameof(typeof(est)))
function describe(est::Union{JointProbabilities,GaussianDecomposition,GaussianMI,GaussianCMI})
    return "$(nameof(typeof(est))) estimator of $(nameof(typeof(est.definition)))"
end
function describe(est::EntropyDecomposition)
    return "EntropyDecomposition estimator of $(nameof(typeof(est.definition))) with a " *
        "$(nameof(typeof(est.est))) entropy estimator"
end

function unsupported_estimator(est)
    throw(ArgumentError(
        "Local (per-sample) values are not implemented for a $(describe(est)). " *
        SUPPORTED_ESTIMATORS
    ))
end

function unsupported_entropy_estimator(est)
    throw(ArgumentError(
        "Local entropies are only defined for estimators whose estimate is the mean of " *
        "per-sample terms: `PlugIn(Shannon())` (discrete, with a counting-based outcome " *
        "space), `Kraskov(Shannon(); k, w)` or `KozachenkoLeonenko(Shannon(); w)` " *
        "(differential). Bias-corrected estimators (`MillerMadow`, `ChaoShen`, " *
        "`HorvitzThompson`, `Jackknife`, `Schuermann`, `GeneralizedSchuermann`) and other " *
        "measures than `Shannon` are not supported. Got a $(describe(est))."
    ))
end

local_association(est, x...; backend = CPU(), precision = nothing) = unsupported_estimator(est)
local_subset_entropies(est, x...; backend = CPU(), precision = nothing) = unsupported_estimator(est)
local_entropy(est, x...; backend = CPU(), precision = nothing) = unsupported_entropy_estimator(est)

# All inputs must have the same number of samples.
function check_input_lengths(x...)
    isempty(x) && throw(ArgumentError("At least one input is needed."))
    if !allequal(length.(x))
        throw(ArgumentError(
            "All inputs must have the same number of samples. Got lengths $(length.(x))."
        ))
    end
    return length(first(x))
end
