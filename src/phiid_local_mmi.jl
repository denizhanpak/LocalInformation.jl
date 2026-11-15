# Per-sample ΦID atoms with the MMI double redundancy (phyid convention), for every
# estimator family with local subset entropies, on the CPU or a GPU.
#
# For each pair (A, B) of nonempty variable sets, A of the past and B of the future, the
# local mutual information i_t(A; B) = h_t(A) + h_t(B) - h_t(A ∪ B) comes from the local
# subset entropies of the estimator. They form the N × np matrix `Lmi`, np = (2ⁿ - 1)²,
# column `a + (b - 1) nm` for the bitmasks a of A and b of B (nm = 2ⁿ - 1). The local double
# redundancy of the lattice pair (α → β) is the column of the pair (A, B) ∈ α × β with the
# smallest mean (ties: the first in the order of `mmi_pair_selection`). With the 0/1
# selection S (np × m², one 1 per column), the local redundancies are `Lmi * S` and the local
# atoms `Lmi * S * kron(μ, μ)`, computed as one product `Lmi * W`, W = S * kron (np × m²),
# without materialising the N × m² redundancies.

const LocalMMIPhiID = IntegratedInformationDecomposition{<:IMMI}

"""
    LocalInformation.PHIID_MMI_TIE_RTOL = 1e-12

Relative tolerance for ties in the pair selection of per-sample ΦID atoms with `IMMI`: a
pair counts as tied with the smallest mean local mutual information `Īₘᵢₙ` if its mean is
at most `Īₘᵢₙ + PHIID_MMI_TIE_RTOL * max(1, maximum(abs, Ī))`, the maximum over all pairs of
variable sets with finite means. Exactly tied means can differ by summation-order rounding
(~1e-14); this makes the documented tie-break independent of it.
"""
const PHIID_MMI_TIE_RTOL = 1e-12

"""
    local_association(est, x₁, …, xₙ; backend = CPU(), precision = nothing, memory_limit = 2^31) → LocalPhiIDResult

Per-sample (local) atoms of the integrated information decomposition with the minimum
mutual information (MMI) double redundancy, for an estimator `est` of
`IntegratedInformationDecomposition(IMMI(); τ, base)`, following the convention of `phyid`
and of the MATLAB ΦID toolbox of Mediano et al. Returns a [`LocalPhiIDResult`](@ref) with one
row per lagged sample, `t = 1, …, N - τ` (row `t`: past at time `t`, future at `t + τ`), and

```julia
vec(mean(local_association(est, x...).atoms; dims = 1)) ≈ vec(association(est, x...).atoms)
```

## Convention

For each pair of nonempty variable sets, ``A`` of the past and ``B`` of the future, the
local mutual information of sample ``t`` is

```math
i_t(A; B) = h_t(\\mathbf{X}^A_t) + h_t(\\mathbf{X}^B_{t+\\tau}) - h_t(\\mathbf{X}^A_t, \\mathbf{X}^B_{t+\\tau}),
```

from the local joint entropies of the estimator (see
[`LocalInformation.local_subset_entropies`](@ref)); its mean over samples ``\\bar I(A; B)``
is the mutual information used by `association(est, x...)`. For every pair of lattice nodes
``(\\alpha \\to \\beta)``, the local double redundancy of sample ``t`` is ``i_t(A^*; B^*)``,
where ``(A^*, B^*) \\in \\alpha \\times \\beta`` is the pair with the smallest *mean*
``\\bar I(A; B)``, i.e. the minimiser of the average double redundancy
``I_\\cap^{\\alpha \\to \\beta} = \\min_{A \\in \\alpha, B \\in \\beta} \\bar I(A; B)``. The
local atoms of a sample are the Möbius inversion of its local double redundancies.
Consequently:

- the mean of every local atom is the average atom of `association(est, x...)` (the mean of
    the selected local MI is the minimum of the averages);
- the atoms of a sample sum to its local ``I(\\mathbf{X}_t; \\mathbf{X}_{t+\\tau})``;
- identities between atoms and single-set double redundancies hold per sample, e.g. the
    local transfer entropy ``i_t(x_t; y_{t+\\tau} \\mid y_t)`` = `xtr + xty + str + sty`
    and the six [`information_dynamics`](@ref) modes sum to the local
    ``I(\\mathbf{X}_t; \\mathbf{X}_{t+\\tau})``.

This is the per-sample MMI ΦID of `phyid` (`calc_PhiID(…, redundancy = "MMI")`, Luppi et al.
2022) and of Mediano et al. (2021, 2025), generalised to 3 variables. It is a convention:
which pair is selected depends on the whole sample (through the means), so a sample's local
double redundancy is not a function of its joint state alone, unlike the pointwise double
redundancies of [`ISx`](@ref) and [`IPM`](@ref). For the PID, `local_association` refuses
`IMMI` for that reason; for ΦID this convention is the established one, used to obtain
time-resolved atoms whose means are the average atoms.

## Ties

The candidate pairs ``(A, B)`` of a lattice pair are ordered by the bitmask
``\\sum_{k \\in A} 2^{k-1}`` of ``A``, then by that of ``B``: variable sets in the order
``\\{1\\}, \\{2\\}, \\{1,2\\}, \\{3\\}, \\{1,3\\}, \\{2,3\\}, \\{1,2,3\\}``. The first pair whose
mean is within [`LocalInformation.PHIID_MMI_TIE_RTOL`](@ref)` * max(1, maximum(abs, Ī))` of
the smallest mean is selected. For two variables the double redundancy
``\\{1\\}\\{2\\} \\to \\{1\\}\\{2\\}`` therefore considers ``I(x_t; x_{t+\\tau})``,
``I(x_t; y_{t+\\tau})``, ``I(y_t; x_{t+\\tau})``, ``I(y_t; y_{t+\\tau})`` in this order and
keeps the first of a tie, as `phyid` does (`np.argmin`); for the single-target
redundancies (e.g. ``\\{1\\}\\{2\\} \\to \\{1\\}``) `phyid` keeps the *second* pair of an exact
tie, this method the first. Ties only change the local values, never their means.

## Compatible estimators

| Estimator | Backend |
|:--|:--|
| `JointProbabilities(def, CodifyVariables(o))` (plug-in probabilities) | counts on the CPU |
| `EntropyDecomposition(def, PlugIn(Shannon()), CodifyVariables(o))` | counts on the CPU |
| `EntropyDecomposition(def, Kraskov(; k, w))`, `EntropyDecomposition(def, KozachenkoLeonenko(; w))` | CPU (k-d tree) or GPU (brute force) |
| `GaussianDecomposition(def; copula)` | CPU or GPU |

with `def = IntegratedInformationDecomposition(IMMI(); τ, base)`. Discrete data are encoded
once over the full series and then lagged, as for `association`. Entropy estimators
without per-sample terms (bias-corrected ones such as `MillerMadow`, or other differential
estimators) throw an `ArgumentError`; use `association(est, x...)` for their average atoms.

For the Gaussian estimator the local entropies are those of the maximum-likelihood
covariance (see `src/gaussian.jl`), whereas `phyid` uses the unbiased one (`ddof = 1`). The
means agree; the local values relate by
`local_ML = average + N′/(N′ - 1) · (local_phyid - average)`, with `N′ = N - τ`.

## Keywords

- `backend`, `precision`: as for every `local_association` (`Float32` on a GPU by default).
    The local entropies are computed on `backend` where the estimator supports it, and the
    inversion (one matrix product) always runs on `backend`.
- `memory_limit`: an `ArgumentError` is thrown, before any large allocation, if the
    estimated memory (bytes, host and device) exceeds it. The result alone takes
    `(N - τ) · m² · sizeof(T)` bytes (`m² = 16` atoms for 2 variables, 324 for 3), e.g.
    2.6 GB for 10⁶ samples of 3 variables in `Float64`.

## Example

```julia
using LocalInformation, Associations
x = randn(10_001); y = randn(10_001)
y[2:end] .+= 0.8 .* x[1:end-1]                   # x drives y
est = GaussianDecomposition(IntegratedInformationDecomposition())
loc = local_association(est, x, y)               # LocalPhiIDResult, 10_000 × 16
atom(loc, :xty)                                  # local unique transfer x → y, per sample
information_dynamics(loc).transfer               # per-sample transfer mode
```

## References

- Luppi, A. I., Mediano, P. A. M., Rosas, F. E., et al. (2022). A synergistic core for human
    brain evolution and cognition. Nature Neuroscience, 25, 771–782 (`phyid`:
    <https://github.com/Imperial-MIND-lab/integrated-info-decomp>).
- Mediano, P. A. M., Rosas, F. E., Luppi, A. I., et al. (2021). Towards an extended
    taxonomy of information dynamics via Integrated Information Decomposition.
    arXiv:2109.13186. Published as: Toward a unified taxonomy of information dynamics via
    Integrated Information Decomposition. PNAS, 122(39), e2423297122 (2025).
    doi:10.1073/pnas.2423297122.
"""
function local_association(est::JointProbabilities{<:LocalMMIPhiID}, x...; kwargs...)
    return local_phiid_mmi_association(est, x...; kwargs...)
end
function local_association(est::EntropyDecomposition{<:LocalMMIPhiID}, x...; kwargs...)
    return local_phiid_mmi_association(est, x...; kwargs...)
end
function local_association(est::GaussianDecomposition{<:LocalMMIPhiID}, x...; kwargs...)
    return local_phiid_mmi_association(est, x...; kwargs...)
end

function local_phiid_mmi_association(est, x...; backend = CPU(), precision = nothing, memory_limit::Real = 2^31)
    T = resolve_precision(backend, precision)
    est′, past, future = phiid_prepare(est, x...)
    return local_phiid_mmi(est′, past, future; backend, precision = T, memory_limit)
end

"""
    LocalInformation.local_phiid_mmi(est, past::Tuple, future::Tuple;
        backend = CPU(), precision = nothing, memory_limit = 2^31) → LocalPhiIDResult

Per-sample MMI ΦID atoms (see `local_association` for an `IntegratedInformationDecomposition`
with `IMMI`) of the system whose past variables are `past` and future variables `future`,
with no lagging or shared encoding: `est` is applied to the `2n` inputs
`(past..., future...)` as given, like [`LocalInformation.phiid_from_past_future`](@ref) does
for the average atoms. Rows are the samples of `past` and `future`.
"""
function local_phiid_mmi(est, past::Tuple, future::Tuple;
        backend = CPU(), precision = nothing, memory_limit::Real = 2^31)
    T = resolve_precision(backend, precision)
    n = length(past)
    length(future) == n || throw(ArgumentError(
        "The past and the future must have the same number of variables; got " *
        "$n and $(length(future))."))
    check_phiid_count(est.definition, n)
    est.definition.redundancy isa IMMI || throw(ArgumentError(
        "`local_phiid_mmi` needs an `IntegratedInformationDecomposition` with `IMMI`; got " *
        "$(nameof(typeof(est.definition.redundancy)))."))
    check_local_mmi_estimator(est)
    N = check_phiid_inputs((past..., future...))
    L = phiid_lattice(n)
    m2 = nnodes(L)^2
    check_local_mmi_memory(backend, T, N, n, m2, memory_limit)
    H = local_subset_entropies(est, past..., future...; backend, precision = T)
    Lmi = local_pair_mutual_informations(H, n, T, N)
    sel = mmi_pair_selection(L, pair_means(Lmi))
    W = mmi_selection_weights(L, sel, size(Lmi, 2))
    A = to_host(to_device(backend, Lmi, T) * to_device(backend, W, T))
    return LocalPhiIDResult(est.definition, L, A)
end

# Estimators whose entropies are means of per-sample terms; the others (bias-corrected
# discrete estimators, other differential estimators, non-plug-in probabilities) have no
# local subset entropies. `JointProbabilities` is validated by `phiid_series` and
# `local_subset_entropies`.
local_mmi_supported(est) = true
function local_mmi_supported(est::EntropyDecomposition)
    return (est.est isa PlugInShannon && est.pest isa RelativeAmount) || est.est isa KNNShannon
end

function check_local_mmi_estimator(est)
    local_mmi_supported(est) && return nothing
    e = est.est
    detail = if e isa PlugInShannon
        " Its probabilities estimator $(nameof(typeof(est.pest))) is not the plug-in " *
        "`RelativeAmount`."
    elseif e isa DiscreteInfoEstimator
        " $(nameof(typeof(e))) is a bias-corrected estimate, not a mean of per-sample terms."
    elseif e isa DifferentialInfoEstimator
        " Of the differential entropy estimators, only `Kraskov` and `KozachenkoLeonenko` " *
        "are means of per-sample terms."
    else
        ""
    end
    throw(ArgumentError(
        "Per-sample (local) ΦID atoms with IMMI need an estimator whose entropies are " *
        "means of local (per-sample) entropies.$detail Use " *
        "`JointProbabilities(def, CodifyVariables(o))`, `EntropyDecomposition(def, " *
        "PlugIn(Shannon()), CodifyVariables(o))`, `EntropyDecomposition(def, Kraskov(; k))`, " *
        "`EntropyDecomposition(def, KozachenkoLeonenko())` or `GaussianDecomposition(def)`, " *
        "or `association(est, x...)` for the average atoms (a `PhiIDResult`). Got a " *
        "$(describe(est))."))
end

# Estimated peak memory (bytes) of `local_phiid_mmi` for N samples of n variables.
function check_local_mmi_memory(backend, ::Type{T}, N, n, m2, limit) where {T}
    s = sizeof(T)
    nm = (1 << n) - 1
    np = nm^2
    oncpu = backend isa KA.CPU
    need = s * N * (np + 2nm + 1)          # local MIs, cached marginal entropies, one joint (host)
    need += s * np * m2 * (oncpu ? 1 : 2)  # selection weights (host and device)
    need += s * N * m2 * (oncpu ? 1 : 2)   # atoms (device and host)
    need += oncpu ? 0 : s * N * np         # device copy of the local MIs
    if need > limit
        throw(ArgumentError(
            "Local ΦID atoms ($m2 atoms of $n variables) for $N samples would need about " *
            "$(round(need / 2^30; digits = 2)) GiB, more than `memory_limit` = " *
            "$(round(limit / 2^30; digits = 2)) GiB. Use fewer samples, `precision = " *
            "Float32`, or raise `memory_limit`."))
    end
    return need
end

# Variables (1..n) of the set with bitmask `mask`.
mask_variables(mask::Integer, n::Int) = [k for k in 1:n if isodd(mask >> (k - 1))]

"""
    LocalInformation.local_pair_mutual_informations(H, n, T, N) → Matrix{T}

The `N × (2ⁿ - 1)²` local mutual informations `i_t(A; B) = h_t(A) + h_t(B) - h_t(A ∪ B)`
between every nonempty set `A` of the past variables (inputs `1:n` of `H`) and `B` of the
future variables (inputs `n+1:2n`), from the local subset entropies `H`. Column
`a + (b - 1)(2ⁿ - 1)` holds the pair with bitmasks `a` of `A` and `b` of `B`. The `2(2ⁿ - 1)`
single-side entropies are cached in `H`; the joint ones are computed once each, uncached.
"""
function local_pair_mutual_informations(H, n::Int, ::Type{T}, N::Int) where {T}
    nm = (1 << n) - 1
    hp = [H(mask_variables(a, n)) for a in 1:nm]
    hf = [H(n .+ mask_variables(b, n)) for b in 1:nm]
    Lmi = Matrix{T}(undef, N, nm^2)
    for b in 1:nm, a in 1:nm
        hj = uncached_subset_entropies(H, vcat(mask_variables(a, n), n .+ mask_variables(b, n)))
        view(Lmi, :, a + (b - 1) * nm) .= hp[a] .+ hf[b] .- hj
    end
    return Lmi
end

# Joint local entropies that are used once, without filling the cache of `H`.
uncached_subset_entropies(H::LocalSubsetEntropies, key::Vector{Int}) = H.h(key)
uncached_subset_entropies(H, key) = H(key)

# Mean of every column, accumulated in Float64 whatever the precision of the local values.
pair_means(Lmi::AbstractMatrix) = [sum(Float64, view(Lmi, :, p)) / size(Lmi, 1) for p in axes(Lmi, 2)]

"""
    LocalInformation.mmi_pair_selection(lattice::PhiIDLattice, μ::AbstractVector) → Vector{Int}

For every ΦID atom column `c = i + (j - 1) m` (past node `i`, future node `j`), the column
of the local-MI matrix (see [`LocalInformation.local_pair_mutual_informations`](@ref)) of the
pair `(A, B)` ∈ αᵢ × αⱼ with the smallest mean `μ`. Candidates are ordered by the bitmask of
`A`, then of `B`; the first whose mean is within
[`LocalInformation.PHIID_MMI_TIE_RTOL`](@ref)` * max(1, maximum(abs, μ))` (finite `μ`) of the
smallest is selected. A `NaN` among the candidates selects the first `NaN` pair.
"""
function mmi_pair_selection(L::PhiIDLattice, μ::AbstractVector{<:Real})
    n = nvariables(L)
    nm = (1 << n) - 1
    length(μ) == nm^2 || throw(ArgumentError(
        "Expected $(nm^2) mean mutual informations for $n variables; got $(length(μ))."))
    scale = maximum(v -> isfinite(v) ? abs(v) : 0.0, μ)
    tol = PHIID_MMI_TIE_RTOL * max(1.0, scale)
    nodes = L.lattice.nodes
    m = length(nodes)
    col(a, b) = Int(a) + (Int(b) - 1) * nm
    sel = Vector{Int}(undef, m^2)
    for j in 1:m, i in 1:m
        α, β = nodes[i], nodes[j]   # sorted bitmasks, so the loops below are in tie order
        μmin = minimum(μ[col(a, b)] for a in α, b in β)
        best = 0
        for a in α, b in β
            v = μ[col(a, b)]
            if isnan(μmin) ? isnan(v) : v ≤ μmin + tol
                best = col(a, b)
                break
            end
        end
        sel[i + (j - 1) * m] = best
    end
    return sel
end

"""
    LocalInformation.mmi_selection_weights(lattice::PhiIDLattice, sel::AbstractVector{Int}, np::Int) → Matrix{Int}

`W = S * lattice.kron` (`np × m²`), with `S[sel[c], c] = 1` the selection of a local-MI
column for every ΦID atom column `c`: the local atoms of local MIs `Lmi` (`N × np`) are
`Lmi * W`. Row `p` of `W` is the sum of the rows `c` of `kron` with `sel[c] == p`.
"""
function mmi_selection_weights(L::PhiIDLattice, sel::AbstractVector{<:Integer}, np::Int)
    K = L.kron
    length(sel) == size(K, 1) || throw(ArgumentError(
        "Expected one selected pair per ΦID atom ($(size(K, 1))); got $(length(sel))."))
    W = zeros(Int, np, size(K, 2))
    for (c, p) in enumerate(sel)
        view(W, p, :) .+= view(K, c, :)
    end
    return W
end
