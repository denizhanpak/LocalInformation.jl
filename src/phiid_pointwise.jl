# ΦID with the pointwise double redundancies ISx and IPM (discrete data, plug-in
# probabilities): average atoms (`association`, through the `phiid_redundancies` hook of
# phiid.jl) and per-sample atoms (`local_association`).
#
# A local double redundancy depends on a sample only through its joint state (all past and
# future variables), so it is computed once per distinct joint state from the counts of
# every subset of the 2n inputs (`SubsetCounts`; past variable i is bit i - 1, future
# variable i bit n + i - 1). The average redundancies are the weighted means over states;
# per-sample atoms are the per-state atoms copied to the samples.

using LinearAlgebra: mul!

"""
    LocalInformation.PhiIDPointwiseData

Data for the ΦID of `n` variables with [`ISx`](@ref) and [`IPM`](@ref), built once per
`n` by `LocalInformation.phiid_pointwise_data(n)` from the
[`LocalInformation.PhiIDLattice`](@ref) (`m` nodes, in node order). Input masks are over
the `2n` inputs of a [`LocalInformation.SubsetCounts`](@ref): past variable `i` is bit
`i - 1`, future variable `i` bit `n + i - 1`.

## Fields

- `lattice::PhiIDLattice`: the shared lattice of `phiid_lattice(n)`.
- `sx::Matrix{Float64}`: `(2²ⁿ - 1) × (2m + m²)` inclusion–exclusion coefficients
    (integers). For subset counts `C` (`K × (2²ⁿ - 1)`, one row per joint state),
    `U = C * sx` holds the counts of the samples in the unions of the events of a state:
    column `i` for past node ``\\alpha_i = \\{A_1, \\ldots\\}``: ``c(\\bigcup_k \\{A_k = a_k\\})``;
    column `m + j` for future node ``\\beta_j``: ``c(\\bigcup_l \\{B_l = b_l\\})``;
    column `2m + i + (j - 1) m`: ``c(\\bigcup_{k,l} \\{A_k \\cup B_l \\text{ take their values}\\})``,
    the union over all pairs (the `vec` layout of phiid.jl).
- `past::Vector{Vector{Int}}`, `future::Vector{Vector{Int}}`: the input masks of the
    variable sets of each node, read as past and as future sets.
"""
struct PhiIDPointwiseData
    lattice::PhiIDLattice
    sx::Matrix{Float64}
    past::Vector{Vector{Int}}
    future::Vector{Vector{Int}}
end

function PhiIDPointwiseData(n::Int)
    L = phiid_lattice(n)
    nodes = L.lattice.nodes
    m = length(nodes)
    past = [Int.(node) for node in nodes]
    future = [Int.(node) .<< n for node in nodes]
    sx = zeros((1 << 2n) - 1, 2m + m^2)
    for i in 1:m
        add_union_coefficients!(view(sx, :, i), past[i])
        add_union_coefficients!(view(sx, :, m + i), future[i])
    end
    for j in 1:m, i in 1:m
        pairs = [a | b for a in past[i] for b in future[j]]
        add_union_coefficients!(view(sx, :, 2m + i + (j - 1) * m), pairs)
    end
    return PhiIDPointwiseData(L, sx, past, future)
end

const PHIID_POINTWISE_DATA = Dict{Int,PhiIDPointwiseData}()
const PHIID_POINTWISE_DATA_LOCK = ReentrantLock()

# Built once per number of variables and cached (thread-safe); do not modify.
function phiid_pointwise_data(n::Int)
    phiid_lattice(n) # validates n
    lock(PHIID_POINTWISE_DATA_LOCK) do
        get!(() -> PhiIDPointwiseData(n), PHIID_POINTWISE_DATA, n)
    end
end

# Blocks of states that bound the temporaries of `phiid_state_redundancies` (`width`
# columns each) to about 32 MiB.
state_blocks(K::Int, width::Int) = Iterators.partition(1:K, max(1, (1 << 22) ÷ width))

# ----------------------------------------------------------------------------------------
# Local double redundancies, in nats, of distinct joint states.
# ----------------------------------------------------------------------------------------
# The local double redundancies of the states `rows` of `sc`: a `length(rows) × m²`
# matrix in the `vec` layout (column i + (j - 1) m = past node i → future node j).
#
# ISx: h_sx(α) + h_sx(β) - h_sx(α × β) with h_sx = log N - log c(union); the union counts
# are one matrix product with the inclusion–exclusion coefficients (exact integers).
function phiid_state_redundancies(::ISx, sc::SubsetCounts, data::PhiIDPointwiseData, rows)
    m = nnodes(data.lattice)
    U = view(sc.counts, rows, :) * data.sx
    map!(log, U, U)
    R = Matrix{Float64}(undef, length(rows), m^2)
    logN = log(sc.N)
    for j in 1:m, i in 1:m
        c = i + (j - 1) * m
        @views @. R[:, c] = logN + U[:, 2m + c] - U[:, i] - U[:, m + j]
    end
    return R
end

# IPM: min_k h(a_k) + min_l h(b_l) - min_{k,l} h(a_k, b_l) with h = log N - log c, i.e.
# log N + max log c(a_k, b_l) - max log c(a_k) - max log c(b_l).
function phiid_state_redundancies(::IPM, sc::SubsetCounts, data::PhiIDPointwiseData, rows)
    m = nnodes(data.lattice)
    lc = log.(view(sc.counts, rows, :))
    b = length(rows)
    function colmax!(out, masks)
        out .= view(lc, :, first(masks))
        for M in Iterators.drop(masks, 1)
            out .= max.(out, view(lc, :, M))
        end
        return out
    end
    lp, lf = Matrix{Float64}(undef, b, m), Matrix{Float64}(undef, b, m)
    for i in 1:m
        colmax!(view(lp, :, i), data.past[i])
        colmax!(view(lf, :, i), data.future[i])
    end
    R = Matrix{Float64}(undef, b, m^2)
    logN = log(sc.N)
    for j in 1:m, i in 1:m
        r = view(R, :, i + (j - 1) * m)
        colmax!(r, (A | B for A in data.past[i] for B in data.future[j]))
        @views @. r = logN + r - lp[:, i] - lf[:, j]
    end
    return R
end

# The per-sample codes of the 2n inputs `(past..., future...)` for a JointProbabilities ΦID
# estimator (as in discrete.jl).
function phiid_codes(est::JointProbabilities, past::Tuple, future::Tuple)
    if !(est.discretization isa CodifyVariables) || !(est.pest isa RelativeAmount)
        name = nameof(typeof(est.definition.redundancy))
        throw(ArgumentError(
            "ΦID with $name needs a `CodifyVariables` discretization and the default " *
            "`RelativeAmount` probabilities estimator (plug-in probabilities), e.g. " *
            "`JointProbabilities(IntegratedInformationDecomposition($name()), " *
            "CodifyVariables(UniqueElements()))`. Got discretization " *
            "$(nameof(typeof(est.discretization))) and probabilities estimator " *
            "$(nameof(typeof(est.pest)))."))
    end
    return joint_codes(est.discretization, past..., future...)
end

# ----------------------------------------------------------------------------------------
# Average double redundancies: the `phiid_redundancies` hook.
# ----------------------------------------------------------------------------------------
# The weighted mean, over the distinct joint states, of the local double redundancies.
function phiid_redundancies(r::LocalRedundancy, est::JointProbabilities, lattice::PhiIDLattice,
        past::Tuple, future::Tuple)
    n = length(past)
    data = phiid_pointwise_data(n)
    data.lattice.lattice.nodes == lattice.lattice.nodes ||
        error("Unexpected ΦID lattice for $n variables.")
    sc = SubsetCounts(phiid_codes(est, past, future))
    m = nnodes(lattice)
    w = sc.weight ./ sc.N
    out = zeros(m^2)
    for rows in state_blocks(length(sc.weight), size(data.sx, 2))
        R = phiid_state_redundancies(r, sc, data, rows)
        mul!(out, R', view(w, rows), true, true)
    end
    out .*= convert_logunit(1.0, ℯ, est.definition.base)
    return reshape(out, m, m)
end

function phiid_discrete_only(est, what)
    name = nameof(typeof(est.definition.redundancy))
    throw(ArgumentError(
        "$what with $name $(what == "ΦID" ? "is" : "are") defined for discrete data only, " *
        "from plug-in counts of the joint states. Use `JointProbabilities` with a " *
        "`CodifyVariables` discretization, e.g. " *
        "`JointProbabilities(IntegratedInformationDecomposition($name()), " *
        "CodifyVariables(ValueBinning(RectangularBinning(8))))`, or " *
        "`IntegratedInformationDecomposition(IMMI())` for other estimators. Got a " *
        "$(describe(est))."))
end

function phiid_redundancies(::LocalRedundancy, est, lattice::PhiIDLattice, past::Tuple, future::Tuple)
    phiid_discrete_only(est, "ΦID")
end

# ----------------------------------------------------------------------------------------
# Per-sample atoms.
# ----------------------------------------------------------------------------------------
"""
    LocalInformation.local_phiid_from_past_future(est, past::Tuple, future::Tuple;
        backend = CPU(), precision = nothing, memory_limit = 2^31) → LocalPhiIDResult

The per-sample ΦID with [`ISx`](@ref) or [`IPM`](@ref) of the system whose past variables
are `past` and future variables `future` (`n = 2` or `3` each, all with the same number of
samples `N`), for `est = JointProbabilities(IntegratedInformationDecomposition(r),
CodifyVariables(o))`: the per-sample analogue of
[`LocalInformation.phiid_from_past_future`](@ref), with no lagging or shared encoding.
Row `t` of the result is sample `t`. `local_association(est, x...)` calls this after
[`LocalInformation.phiid_prepare`](@ref); it is also the entry point for an exact joint
distribution given as samples of `(past, future)` states. The keywords are those of
`local_association`.
"""
function local_phiid_from_past_future(est::JointProbabilities{<:PhiID{<:LocalRedundancy}},
        past::Tuple, future::Tuple; backend = CPU(), precision = nothing, memory_limit::Real = 2^31)
    T = resolve_precision(backend, precision)
    n = length(past)
    length(future) == n || throw(ArgumentError(
        "The past and the future must have the same number of variables; got " *
        "$n and $(length(future))."))
    check_phiid_count(est.definition, n)
    N = check_phiid_inputs((past..., future...))
    L = phiid_lattice(n)
    data = phiid_pointwise_data(n)
    r = est.definition.redundancy
    check_phiid_pointwise_memory(backend, T, r, n, N, nothing, memory_limit) # before counting
    sc = SubsetCounts(phiid_codes(est, past, future))
    K = length(sc.weight)
    check_phiid_pointwise_memory(backend, T, r, n, N, K, memory_limit)
    R = Matrix{Float64}(undef, K, nnodes(L)^2)
    for rows in state_blocks(K, size(data.sx, 2))
        R[rows, :] = phiid_state_redundancies(r, sc, data, rows)
    end
    R .*= convert_logunit(1.0, ℯ, est.definition.base)
    return local_phiid_result(est.definition, L, R, sc.state; backend, precision, memory_limit)
end

# Estimated peak memory (bytes) of the per-sample ΦID with ISx/IPM: the subset counts and
# local redundancies of `K` states (Float64), then `phiid_local_atoms` for `N` samples.
# With `K = nothing` (before counting) only the parts that do not depend on `K` count.
function check_phiid_pointwise_memory(backend, ::Type{T}, r, n, N, K, limit) where {T}
    m2 = nnodes(phiid_lattice(n))^2
    k = something(K, 0)
    need = 8k * ((1 << 2n) - 1) + 8k * m2
    need += check_phiid_memory(backend, T, Float64, k, N, true, m2, Inf, n)
    if need > limit
        states = K === nothing ? "" : " ($K distinct states)"
        throw(ArgumentError(
            "Local ΦID atoms with $(nameof(typeof(r))) ($m2 atoms of $n variables) for $N " *
            "samples$states would need about $(round(need / 2^30; digits = 2)) GiB, more " *
            "than `memory_limit` = $(round(limit / 2^30; digits = 2)) GiB. Use fewer " *
            "samples or raise `memory_limit`."))
    end
    return need
end

"""
    local_association(est::JointProbabilities{<:IntegratedInformationDecomposition{<:Union{ISx,IPM}}}, x₁, …, xₙ;
        backend = CPU(), precision = nothing, memory_limit = 2^31) → LocalPhiIDResult

Per-sample integrated information decomposition (ΦID) of `n = 2` or `3` variables with
the pointwise double redundancies [`ISx`](@ref) and [`IPM`](@ref), from plug-in
probabilities (discrete data: `JointProbabilities` with a `CodifyVariables`
discretization). Returns a [`LocalPhiIDResult`](@ref) with one row per lagged sample
`t = 1, …, N - τ`, the time index of the past ``\\mathbf{X}_t``: use `atom(res, α => β)`,
`atom(res, :xty)` (two variables), `atoms(res)` or
[`information_dynamics`](@ref)`(res)` for per-sample vectors. For every atom,
`mean(atom(res, a)) ≈ atom(association(est, x₁, …, xₙ), a)`.

The inputs are those of `association` (see [`IntegratedInformationDecomposition`](@ref)):
each series is encoded once over its full length, then lagged by `τ`. With an outcome
space that embeds the data (e.g. `OrdinalPatterns`), rows are lagged encoded symbols.

## Double redundancies

Let a sample have, for each set ``A_k`` of the past node
``\\alpha = \\{A_1, \\ldots, A_K\\}``, the joint outcome ``a_k`` of the past variables in
``A_k``, and for each set ``B_l`` of the future node ``\\beta = \\{B_1, \\ldots, B_L\\}`` the
outcome ``b_l`` of the future variables in ``B_l``. Write ``h(\\cdot) = -\\log P(\\cdot)``
(plug-in probabilities) and ``a_1 \\cup \\ldots \\cup a_K`` for the event that at least one
set takes its value of the sample.

- [`ISx`](@ref): the shared-exclusion double redundancy of Varley (2023) (`i_tsx` in
    syntropy),

```math
i^{sx}_\\cap(\\alpha \\to \\beta) = h_{sx}(\\alpha) + h_{sx}(\\beta) - h_{sx}(\\alpha \\times \\beta),
```

with ``h_{sx}(\\alpha) = -\\log P(a_1 \\cup \\ldots \\cup a_K)``,
``h_{sx}(\\beta) = -\\log P(b_1 \\cup \\ldots \\cup b_L)`` and
``h_{sx}(\\alpha \\times \\beta) = -\\log P\\big(\\bigcup_{k,l} (a_k \\cap b_l)\\big)``, the
union over all ``K L`` pairs of the joint events of ``A_k \\cup B_l``. Union probabilities
are computed exactly by inclusion–exclusion over subsets of the events (the intersection
of events is the event of the union of their variable sets), from integer counts.

- [`IPM`](@ref): the ``i_\\pm`` of Finn & Lizier (2018) with past and future source sets,

```math
i_\\pm(\\alpha \\to \\beta) = \\min_k h(a_k) + \\min_l h(b_l) - \\min_{k,l} h(a_k, b_l).
```

For single sets (``\\alpha = \\{A\\}``, ``\\beta = \\{B\\}``) both are the local mutual
information ``i(a; b)``; with ``\\beta = \\{\\{1, \\ldots, n\\}\\}`` (the joint future) they are
the PID local redundancy of the same name of the past sets about the joint future. So the
local atoms of a sample sum to its local time-delayed mutual information
``i(\\mathbf{x}_t; \\mathbf{x}_{t+\\tau})``, and for every past node ``\\alpha``,
``\\sum_\\beta \\Pi(\\alpha \\to \\beta)`` is the local PID atom ``\\alpha`` of the past
variables about the joint future (and symmetrically for the future nodes). The average
double redundancy of `association` is the mean of the local ones.

The local atoms of a sample are the inversion of its local double redundancies ``R``,
``\\Pi = Z^{-\\top} R Z^{-1}`` with the zeta matrix ``Z`` of the PID lattice. Local
redundancies only depend on the joint state of a sample (past and future), so they are
computed once per distinct joint state, inverted for all states at once as one matrix
product with ``Z^{-1} \\otimes Z^{-1}`` (see
[`LocalInformation.phiid_local_atoms`](@ref)), and copied to the samples.

## Keywords

- `backend`: where the inversion (a matrix product) runs, e.g. `CUDABackend()`. Counting
    and local redundancies are computed on the CPU.
- `precision`: element type of the inversion and of the result (default `Float64` on
    the CPU, `Float32` on a GPU). Local redundancies are computed in `Float64`.
- `memory_limit`: an `ArgumentError` is thrown, before any large allocation, if the
    estimated memory (bytes) exceeds this; e.g. three variables (324 atoms) need about
    ``8 \\cdot 324 N`` bytes for the result alone in `Float64`.

`EntropyDecomposition` and `GaussianDecomposition` throw an `ArgumentError` with `ISx` and
`IPM`.

## Example

```julia
using LocalInformation, Associations, Statistics
x, y = rand(0:1, 10_001), rand(0:1, 10_001)
y[2:end] .= x[1:end-1] .⊻ (rand(10_000) .< 0.1)    # noisy copy of x into y
est = JointProbabilities(IntegratedInformationDecomposition(ISx()), CodifyVariables(UniqueElements()))
loc = local_association(est, x, y)                  # LocalPhiIDResult, 10_000 × 16 atoms
mean(atom(loc, :xty)) ≈ atom(association(est, x, y), :xty)
information_dynamics(loc).transfer                  # per-sample transfer
```

## References

- Varley, T. F. (2023). Decomposing past and future: Integrated information decomposition
    based on shared probability mass exclusions. PLOS ONE, 18(3), e0282950.
- Makkeh, A., Gutknecht, A. J., & Wibral, M. (2021). Introducing a differentiable measure
    of pointwise shared information. Physical Review E, 103(3), 032149.
- Finn, C., & Lizier, J. T. (2018). Pointwise partial information decomposition using the
    specificity and ambiguity lattices. Entropy, 20(4), 297.
- Mediano, P. A. M., Rosas, F. E., Luppi, A. I., Carhart-Harris, R. L., Bor, D.,
    Seth, A. K., & Barrett, A. B. (2021). Towards an extended taxonomy of information
    dynamics via Integrated Information Decomposition. arXiv:2109.13186.
"""
function local_association(est::JointProbabilities{<:PhiID{<:LocalRedundancy}}, x...;
        backend = CPU(), precision = nothing, memory_limit::Real = 2^31)
    resolve_precision(backend, precision) # validate before encoding
    est′, past, future = phiid_prepare(est, x...)
    return local_phiid_from_past_future(est′, past, future; backend, precision, memory_limit)
end

# One method per estimator family, each more specific than phiid.jl's fallback on
# `PHIID_ESTIMATORS` (and than family-wide methods for other redundancy functions).
function local_association(est::EntropyDecomposition{<:PhiID{<:LocalRedundancy}}, x...; kwargs...)
    phiid_discrete_only(est, "Per-sample ΦID atoms")
end
function local_association(est::GaussianDecomposition{<:PhiID{<:LocalRedundancy}}, x...; kwargs...)
    phiid_discrete_only(est, "Per-sample ΦID atoms")
end
