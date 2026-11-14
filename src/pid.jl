# Partial information decomposition (PID) with local (pointwise) redundancy functions,
# and per-sample PID atoms.
#
# Two redundancy functions with a pointwise form are added to Associations' PID through its
# extension API (`Associations.pid_redundancies`): `ISx` (Makkeh, Gutknecht & Wibral 2021)
# and `IPM` (Finn & Lizier 2018). Their redundancy for a lattice node α is the sample mean
# of a local redundancy r(α)_i, so the per-sample atoms are the Möbius inversion of the
# local redundancies, and their means are the atoms of `association(est, x...)`.
#
# Everything is computed from plug-in counts. A local redundancy depends on a sample only
# through its joint state (all sources and the target), so redundancies and atoms are
# computed once per distinct joint state and then copied to the samples.

using Associations: PartialInformationDecomposition, PIDAtom, RedundancyFunction, RedundancyLattice,
    MIDecomposition, IMin, IMMI, redundancy_lattice, lattice_node_vectors
import Associations: atom, atoms, synergy, redundancy, unique_information
using LinearAlgebra: UnitUpperTriangular

export ISx, IPM, LocalPIDResult

"""
    ISx <: RedundancyFunction
    ISx()

The pointwise shared-exclusion redundancy ``i^{sx}_\\cap`` of
Makkeh, Gutknecht & Wibral (2021), for a [`PartialInformationDecomposition`](@ref).

## Description

Let a sample have the target outcome ``t`` and, for each source set ``A_j`` of the
lattice node ``\\alpha = \\{A_1, \\ldots, A_k\\}``, the joint outcome ``a_j`` of the sources
in ``A_j``. Write ``a_1 \\cup \\ldots \\cup a_k`` for the event that *at least one* source
set takes its value of the sample (the "shared exclusion": the outcomes ruled out by all of
``a_1, \\ldots, a_k``). The local redundancy is the local mutual information between that
event and ``t``,

```math
i^{sx}_\\cap(\\alpha; t)
= \\log \\frac{P(t \\cap (a_1 \\cup \\ldots \\cup a_k))}{P(t)\\, P(a_1 \\cup \\ldots \\cup a_k)}
= \\log \\frac{P(t) - P(t \\cap \\bar a_1 \\cap \\ldots \\cap \\bar a_k)}
              {1 - P(\\bar a_1 \\cap \\ldots \\cap \\bar a_k)} - \\log P(t),
```

or, with the shared-exclusion entropy ``h_{sx}(\\alpha) = -\\log P(a_1 \\cup \\ldots \\cup a_k)``,
``i^{sx}_\\cap(\\alpha; t) = h_{sx}(\\alpha) + h(t) - h_{sx}(\\alpha, t)``, where
``h_{sx}(\\alpha, t) = -\\log P((a_1 \\cap t) \\cup \\ldots \\cup (a_k \\cap t))``. For a single
source set it is the local mutual information ``i(a; t)``. The redundancy of ``\\alpha`` is
the mean of ``i^{sx}_\\cap(\\alpha; t)`` over samples.

The probabilities are plug-in estimates. The probability of the union is computed by
inclusion–exclusion over the subsets ``J`` of ``\\alpha``, from the counts of the joint
outcomes of the source sets ``\\bigcup_{j \\in J} A_j`` (exactly, in integer arithmetic).

## Usage

Discrete data only: use [`JointProbabilities`](@ref) with `CodifyVariables`, e.g.

```julia
est = JointProbabilities(PartialInformationDecomposition(ISx()), CodifyVariables(UniqueElements()))
association(est, s1, s2, t)          # PIDResult (average atoms)
local_association(est, s1, s2, t)    # LocalPIDResult (per-sample atoms)
```

Atoms can be negative, also on average (e.g. the redundancy of XOR is ``\\log_2(2/3)``).

## Reference

Makkeh, A., Gutknecht, A. J., & Wibral, M. (2021). Introducing a differentiable measure of
pointwise shared information. Physical Review E, 103(3), 032149.
"""
struct ISx <: RedundancyFunction end

"""
    IPM <: RedundancyFunction
    IPM()

The pointwise redundancy ``i_\\pm`` of Finn & Lizier (2018), for a
[`PartialInformationDecomposition`](@ref).

## Description

With the notation of [`ISx`](@ref) (``a_j`` the joint outcome of source set ``A_j`` in a
sample, ``t`` the target outcome), the local redundancy of ``\\alpha = \\{A_1, \\ldots, A_k\\}``
is the difference of a redundant *specificity* and a redundant *ambiguity*,

```math
i_\\pm(\\alpha; t) = i_+(\\alpha) - i_-(\\alpha; t)
= \\min_j h(a_j) - \\min_j h(a_j \\mid t),
\\qquad h(a) = -\\log P(a),\\; h(a \\mid t) = -\\log P(a \\mid t),
```

so that ``i_\\pm(\\alpha; t) = \\log \\max_j P(a_j, t) - \\log \\max_j P(a_j) - \\log P(t)``.
For a single source set it is the local mutual information ``i(a; t)``. Finn & Lizier
decompose ``i_+`` and ``i_-`` separately (on the specificity and ambiguity lattices) and
subtract the atoms; Möbius inversion is linear, so this gives the same atoms as inverting
``i_\\pm`` directly, which is what is done here. The redundancy of ``\\alpha`` is the mean of
``i_\\pm(\\alpha; t)`` over samples, with plug-in probabilities.

## Usage

Discrete data only: use [`JointProbabilities`](@ref) with `CodifyVariables`, as for
[`ISx`](@ref). `local_association` gives per-sample atoms.

## Reference

Finn, C., & Lizier, J. T. (2018). Pointwise partial information decomposition using the
specificity and ambiguity lattices. Entropy, 20(4), 297.
"""
struct IPM <: RedundancyFunction end

const LocalRedundancy = Union{ISx,IPM}

# ----------------------------------------------------------------------------------------
# Plug-in counts of every source set, per distinct joint state.
# ----------------------------------------------------------------------------------------
"""
    LocalInformation.PIDCounts

Plug-in counts for a PID of `n` sources and a target, per distinct joint state of all the
inputs (`K` states, `N` samples). Source set `M` is the bitmask of Associations'
redundancy lattice (source `i` is bit `i - 1`).

- `state[i]`: the joint state (`1:K`) of sample `i`; `weight[s]`: its number of samples.
- `src[s, M]`: number of samples whose sources in `M` have the values of state `s`.
- `srct[s, M]`: the same, jointly with the target.
- `tgt[s]`: number of samples with the target value of state `s`.

Counts are stored as `Float64` (exact for counts below 2⁵³), so that inclusion–exclusion
sums are matrix products.
"""
struct PIDCounts
    nsources::Int
    N::Int
    state::Vector{Int}
    weight::Vector{Int}
    src::Matrix{Float64}
    srct::Matrix{Float64}
    tgt::Vector{Float64}
end

# Relabel the values of `v` as 1, 2, … in order of appearance. Returns labels and count.
function relabel(v::AbstractVector)
    d = Dict{eltype(v),Int}()
    out = Vector{Int}(undef, length(v))
    for (i, x) in pairs(v)
        out[i] = get!(d, x, length(d) + 1)
    end
    return out, length(d)
end

# Labels of the joint outcome of two labelled variables (labels in 1:ka and 1:kb).
function joint_labels(a::Vector{Int}, ka::Int, b::Vector{Int}, kb::Int)
    N = length(a)
    out = Vector{Int}(undef, N)
    k = 0
    if widemul(ka, kb) ≤ max(4N, 1 << 16) # dense lookup table
        table = zeros(Int, ka * kb)
        for i in 1:N
            key = (a[i] - 1) * kb + b[i]
            l = table[key]
            l == 0 && (l = table[key] = (k += 1))
            out[i] = l
        end
    else
        d = Dict{Int,Int}()
        for i in 1:N
            out[i] = get!(d, (a[i] - 1) * kb + b[i], length(d) + 1)
        end
        k = length(d)
    end
    return out, k
end

function PIDCounts(codes::AbstractVector)
    N = check_input_lengths(codes...)
    n = length(codes) - 1
    lab = relabel.(codes)
    # Distinct joint states of all inputs, one representative sample each.
    state, K = lab[1]
    for (l, k) in lab[2:end]
        state, K = joint_labels(state, K, l, k)
    end
    weight = zeros(Int, K)
    rep = zeros(Int, K)
    for i in 1:N
        s = state[i]
        weight[s] += 1
        rep[s] == 0 && (rep[s] = i)
    end
    nm = (1 << n) - 1
    src = Matrix{Float64}(undef, K, nm)
    srct = Matrix{Float64}(undef, K, nm)
    tgt = Vector{Float64}(undef, K)
    counts = Int[]
    # Depth-first over the masks of the n + 1 inputs (bit n is the target), building each
    # joint labelling from its parent's, so at most n + 1 label vectors are alive.
    function visit(l, k, mask, topbit)
        resize!(counts, k)
        fill!(counts, 0)
        for x in l
            counts[x] += 1
        end
        col = mask & nm
        dest = mask == col ? view(src, :, col) : col == 0 ? tgt : view(srct, :, col)
        for s in 1:K
            dest[s] = counts[l[rep[s]]]
        end
        for b in (topbit + 1):n
            lb, kb = joint_labels(l, k, lab[b + 1]...)
            visit(lb, kb, mask | (1 << b), b)
        end
    end
    for b in 0:n
        visit(lab[b + 1]..., 1 << b, b)
    end
    return PIDCounts(n, N, state, weight, src, srct, tgt)
end

# The per-sample codes of `x...` for a JointProbabilities PID estimator (as in discrete.jl).
function pid_codes(est::JointProbabilities, x...)
    if !(est.discretization isa CodifyVariables) || !(est.pest isa RelativeAmount)
        throw(ArgumentError(
            "$(nameof(typeof(est.definition.redundancy))) needs a `CodifyVariables` " *
            "discretization and the default `RelativeAmount` probabilities estimator " *
            "(plug-in probabilities), e.g. `JointProbabilities(PartialInformationDecomposition(" *
            "$(nameof(typeof(est.definition.redundancy)))()), CodifyVariables(UniqueElements()))`. " *
            "Got discretization $(nameof(typeof(est.discretization))) and probabilities " *
            "estimator $(nameof(typeof(est.pest)))."
        ))
    end
    return joint_codes(est.discretization, x...)
end

# ----------------------------------------------------------------------------------------
# Lattice data, cached per number of sources.
# ----------------------------------------------------------------------------------------
"""
    LocalInformation.PIDLatticeData

Data derived once per number of sources `n` from Associations' (shared, never modified)
`redundancy_lattice(n)`:

- `sx`: the `(2ⁿ - 1) × m` inclusion–exclusion coefficients of [`ISx`](@ref):
    ``P(a_1 \\cup \\ldots \\cup a_k) = \\sum_M`` `sx[M, node]` ``P(x_M)``, with
    `sx[M, node]` ``= \\sum_{J \\subseteq \\alpha,\\, \\bigcup J = M} (-1)^{|J|+1}``.
- `zeta`: the zeta matrix of the lattice in the order `lattice.order`, `zeta[a, b] = 1`
    if node `order[a]` ≼ node `order[b]`. It is unit upper triangular, and the atoms are
    ``A = R Z^{-1}`` for local redundancies `R` (one column per node, in that order).
"""
struct PIDLatticeData
    lattice::RedundancyLattice
    sx::Matrix{Float64}
    zeta::BitMatrix
end

function PIDLatticeData(n::Int)
    L = redundancy_lattice(n)
    m = length(L.nodes)
    sx = zeros((1 << n) - 1, m)
    for (j, node) in enumerate(L.nodes)
        k = length(node)
        for J in 1:((1 << k) - 1)
            M = UInt(0)
            for e in 1:k
                (J >> (e - 1)) & 1 == 1 && (M |= node[e])
            end
            sx[M, j] += isodd(count_ones(J)) ? 1 : -1
        end
    end
    zeta = L.below[L.order, L.order]
    for a in 1:m
        zeta[a, a] = true
    end
    return PIDLatticeData(L, sx, zeta)
end

const PID_LATTICE_DATA = Dict{Int,PIDLatticeData}()
const PID_LATTICE_DATA_LOCK = ReentrantLock()

function pid_lattice_data(n::Int)
    lock(PID_LATTICE_DATA_LOCK) do
        get!(() -> PIDLatticeData(n), PID_LATTICE_DATA, n)
    end
end

# The Möbius function μ(i, j) of the lattice for a fixed node j, as the sparse column j of
# Z⁻¹: node indices i ≼ j with μ(i, j) ≠ 0, and the values. Exact integer recursion
# μ(j, j) = 1, μ(i, j) = -Σ_{i ≺ k ≼ j} μ(k, j), over the nodes below j from the top down;
# only nonzero terms are kept, and atom j is Σ_i μ(i, j) r(i).
function mobius_column(L::RedundancyLattice, j::Int)
    idx, val = [j], [1]
    below = L.below
    for i in Iterators.reverse(L.order)
        below[i, j] || continue
        s = 0
        for (k, v) in zip(idx, val)
            (k == j || below[i, k]) && (s += v)
        end
        s == 0 || (push!(idx, i); push!(val, -s))
    end
    return idx, val
end

# ----------------------------------------------------------------------------------------
# Local redundancies, in nats, of the distinct states.
# ----------------------------------------------------------------------------------------
# Fill `R[:, k]` with the local redundancies of `lattice.nodes[nodes[k]]` for every state.
# `c0[s] = log N - log n(t_s) = h(t)` of state s.
function fill_redundancies!(R, ::ISx, pc::PIDCounts, data::PIDLatticeData, nodes)
    K = size(R, 1)
    c0 = log(pc.N) .- log.(pc.tgt)
    # Union counts are sums over source sets with the inclusion–exclusion coefficients:
    # two matrix products (exact integers in Float64). Blocks bound the temporaries.
    bs = max(1, (1 << 22) ÷ max(K, 1))
    for b in Iterators.partition(eachindex(nodes), bs)
        C = data.sx[:, nodes[b]]
        U = pc.src * C   # count of samples matching at least one a_j
        UT = pc.srct * C # the same, with the target value t
        @views @. R[:, b] = log(UT) - log(U) + c0
    end
    return R
end

function fill_redundancies!(R, ::IPM, pc::PIDCounts, data::PIDLatticeData, nodes)
    c0 = log(pc.N) .- log.(pc.tgt)
    lsrc, lsrct = log.(pc.src), log.(pc.srct)
    L = data.lattice
    for (k, j) in enumerate(nodes)
        node = L.nodes[j]
        r = view(R, :, k)
        a, at = view(lsrc, :, node[1]), view(lsrct, :, node[1])
        @. r = at - a
        if length(node) > 1
            amax, atmax = copy(a), copy(at)
            for M in node[2:end]
                @views amax .= max.(amax, lsrc[:, M])
                @views atmax .= max.(atmax, lsrct[:, M])
            end
            @. r = atmax - amax
        end
        r .+= c0
    end
    return R
end

# ----------------------------------------------------------------------------------------
# Average redundancies: the extension hook of Associations' PID.
# ----------------------------------------------------------------------------------------
function Associations.pid_redundancies(r::LocalRedundancy, est::JointProbabilities, lattice::RedundancyLattice, x...)
    pc = PIDCounts(pid_codes(est, x...))
    data = pid_lattice_data(lattice.nsources)
    data.lattice === lattice || data.lattice.nodes == lattice.nodes ||
        error("Unexpected redundancy lattice for $(lattice.nsources) sources.")
    m = length(lattice.nodes)
    K = length(pc.weight)
    w = pc.weight ./ pc.N
    out = Vector{Float64}(undef, m)
    bs = max(1, (1 << 24) ÷ max(K, 1))
    for b in Iterators.partition(1:m, bs)
        R = fill_redundancies!(Matrix{Float64}(undef, K, length(b)), r, pc, data, b)
        out[b] = R' * w
    end
    out .*= convert_logunit(1.0, ℯ, est.definition.base)
    return out
end

function Associations.pid_redundancies(r::LocalRedundancy, est, lattice::RedundancyLattice, x...)
    name = nameof(typeof(r))
    throw(ArgumentError(
        "$name is defined for discrete data only. Use `JointProbabilities` with a " *
        "discretization, e.g. `JointProbabilities(PartialInformationDecomposition($name()), " *
        "CodifyVariables(ValueBinning(RectangularBinning(8))))`. " *
        "Got a $(nameof(typeof(est))) estimator."
    ))
end

# ----------------------------------------------------------------------------------------
# Per-sample atoms.
# ----------------------------------------------------------------------------------------
"""
    LocalPIDResult

The per-sample (local) partial information decomposition returned by
[`local_association`](@ref) for a `JointProbabilities` estimator of a
`PartialInformationDecomposition` with [`ISx`](@ref) or [`IPM`](@ref).

## Fields

- `definition`: the `PartialInformationDecomposition`.
- `lattice`: Associations' redundancy lattice (shared; do not modify).
- `nodes::Vector{Int}`: the lattice node (index into `lattice.nodes`) of each column of
    `atoms`. With `atoms = :all` these are all nodes, bottom (redundancy) first and top
    (synergy) last, as in `Associations.atoms(::PIDResult)`.
- `atoms::Matrix{T}`: `N × length(nodes)`, the local atoms of every sample (rows) in the
    unit `definition.base`.

## Accessors

The Associations.jl accessors return per-sample vectors (views into `res.atoms`):

- `atom(res, α)`: the local atom of node `α`, e.g. `atom(res, [[1], [2, 3]])`.
- `synergy(res)`, `redundancy(res)`, `unique_information(res, i)`.
- `atoms(res)`: every selected atom as a `node => values` pair.

Accessing an atom that was not selected (with the `atoms` keyword) throws an
`ArgumentError`. With `atoms = :all`, the local atoms of a sample sum to its local mutual
information ``i(s_1, \\ldots, s_n; t)``, and `mean(atom(res, α))` equals
`atom(association(est, x...), α)`.
"""
struct LocalPIDResult{D<:PartialInformationDecomposition,T<:Real}
    definition::D
    lattice::RedundancyLattice
    nodes::Vector{Int}
    atoms::Matrix{T}
end

# Lattice node index of a user-given antichain such as `[[1], [2, 3]]`.
function node_index(L::RedundancyLattice, α)
    n = L.nsources
    masks = UInt[]
    for A in α
        if !(A isa Union{AbstractVector,Tuple,AbstractSet})
            throw(ArgumentError(
                "An atom is a collection of source sets, e.g. `[[1], [2, 3]]`; got $(repr(α)). " *
                "To select one atom with the `atoms` keyword, wrap it: `atoms = [[[1], [2]]]`."
            ))
        end
        idxs = collect(Int, A)
        isempty(idxs) && throw(ArgumentError("Each source set of an atom must be nonempty; got $(repr(α))."))
        all(i -> 1 ≤ i ≤ n, idxs) || throw(ArgumentError(
            "Source indices must be between 1 and $n; got $(repr(α))."))
        push!(masks, reduce(|, (UInt(1) << (i - 1) for i in idxs)))
    end
    node = sort!(unique!(masks))
    haskey(L.index, node) || throw(ArgumentError(
        "$(repr(α)) is not a node of the redundancy lattice: it must be a nonempty " *
        "collection of source sets, none of which contains another."))
    return L.index[node]
end

function column(res::LocalPIDResult, α)
    j = node_index(res.lattice, α)
    k = findfirst(==(j), res.nodes)
    if k === nothing
        sel = join((node_string(res.lattice.nodes[i]) for i in res.nodes), ", ")
        throw(ArgumentError(
            "The atom $(node_string(res.lattice.nodes[j])) was not computed; the selected " *
            "atoms are $sel. Use `atoms = :all` or include it in `atoms`."
        ))
    end
    return k
end

node_string(node) = join("{" * join(A) * "}" for A in lattice_node_vectors(node))

atom(res::LocalPIDResult, α) = view(res.atoms, :, column(res, α))
synergy(res::LocalPIDResult) = atom(res, [collect(1:res.lattice.nsources)])
redundancy(res::LocalPIDResult) = atom(res, [[i] for i in 1:res.lattice.nsources])
unique_information(res::LocalPIDResult, i::Integer) = atom(res, [[i]])
function atoms(res::LocalPIDResult)
    return [lattice_node_vectors(res.lattice.nodes[j]) => view(res.atoms, :, k)
            for (k, j) in enumerate(res.nodes)]
end

function Base.show(io::IO, ::MIME"text/plain", res::LocalPIDResult)
    N, c = size(res.atoms)
    r = nameof(typeof(res.definition.redundancy))
    println(io, "LocalPIDResult: $(res.lattice.nsources) sources, redundancy function $r, ",
        "base $(res.definition.base), $N samples × $c atoms")
    nshow = c ≤ 30 ? c : 20
    print(io, "  atom          mean over samples")
    for k in 1:nshow
        print(io, "\n  ", rpad(node_string(res.lattice.nodes[res.nodes[k]]), 14),
            sum(view(res.atoms, :, k)) / max(N, 1))
    end
    nshow < c && print(io, "\n  ⋮ ($(c - nshow) more atoms; use `atoms(res)`)")
end

"""
    local_association(est::JointProbabilities{<:PartialInformationDecomposition}, s₁, …, sₙ, t;
        atoms = :all, backend = CPU(), precision = nothing, memory_limit = 2^31) → LocalPIDResult

Per-sample partial information decomposition of the information that the sources
`s₁, …, sₙ` (2 to 5) carry about the target `t` (the last input), for the redundancy
functions [`ISx`](@ref) and [`IPM`](@ref). Returns a [`LocalPIDResult`](@ref): use
`atom(res, α)`, `synergy(res)`, `redundancy(res)`, `unique_information(res, i)` or
`atoms(res)` for per-sample vectors. For every atom `α`,
`mean(atom(res, α)) ≈ atom(association(est, s₁, …, sₙ, t), α)`.

The local atoms ``a`` of a sample are the Möbius inversion of its local redundancies
``r``: ``r_j = \\sum_{i \\preceq j} a_i``. With the zeta matrix ``Z`` of the lattice
(``Z_{ij} = 1`` if node ``i`` ≼ node ``j``), the atoms of all samples are ``A = R Z^{-1}``,
computed as one triangular solve (``Z`` is unit upper triangular in the order
`lattice.order`). Since local redundancies only depend on the joint state of a sample,
`R` has one row per distinct joint state.

## Keywords

- `atoms`: `:all` (default) or a collection of atoms, each a collection of source sets,
    e.g. `atoms = [[[1, 2, 3]], [[1], [2], [3]]]`. With a selection, only the needed
    columns of ``Z^{-1}`` (the Möbius function, computed exactly) and the redundancies of
    the nodes where it is nonzero are computed, which keeps 5 sources (7579 atoms) with
    many samples cheap.
- `backend`: where the Möbius inversion runs (a triangular solve for `atoms = :all`, a
    matrix product for a selection), e.g. `CUDABackend()`. Counting and local redundancies
    are computed on the CPU.
- `precision`: element type of the solve and of the result (default `Float64` on the
    CPU, `Float32` on a GPU). Local redundancies are computed in `Float64`.
- `memory_limit`: an `ArgumentError` is thrown, before any large allocation, if the
    estimated memory (bytes) exceeds this; e.g. `atoms = :all` for 5 sources needs about
    ``8 N m`` bytes for the result alone (``m = 7579``).

Only `JointProbabilities` with `CodifyVariables` (plug-in probabilities) is supported.
`IMin` and `IMMI` have no per-sample form and throw an `ArgumentError`.
"""
function local_association(
        est::JointProbabilities{<:PartialInformationDecomposition{<:LocalRedundancy}}, x...;
        atoms = :all, backend = CPU(), precision = nothing, memory_limit::Real = 2^31,
    )
    T = resolve_precision(backend, precision)
    Associations.verify_number_of_inputs_vars(est.definition, length(x))
    n = length(x) - 1
    data = pid_lattice_data(n)
    L = data.lattice
    m = length(L.nodes)
    pc = PIDCounts(pid_codes(est, x...))
    N, K = pc.N, length(pc.weight)
    if atoms === :all
        nodes = copy(L.order)
        ns = m
    else
        nodes = selected_nodes(L, atoms)
        # Sparse columns of Z⁻¹ for the selected nodes; R is needed on their support only.
        cols = [mobius_column(L, j) for j in nodes]
        support = sort!(unique!(reduce(vcat, first.(cols))))
        ns = length(support)
    end
    check_pid_memory(backend, T, K, N, ns, length(nodes), atoms === :all ? m : 0, memory_limit, n)
    r = est.definition.redundancy
    scale = convert_logunit(1.0, ℯ, est.definition.base)
    if atoms === :all
        R = fill_redundancies!(Matrix{Float64}(undef, K, m), r, pc, data, nodes)
        R .*= scale
        Rd = to_device(backend, R, T)
        rdiv!(Rd, UnitUpperTriangular(to_device(backend, data.zeta, T)))
        AK = to_host(Rd)
    else
        pos = Dict(i => p for (p, i) in enumerate(support))
        μ = zeros(Int, ns, length(nodes))
        for (c, (idx, val)) in enumerate(cols), (i, v) in zip(idx, val)
            μ[pos[i], c] = v
        end
        R = fill_redundancies!(Matrix{Float64}(undef, K, ns), r, pc, data, support)
        R .*= scale
        AK = to_host(to_device(backend, R, T) * to_device(backend, μ, T))
    end
    return LocalPIDResult(est.definition, L, nodes, AK[pc.state, :])
end

# Estimated peak memory (bytes) of `local_association` for a PID: `K` distinct states,
# `N` samples, `ns` redundancy columns, `nsel` atoms, and an `mz × mz` zeta matrix for the
# triangular solve (`mz = 0` for a selection, which uses an `ns × nsel` Möbius matrix).
function check_pid_memory(backend, ::Type{T}, K, N, ns, nsel, mz, limit, n) where {T}
    s = sizeof(T)
    oncpu = backend isa KA.CPU
    need = 8K * ns                                   # local redundancies (Float64, host)
    need += (oncpu && T === Float64) ? 0 : s * K * ns # converted / device copy
    if mz > 0
        need += s * mz^2 * (oncpu ? 1 : 2)           # zeta matrix (host and device)
        need += oncpu ? 0 : s * K * nsel             # atoms copied back to the host
    else
        need += s * ns * nsel * (oncpu ? 1 : 2) + s * K * nsel * (oncpu ? 1 : 2)
    end
    need += s * N * nsel                             # the result
    if need > limit
        throw(ArgumentError(
            "Local PID atoms for $nsel atoms of $n sources and $N samples ($K distinct " *
            "states) would need about $(round(need / 2^30; digits = 2)) GiB, more than " *
            "`memory_limit` = $(round(limit / 2^30; digits = 2)) GiB. Select fewer atoms, " *
            "e.g. `atoms = [[$(collect(1:n))]]` for the synergy only, use fewer samples, " *
            "or raise `memory_limit`."
        ))
    end
    return need
end

# A single atom (`PIDAtom`): its per-sample values, a `Vector`.
function local_association(
        est::JointProbabilities{<:PIDAtom{<:PartialInformationDecomposition{<:LocalRedundancy}}}, x...;
        backend = CPU(), precision = nothing, memory_limit::Real = 2^31,
    )
    a, n = est.definition.atom, length(x) - 1
    α = a === :synergy ? [collect(1:n)] : a === :redundancy ? [[i] for i in 1:n] :
        a isa Pair ? [[last(a)]] : a
    pidest = JointProbabilities(est.definition.pid, est.discretization, est.pest)
    res = local_association(pidest, x...; atoms = [α], backend, precision, memory_limit)
    return res.atoms[:, 1]
end

function selected_nodes(L::RedundancyLattice, atoms)
    if atoms isa Symbol || !(applicable(iterate, atoms))
        throw(ArgumentError(
            "`atoms` must be `:all` or a collection of atoms, e.g. " *
            "`atoms = [[[1, 2]], [[1], [2]]]`; got $(repr(atoms))."
        ))
    end
    nodes = unique!([node_index(L, α) for α in atoms])
    isempty(nodes) && throw(ArgumentError("`atoms` selects no atom."))
    return nodes
end

# No per-sample form for IMin and IMMI, nor for non-discrete estimators.
const PID_ESTIMATORS = Union{
    JointProbabilities{<:PartialInformationDecomposition},
    EntropyDecomposition{<:PartialInformationDecomposition},
    MIDecomposition{<:PartialInformationDecomposition},
    GaussianDecomposition{<:PartialInformationDecomposition},
}

function local_association(est::PID_ESTIMATORS, x...; kwargs...)
    r = est.definition.redundancy
    name = nameof(typeof(r))
    if r isa LocalRedundancy
        throw(ArgumentError(
            "Per-sample PID atoms with $name are defined for discrete data only. Use " *
            "`JointProbabilities` with a discretization, e.g. " *
            "`JointProbabilities(PartialInformationDecomposition($name()), " *
            "CodifyVariables(ValueBinning(RectangularBinning(8))))`. " *
            "Got a $(nameof(typeof(est))) estimator."
        ))
    end
    why = if r isa IMin
        "IMin's redundancy Σₜ p(t) min_A I(T = t; A) takes, for each target outcome, a " *
        "minimum over source sets of specific informations, which are averages over the " *
        "source outcomes"
    elseif r isa IMMI
        "IMMI's redundancy min_A I(A; T) is a minimum of mutual informations, i.e. of " *
        "averages over samples"
    else
        "$name has no per-sample form in LocalInformation.jl"
    end
    throw(ArgumentError(
        "Per-sample PID atoms need a redundancy function that is the mean of per-sample " *
        "(local) redundancies. $why; a minimum of averages is not an average of " *
        "per-sample terms, so it has no local atoms whose mean is the PID. Use a " *
        "pointwise redundancy function: `ISx()` (Makkeh et al. 2021) or `IPM()` (Finn & " *
        "Lizier 2018), e.g. `JointProbabilities(PartialInformationDecomposition(ISx()), " *
        "CodifyVariables(UniqueElements()))`."
    ))
end
