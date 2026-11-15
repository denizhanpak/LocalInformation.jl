# Integrated information decomposition (ΦID) of Mediano, Rosas et al.
#
# ΦID splits the time-delayed mutual information I(past; future) of n variables into atoms
# Π(α → β), one per pair of nodes of the PID redundancy lattice of n sources: α over the
# past variables and β over the future ones. Pairs are ordered by the product order,
# (α, β) ≼ (α', β') iff α ≼ α' and β ≼ β'. A double-redundancy function I∩∩(α → β) gives
#
#     R[α, β] = Σ_{α' ≼ α, β' ≼ β} Π[α', β'],   i.e.   R = Zᵀ Π Z,   Π = Z⁻ᵀ R Z⁻¹,
#
# with Z the zeta matrix of the PID lattice (Z[i, j] = 1 iff node i ≼ node j). For
# per-sample values, with local redundancies as rows, vec(Π)ᵀ = vec(R)ᵀ (Z⁻¹ ⊗ Z⁻¹).
#
# Layout conventions (shared with later phases):
# - Nodes are referred to by their index in `redundancy_lattice(n).nodes` ("node order").
# - Average redundancies and atoms are m × m matrices in node order, rows = past node,
#   columns = future node. Per-sample values are K × m² matrices whose column
#   c = i + (j - 1) m holds the pair (past node i → future node j), i.e. `vec` of the m × m
#   matrix.
# - Internally the system has 2n inputs: past variables 1..n, future variables n+1..2n.
#
# Pipeline: `association(est, x...)` → `phiid_prepare` (validate, encode discrete data once
# over the full series, lag by τ) → `phiid_from_past_future(est′, past, future)` →
# `phiid_redundancies(r, est′, lattice, past, future)` (the extension hook, m × m) →
# `phiid_mobius` → `PhiIDResult`. Per-sample atoms go through `phiid_local_atoms`.

using Associations: RedundancyFunction, RedundancyLattice, IMMI, IMin, redundancy_lattice,
    lattice_node_vectors, MultivariateInformationMeasureEstimator

export IntegratedInformationDecomposition, PhiIDResult, LocalPhiIDResult, information_dynamics

"""
    IntegratedInformationDecomposition <: MultivariateInformationMeasure
    IntegratedInformationDecomposition(redundancy = IMMI(); τ = 1, base = 2)

The integrated information decomposition (ΦID) of Mediano, Rosas et al. of the information
that the past of a system of `n` variables carries about its future, using the double
redundancy built from the [`RedundancyFunction`](@ref) `redundancy`.

## Usage

`association(est, x₁, …, xₙ)` with an estimator `est` whose `definition` is an
`IntegratedInformationDecomposition`. The result is a [`PhiIDResult`](@ref); use
`atom(res, α => β)`, `atoms(res)` and, for two variables, `atom(res, :xty)` and
[`information_dynamics`](@ref)`(res)`.

## Inputs

2 or 3 variables `x₁, …, xₙ` (16 or 324 atoms), each an `AbstractVector` or a
`StateSpaceSet` (a multivariate variable), all with the same number of samples `N > τ`.
The past of variable `k` is `xₖ[1:N-τ]` and its future `xₖ[1+τ:N]`.

## Description

Let ``\\mathbf{X}_t = (X^1_t, \\ldots, X^n_t)`` and ``\\mathbf{X}_{t+\\tau}`` be the past and the
future. The atoms ``\\Pi(\\alpha \\to \\beta)`` are indexed by pairs of nodes of the PID
redundancy lattice of `n` sources, ``\\alpha`` over the past variables and ``\\beta`` over the
future ones, ordered by ``(\\alpha, \\beta) \\preceq (\\alpha', \\beta')`` if and only if
``\\alpha \\preceq \\alpha'`` and ``\\beta \\preceq \\beta'``. Given a double redundancy
``I_\\cap^{\\alpha \\to \\beta}``, the atoms are defined by

```math
I_\\cap^{\\alpha \\to \\beta} = \\sum_{\\alpha' \\preceq \\alpha,\\; \\beta' \\preceq \\beta} \\Pi(\\alpha' \\to \\beta'),
```

i.e. with the zeta matrix ``Z`` of the PID lattice (``Z_{ij} = 1`` if node ``i \\preceq j``)
and the matrices ``R_{\\alpha\\beta} = I_\\cap^{\\alpha \\to \\beta}`` and ``\\Pi``,
``R = Z^\\top \\Pi Z`` and ``\\Pi = Z^{-\\top} R Z^{-1}``. The atoms sum to
``I(\\mathbf{X}_t; \\mathbf{X}_{t+\\tau})``. For every past node ``\\alpha``,
``\\sum_\\beta \\Pi(\\alpha \\to \\beta)`` is the PID atom ``\\alpha`` of the information that the
past variables carry about the joint future (and symmetrically for the future nodes).

With [`IMMI`](@ref) (minimum mutual information, the default), the double redundancy is

```math
I_\\cap^{\\alpha \\to \\beta} = \\min_{A \\in \\alpha,\\, B \\in \\beta} I(\\mathbf{X}^A_t; \\mathbf{X}^B_{t+\\tau}),
```

where ``\\mathbf{X}^A`` is the joint variable of the variables in the set ``A``. This is
the ΦID used by `phyid` (Luppi et al. 2022). Like the PID with ``I_{MMI}``, it gives
redundancy whenever several source sets carry the same *amount* of information: for two
independent copied bits (``X^1_{t+1} = X^1_t``, ``X^2_{t+1} = X^2_t``) it gives 1 bit of
``\\{1\\}\\{2\\} \\to \\{12\\}``, not only 1 bit each of ``\\{1\\} \\to \\{1\\}`` and
``\\{2\\} \\to \\{2\\}``. Other redundancy functions throw an `ArgumentError`.

For two variables the atoms have names (as in `phyid`): `:αtβ` with `α` (past) and `β`
(future) one of `r` = ``\\{1\\}\\{2\\}`` (redundancy), `x` = ``\\{1\\}``, `y` = ``\\{2\\}``,
`s` = ``\\{12\\}`` (synergy), e.g. `:rtr`, `:xty`, `:sts`.

## Compatible estimators

| Estimator | Notes |
|:--|:--|
| `JointProbabilities(def, CodifyVariables(o))` | plug-in probabilities (`RelativeAmount`) |
| `EntropyDecomposition(def, PlugIn(Shannon()), CodifyVariables(o))` | or another discrete Shannon entropy estimator |
| `EntropyDecomposition(def, Kraskov(; k))`, `EntropyDecomposition(def, KozachenkoLeonenko())` | or another differential Shannon entropy estimator |
| `GaussianDecomposition(def; copula)` | closed-form Gaussian entropies |

With a discretization, each series is encoded *once*, over its full length, and the codes
are then lagged, so the past and the future of a variable share one encoding. With an
outcome space that embeds the data (e.g. `OrdinalPatterns`), the lag `τ` counts encoded
symbols. Continuous estimators lag the raw data. Mutual informations are
``H(A) + H(B) - H(A \\cup B)`` from the joint entropies of the estimator, in the unit `base`.

## Example

```julia
using LocalInformation, Associations
x, y = rand(0:1, 10_001), rand(0:1, 10_001)
y[2:end] .= x[1:end-1]                       # x is copied into y
est = JointProbabilities(IntegratedInformationDecomposition(), CodifyVariables(UniqueElements()))
res = association(est, x, y)                 # PhiIDResult
atom(res, :xty)                              # == atom(res, [[1]] => [[2]])
information_dynamics(res).transfer
```

## References

- Mediano, P. A. M., Rosas, F., Carhart-Harris, R. L., Seth, A. K., & Barrett, A. B.
    (2019). Beyond integrated information: A taxonomy of information dynamics phenomena.
    arXiv:1909.02297.
- Mediano, P. A. M., Rosas, F. E., Luppi, A. I., Carhart-Harris, R. L., Bor, D.,
    Seth, A. K., & Barrett, A. B. (2021). Towards an extended taxonomy of information
    dynamics via Integrated Information Decomposition. arXiv:2109.13186.
- Luppi, A. I., Mediano, P. A. M., Rosas, F. E., et al. (2022). A synergistic core for
    human brain evolution and cognition. Nature Neuroscience, 25, 771–782.
"""
struct IntegratedInformationDecomposition{R<:RedundancyFunction,B} <: MultivariateInformationMeasure
    redundancy::R
    τ::Int
    base::B
    function IntegratedInformationDecomposition{R,B}(redundancy::R, τ::Integer, base::B) where {R<:RedundancyFunction,B}
        τ ≥ 1 || throw(ArgumentError("The lag `τ` must be at least 1. Got τ = $τ."))
        return new{R,B}(redundancy, Int(τ), base)
    end
end
function IntegratedInformationDecomposition(redundancy::R, τ::Integer, base::B) where {R<:RedundancyFunction,B}
    return IntegratedInformationDecomposition{R,B}(redundancy, τ, base)
end
function IntegratedInformationDecomposition(redundancy::RedundancyFunction = IMMI(); τ::Integer = 1, base = 2)
    return IntegratedInformationDecomposition(redundancy, τ, base)
end

const PhiID = IntegratedInformationDecomposition

Associations.min_inputs_vars(::PhiID) = 2
Associations.max_inputs_vars(::PhiID) = 3

# Entropy estimators inside an `EntropyDecomposition` use the ΦID's base.
function Associations.estimator_with_overridden_parameters(def::PhiID, est::InformationMeasureEstimator{<:Shannon})
    return Associations.estimator_with_overridden_parameters(MIShannon(; base = def.base), est)
end

# ----------------------------------------------------------------------------------------
# The product lattice, cached per number of variables.
# ----------------------------------------------------------------------------------------
"""
    LocalInformation.PhiIDLattice

Data for the ΦID of `n` variables, built once per `n` by `LocalInformation.phiid_lattice(n)`
from Associations' (shared, never modified) PID lattice `redundancy_lattice(n)`, with `m`
nodes (4 for `n = 2`, 18 for `n = 3`) and `m²` ΦID atoms.

## Fields

- `lattice::RedundancyLattice`: the PID lattice. Nodes are referred to by their index in
    `lattice.nodes` ("node order").
- `zeta::BitMatrix`: the `m × m` zeta matrix in the order `lattice.order`,
    `zeta[a, b] = 1` iff node `order[a]` ≼ node `order[b]`; unit upper triangular.
- `mobius::Matrix{Int}`: ``Z^{-1}`` (the Möbius function, exact) in node order:
    `mobius[i, j]` = ``\\mu(\\alpha_i, \\alpha_j)``. For redundancies `R` (`m × m`, node
    order, rows = past) the atoms are `mobius' * R * mobius`.
- `kron::Matrix{Int}`: `kron(mobius, mobius)`, `m² × m²`. For local redundancies as rows
    of a `K × m²` matrix in the `vec` layout (column `i + (j - 1) m` for past node `i`
    and future node `j`), the local atoms are `R * kron`, in the same layout.
- `pairs::Vector{CartesianIndex{2}}`: all `(past node, future node)` pairs from the bottom
    to the top of the product lattice (lexicographic in `lattice.order`, past first; a
    linear extension of the product order). For `n = 2` this is the order of `phyid`:
    `rtr, rtx, rty, rts, xtr, …, sts`.
"""
struct PhiIDLattice
    lattice::RedundancyLattice
    zeta::BitMatrix
    mobius::Matrix{Int}
    kron::Matrix{Int}
    pairs::Vector{CartesianIndex{2}}
end

function PhiIDLattice(n::Int)
    L = redundancy_lattice(n)
    m = length(L.nodes)
    o = L.order
    zeta = L.below[o, o]
    for a in 1:m
        zeta[a, a] = true
    end
    # Exact inverse of the unit upper triangular 0/1 matrix by back substitution.
    μo = zeros(Int, m, m)
    for j in 1:m
        μo[j, j] = 1
        for i in (j - 1):-1:1
            s = 0
            for k in (i + 1):j
                zeta[i, k] && (s += μo[k, j])
            end
            μo[i, j] = -s
        end
    end
    μ = zeros(Int, m, m)
    μ[o, o] = μo
    pairs = [CartesianIndex(o[a], o[b]) for a in 1:m for b in 1:m]
    return PhiIDLattice(L, zeta, μ, kron(μ, μ), pairs)
end

const PHIID_LATTICES = Dict{Int,PhiIDLattice}()
const PHIID_LATTICES_LOCK = ReentrantLock()

"""
    LocalInformation.phiid_lattice(n::Int) → PhiIDLattice

The [`LocalInformation.PhiIDLattice`](@ref) of `n` variables (`1 ≤ n ≤ 3`), built once
and cached (thread-safe). Do not modify it.
"""
function phiid_lattice(n::Int)
    1 ≤ n ≤ 3 || throw(ArgumentError("The ΦID lattice is implemented for 1 to 3 variables; got $n."))
    lock(PHIID_LATTICES_LOCK) do
        get!(() -> PhiIDLattice(n), PHIID_LATTICES, n)
    end
end

nvariables(L::PhiIDLattice) = L.lattice.nsources
nnodes(L::PhiIDLattice) = length(L.lattice.nodes)

"""
    LocalInformation.phiid_mobius(lattice::PhiIDLattice, R::AbstractMatrix) → Matrix

The ΦID atoms ``\\Pi = Z^{-\\top} R Z^{-1}`` of the double redundancies `R` (`m × m`, node
order, rows = past nodes), in the same layout: `lattice.mobius' * R * lattice.mobius`.
The Möbius matrix is exact (integers), so integer redundancies give exact atoms.
"""
function phiid_mobius(L::PhiIDLattice, R::AbstractMatrix)
    m = nnodes(L)
    size(R) == (m, m) || throw(ArgumentError(
        "Double redundancies must be an $m × $m matrix (one row per past node, one column " *
        "per future node); got size $(size(R))."))
    return L.mobius' * R * L.mobius
end

# ----------------------------------------------------------------------------------------
# Atom names.
# ----------------------------------------------------------------------------------------
# For two variables: r = {1}{2}, x = {1}, y = {2}, s = {12} (phyid's names).
const PHIID_LETTERS = ('r' => [[1], [2]], 'x' => [[1]], 'y' => [[2]], 's' => [[1, 2]])

"""
    LocalInformation.PHIID_INFORMATION_DYNAMICS

The six information-dynamics modes of a two-variable ΦID (Mediano et al. 2021, 2025), as
sums of named atoms (see [`information_dynamics`](@ref)); they partition the 16 atoms.
The single place to change the definitions.
"""
const PHIID_INFORMATION_DYNAMICS = (
    storage = (:rtr, :xtx, :yty, :sts),
    copy = (:xtr, :ytr),
    transfer = (:xty, :ytx),
    erasure = (:rtx, :rty),
    downward_causation = (:str, :stx, :sty),
    upward_causation = (:rts, :xts, :yts),
)

function letter_node(c::Char)
    for (l, α) in PHIID_LETTERS
        l == c && return α
    end
    return nothing
end

function symbol_pair(L::PhiIDLattice, s::Symbol)
    n = nvariables(L)
    n == 2 || throw(ArgumentError(
        "Atom names such as `:rtr` are defined for 2 variables only; this ΦID has $n. " *
        "Give the atom as `α => β`, e.g. `atom(res, [[1], [2]] => [[1, 2, 3]])`."))
    str = String(s)
    if length(str) == 3 && str[2] == 't'
        α, β = letter_node(str[1]), letter_node(str[3])
        α === nothing || β === nothing || return α => β
    end
    throw(ArgumentError(
        "Unknown ΦID atom $(repr(s)). Atoms of 2 variables are named `:αtβ`, with α (past) " *
        "and β (future) one of r = {1}{2} (redundancy), x = {1}, y = {2}, s = {12} " *
        "(synergy), e.g. `:rtr`, `:xty` or `:sts`."))
end

pair_symbol(L::PhiIDLattice, c::CartesianIndex{2}) = Symbol(node_letter(L, c[1]), 't', node_letter(L, c[2]))
function node_letter(L::PhiIDLattice, i::Int)
    v = lattice_node_vectors(L.lattice.nodes[i])
    for (l, α) in PHIID_LETTERS
        α == v && return l
    end
    error("Unexpected node $v of the 2-variable lattice.")
end

# Column of the pair `α => β` in the `vec` layout (also the linear index into an m × m matrix).
function phiid_column(L::PhiIDLattice, p::Pair)
    i = node_index(L.lattice, first(p))
    j = node_index(L.lattice, last(p))
    return i + (j - 1) * nnodes(L)
end
phiid_column(L::PhiIDLattice, s::Symbol) = phiid_column(L, symbol_pair(L, s))
function phiid_column(L::PhiIDLattice, a)
    throw(ArgumentError(
        "A ΦID atom is given as `α => β`, with α (past) and β (future) collections of " *
        "variable sets, e.g. `[[1], [2]] => [[1, 2]]`, or, for 2 variables, by name, e.g. " *
        "`:rts`. Got $(repr(a))."))
end

pair_string(L::PhiIDLattice, c::CartesianIndex{2}) =
    node_string(L.lattice.nodes[c[1]]) * " → " * node_string(L.lattice.nodes[c[2]])
function pair_label(L::PhiIDLattice, c::CartesianIndex{2})
    s = pair_string(L, c)
    return nvariables(L) == 2 ? rpad(string(pair_symbol(L, c)), 5) * s : s
end
pair_nodes(L::PhiIDLattice, c::CartesianIndex{2}) =
    lattice_node_vectors(L.lattice.nodes[c[1]]) => lattice_node_vectors(L.lattice.nodes[c[2]])

# ----------------------------------------------------------------------------------------
# Results.
# ----------------------------------------------------------------------------------------
"""
    PhiIDResult

The result of an [`IntegratedInformationDecomposition`](@ref), returned by
`association`.

## Fields

- `definition`: the `IntegratedInformationDecomposition`.
- `lattice`: the [`LocalInformation.PhiIDLattice`](@ref) (shared; do not modify).
- `redundancies::Matrix`: the `m × m` double redundancies
    ``I_\\cap^{\\alpha_i \\to \\alpha_j}``, rows = past node `i`, columns = future node `j`,
    nodes indexed as in `lattice.lattice.nodes`.
- `atoms::Matrix`: the atoms ``\\Pi(\\alpha_i \\to \\alpha_j)`` in the same layout.

## Accessors

- `atom(res, α => β)`: the atom of past node `α` and future node `β`, each a collection of
    variable sets, e.g. `atom(res, [[1], [2]] => [[1, 2]])`.
- `atom(res, name)` for two variables: `name` is `:αtβ` with `α`, `β` one of `r` =
    ``\\{1\\}\\{2\\}``, `x` = ``\\{1\\}``, `y` = ``\\{2\\}``, `s` = ``\\{12\\}``, e.g.
    `atom(res, :rts) == atom(res, [[1], [2]] => [[1, 2]])`.
- `atoms(res)`: every atom as an `(α => β) => value` pair, from the bottom
    (``\\{1\\}\\ldots\\{n\\} \\to \\{1\\}\\ldots\\{n\\}``) to the top
    (``\\{1 \\ldots n\\} \\to \\{1 \\ldots n\\}``) of the product lattice.
- [`information_dynamics`](@ref)`(res)` for two variables.

The atoms sum to ``I(\\mathbf{X}_t; \\mathbf{X}_{t+\\tau})``, in the unit
`definition.base`.
"""
struct PhiIDResult{D<:PhiID,T<:Real}
    definition::D
    lattice::PhiIDLattice
    redundancies::Matrix{T}
    atoms::Matrix{T}
end

"""
    LocalPhiIDResult

A per-sample (local) integrated information decomposition: the local atoms of every
sample of the lagged system, with the same accessors as [`PhiIDResult`](@ref) returning
per-sample vectors (views). Per-sample atoms are the inversion of per-sample double
redundancies, so they need a double redundancy that is a mean of per-sample terms; it is
built from them by `LocalInformation.local_phiid_result`. `local_association` does not
return it yet for any estimator (it throws an `ArgumentError`).

## Fields

- `definition`: the `IntegratedInformationDecomposition`.
- `lattice`: the [`LocalInformation.PhiIDLattice`](@ref) (shared; do not modify).
- `atoms::Matrix{T}`: `(N - τ) × m²`, the local atoms of every sample (rows); column
    `i + (j - 1) m` is the pair (past node `i` → future node `j`), nodes indexed as in
    `lattice.lattice.nodes` (the `vec` layout of the `m × m` atom matrix).

## Accessors

- `atom(res, α => β)`, `atom(res, name)`: the local atom, e.g. `atom(res, :xty)`.
- `atoms(res)`: every atom as an `(α => β) => values` pair, bottom to top.
- [`information_dynamics`](@ref)`(res)` (two variables): a `NamedTuple` of vectors.

The local atoms of a sample sum to its local time-delayed mutual information, and the
mean of every local atom is the corresponding average atom.
"""
struct LocalPhiIDResult{D<:PhiID,T<:Real}
    definition::D
    lattice::PhiIDLattice
    atoms::Matrix{T}
end

const PhiIDResults = Union{PhiIDResult,LocalPhiIDResult}

Associations.atom(res::PhiIDResult, a) = res.atoms[phiid_column(res.lattice, a)]
Associations.atom(res::LocalPhiIDResult, a) = view(res.atoms, :, phiid_column(res.lattice, a))

function Associations.atoms(res::PhiIDResult)
    L = res.lattice
    return [pair_nodes(L, c) => res.atoms[c] for c in L.pairs]
end
function Associations.atoms(res::LocalPhiIDResult)
    L = res.lattice
    m = nnodes(L)
    return [pair_nodes(L, c) => view(res.atoms, :, c[1] + (c[2] - 1) * m) for c in L.pairs]
end

"""
    information_dynamics(res::PhiIDResult) → NamedTuple
    information_dynamics(res::LocalPhiIDResult) → NamedTuple of vectors

The six information-dynamics modes of a two-variable ΦID (Mediano et al. 2021, 2025), as
sums of atoms (named as in [`PhiIDResult`](@ref): first letter past, last letter future):

| Mode | Atoms |
|:--|:--|
| `storage` | `rtr + xtx + yty + sts` |
| `copy` | `xtr + ytr` |
| `transfer` | `xty + ytx` |
| `erasure` | `rtx + rty` |
| `downward_causation` | `str + stx + sty` |
| `upward_causation` | `rts + xts + yts` |

The modes are disjoint and cover all 16 atoms, so they sum to
``I(\\mathbf{X}_t; \\mathbf{X}_{t+\\tau})``; `sts` (synergistic storage) is part of
`storage`. For a `LocalPhiIDResult` each entry is a per-sample vector. Throws an
`ArgumentError` for more than two variables.

## References

- Mediano, P. A. M., Rosas, F. E., Luppi, A. I., et al. (2021). Towards an extended
    taxonomy of information dynamics via Integrated Information Decomposition.
    arXiv:2109.13186.
- Mediano, P. A. M., Rosas, F. E., Luppi, A. I., et al. (2025). Proceedings of the
    National Academy of Sciences, 122(39), e2423297122 (the published version of the
    above).
"""
function information_dynamics(res::PhiIDResult)
    check_two_variables(res)
    return map(syms -> sum(s -> atom(res, s), syms), PHIID_INFORMATION_DYNAMICS)
end
function information_dynamics(res::LocalPhiIDResult{D,T}) where {D,T}
    check_two_variables(res)
    N = size(res.atoms, 1)
    return map(syms -> foldl((acc, s) -> (acc .+= atom(res, s)), syms; init = zeros(T, N)),
        PHIID_INFORMATION_DYNAMICS)
end

function check_two_variables(res)
    n = nvariables(res.lattice)
    n == 2 || throw(ArgumentError(
        "`information_dynamics` is defined for a ΦID of 2 variables; this one has $n. " *
        "Use `atom(res, α => β)` and `atoms(res)`."))
    return nothing
end

function phiid_header(res, kind)
    d = res.definition
    return "$kind: $(nvariables(res.lattice)) variables, τ = $(d.τ), redundancy function " *
        "$(nameof(typeof(d.redundancy))), base $(d.base)"
end

function show_atoms(io::IO, L::PhiIDLattice, value)
    pairs = L.pairs
    nshow = length(pairs) ≤ 30 ? length(pairs) : 20
    w = nvariables(L) == 2 ? 22 : 30
    for c in pairs[1:nshow]
        print(io, "\n  ", rpad(pair_label(L, c), w), value(c))
    end
    nshow < length(pairs) && print(io, "\n  ⋮ ($(length(pairs) - nshow) more atoms; use `atoms(res)`)")
end

function Base.show(io::IO, ::MIME"text/plain", res::PhiIDResult)
    print(io, phiid_header(res, "PhiIDResult"))
    show_atoms(io, res.lattice, c -> res.atoms[c])
    print(io, "\n  total (sum of atoms = I(past; future)): ", sum(res.atoms))
end

function Base.show(io::IO, ::MIME"text/plain", res::LocalPhiIDResult)
    N, c = size(res.atoms)
    print(io, phiid_header(res, "LocalPhiIDResult"), ", $N samples × $c atoms")
    print(io, "\n  atom (mean over samples)")
    m = nnodes(res.lattice)
    show_atoms(io, res.lattice, p -> sum(view(res.atoms, :, p[1] + (p[2] - 1) * m)) / max(N, 1))
end

# ----------------------------------------------------------------------------------------
# Inputs: validation, encoding (once, over the full series) and lagging.
# ----------------------------------------------------------------------------------------
const PHIID_ESTIMATORS = Union{
    JointProbabilities{<:PhiID},
    EntropyDecomposition{<:PhiID},
    GaussianDecomposition{<:PhiID},
}

const PHIID_SUPPORTED_ESTIMATORS = """
Supported estimators: `JointProbabilities(def, CodifyVariables(o))` (plug-in \
probabilities), `EntropyDecomposition(def, PlugIn(Shannon()), CodifyVariables(o))`, \
`EntropyDecomposition(def, Kraskov(; k))`, `EntropyDecomposition(def, KozachenkoLeonenko())` \
and `GaussianDecomposition(def)`, with `def = IntegratedInformationDecomposition(IMMI())`; \
e.g. `association(JointProbabilities(IntegratedInformationDecomposition(), \
CodifyVariables(UniqueElements())), x, y)`."""

function unsupported_phiid_estimator(est, why = "")
    throw(ArgumentError(
        "ΦID is not implemented for a $(describe(est))$why. " * PHIID_SUPPORTED_ESTIMATORS))
end

function check_phiid_count(def::PhiID, n::Integer)
    nmin, nmax = Associations.min_inputs_vars(def), Associations.max_inputs_vars(def)
    if !(nmin ≤ n ≤ nmax)
        throw(ArgumentError(
            "IntegratedInformationDecomposition needs $nmin to $nmax variables (16 or 324 " *
            "atoms; 4 variables would have 166² = 27556); got $n."))
    end
    return nothing
end

# Types and equal lengths of the inputs; returns the number of samples.
function check_phiid_inputs(x::Tuple)
    for (k, xk) in enumerate(x)
        xk isa Union{AbstractVector,AbstractStateSpaceSet} || throw(ArgumentError(
            "Each ΦID input must be an `AbstractVector` or a `StateSpaceSet`; input $k is " *
            "a $(typeof(xk))."))
    end
    return check_input_lengths(x...)
end

"""
    LocalInformation.phiid_series(est, x::Tuple) → (est′, s::Tuple)

The full-length series that ΦID lags, and the estimator to apply to the lagged series.

- Discrete estimators: `s` are the per-sample codes of `x` (relabelled to `1, 2, …` per
    variable), encoded once over the full length with the estimator's discretization,
    exactly as `JointProbabilities` (see `joint_codes`) or `EntropyDecomposition`
    (`decomposition_codes`) encode them; `est′` is the same estimator family with
    `CodifyVariables(UniqueElements())`.
- Continuous estimators: `s = x` and `est′ = est`.

Unsupported estimators throw an `ArgumentError`.
"""
phiid_series(est, x::Tuple) = unsupported_phiid_estimator(est)

function phiid_series(est::JointProbabilities{<:PhiID}, x::Tuple)
    if !(est.discretization isa CodifyVariables) || !(est.pest isa RelativeAmount)
        unsupported_phiid_estimator(est,
            " with discretization $(nameof(typeof(est.discretization))) and probabilities " *
            "estimator $(nameof(typeof(est.pest))) (a `CodifyVariables` discretization and " *
            "the default `RelativeAmount` are needed)")
    end
    codes = relabel_codes(joint_codes(est.discretization, x...))
    return JointProbabilities(est.definition, CodifyVariables(UniqueElements())), codes
end

function phiid_series(est::EntropyDecomposition{<:PhiID,<:DiscreteInfoEstimator{<:Shannon}}, x::Tuple)
    if !(est.discretization isa Union{CodifyVariables,OutcomeSpace})
        unsupported_phiid_estimator(est,
            " with discretization $(nameof(typeof(est.discretization))) (a `CodifyVariables` " *
            "discretization is needed)")
    end
    codes = relabel_codes(decomposition_codes(est.discretization, x...))
    return EntropyDecomposition(est.definition, est.est, CodifyVariables(UniqueElements()), est.pest), codes
end

phiid_series(est::EntropyDecomposition{<:PhiID,<:DifferentialInfoEstimator{<:Shannon}}, x::Tuple) = (est, x)
phiid_series(est::GaussianDecomposition{<:PhiID}, x::Tuple) = (est, x)

# Integer labels 1, 2, … per variable (a bijection, so entropies are unchanged).
relabel_codes(codes) = Tuple(first(relabel(c)) for c in codes)

"""
    LocalInformation.lag_series(s::Tuple, τ::Int) → (past::Tuple, future::Tuple)

`past[k] = s[k][1:N-τ]` and `future[k] = s[k][1+τ:N]` (copies), for series of equal
length `N > τ`.
"""
function lag_series(s::Tuple, τ::Int)
    N = check_input_lengths(s...)
    N > τ || throw(ArgumentError(
        "ΦID with lag τ = $τ needs more than τ samples (encoded symbols, for an outcome " *
        "space that embeds the data) per variable; got $N."))
    return map(v -> v[1:(N - τ)], s), map(v -> v[(1 + τ):N], s)
end

"""
    LocalInformation.phiid_prepare(est, x...) → (est′, past::Tuple, future::Tuple)

Validate the inputs of `association(est, x...)` for a ΦID estimator, encode discrete data
once with [`LocalInformation.phiid_series`](@ref) and lag by `est.definition.τ` with
[`LocalInformation.lag_series`](@ref). `past` and `future` are then given to
[`LocalInformation.phiid_from_past_future`](@ref)`(est′, past, future)`.
"""
function phiid_prepare(est, x...)
    check_phiid_count(est.definition, length(x))
    check_phiid_inputs(x)
    est′, s = phiid_series(est, x)
    past, future = lag_series(s, est.definition.τ)
    return est′, past, future
end

# ----------------------------------------------------------------------------------------
# Average atoms.
# ----------------------------------------------------------------------------------------
function Associations.association(est::MultivariateInformationMeasureEstimator{<:PhiID}, x...)
    est′, past, future = phiid_prepare(est, x...)
    return phiid_from_past_future(est′, past, future)
end

"""
    LocalInformation.phiid_from_past_future(est, past::Tuple, future::Tuple) → PhiIDResult

The ΦID of the system whose past variables are `past` and future variables `future`
(`n = 2` or `3` each, all with the same number of samples), with no lagging or shared
encoding: `est` is applied to the `2n` inputs `(past..., future...)` as given (inputs
`1:n` are the past, `n+1:2n` the future). `association(est, x...)` calls this after
[`LocalInformation.phiid_prepare`](@ref); it is also the entry point for an exact joint
distribution given as samples of `(past, future)` states, e.g. with
`JointProbabilities(def, CodifyVariables(UniqueElements()))` and integer states.

The double redundancies come from
[`LocalInformation.phiid_redundancies`](@ref)`(est.definition.redundancy, est, lattice, past, future)`
and the atoms from [`LocalInformation.phiid_mobius`](@ref). `est.definition.τ` is not used.
"""
function phiid_from_past_future(est, past::Tuple, future::Tuple)
    n = length(past)
    length(future) == n || throw(ArgumentError(
        "The past and the future must have the same number of variables; got " *
        "$n and $(length(future))."))
    check_phiid_count(est.definition, n)
    check_phiid_inputs((past..., future...))
    L = phiid_lattice(n)
    m = nnodes(L)
    R = phiid_redundancies(est.definition.redundancy, est, L, past, future)
    if !(R isa AbstractMatrix) || size(R) != (m, m)
        throw(ArgumentError(
            "`phiid_redundancies` for $(nameof(typeof(est.definition.redundancy))) must " *
            "return an $m × $m matrix; got $(summary(R))."))
    end
    R = Matrix(R)
    return PhiIDResult(est.definition, L, R, phiid_mobius(L, R))
end

"""
    LocalInformation.phiid_redundancies(r::RedundancyFunction, est, lattice::PhiIDLattice, past::Tuple, future::Tuple) → Matrix

The extension hook for double redundancies: return the `m × m` matrix `R` with
`R[i, j]` = ``I_\\cap^{\\alpha_i \\to \\alpha_j}``, the double redundancy of past node `i`
and future node `j` (indices into `lattice.lattice.nodes`), in the unit
`est.definition.base`.

`est` is applied to the `2n` inputs `(past..., future...)` as given (for discrete data
these are integer codes and `est` uses `CodifyVariables(UniqueElements())`, see
[`LocalInformation.phiid_series`](@ref)). `Associations.lattice_node_vectors` reads a node
as variable sets.

## Methods

- `IMMI`, any estimator with `Associations.subset_entropies`:
    `R[i, j] = min(I(past_A; future_B) for A in αᵢ, B in αⱼ)`, with
    `I = H(A) + H(B) - H(A ∪ B)` (one mutual information per pair of variable sets).
- Every other redundancy function throws an `ArgumentError`.

Add a method for a new redundancy function (and estimator), e.g.
`LocalInformation.phiid_redundancies(r::MyRedundancy, est::JointProbabilities, lattice, past, future)`.
"""
function phiid_redundancies end

function phiid_redundancies(::IMMI, est, lattice::PhiIDLattice, past::Tuple, future::Tuple)
    n = length(past)
    H = phiid_subset_entropies(est, past..., future...)
    nm = (1 << n) - 1
    vars(mask) = [k for k in 1:n if (mask >> (k - 1)) & 1 == 1]
    mi = Matrix{Float64}(undef, nm, nm) # mi[A, B] = I(past_A; future_B) for bitmasks A, B
    for a in 1:nm, b in 1:nm
        A, B = vars(a), n .+ vars(b)
        mi[a, b] = H(A) + H(B) - H(vcat(A, B))
    end
    nodes = lattice.lattice.nodes
    return [minimum(mi[A, B] for A in α, B in β) for α in nodes, β in nodes]
end

function phiid_redundancies(r::RedundancyFunction, est, lattice::PhiIDLattice, past::Tuple, future::Tuple)
    name = nameof(typeof(r))
    why = if r isa Union{ISx,IPM}
        "$name is not yet supported for ΦID"
    elseif r isa IMin
        "IMin has no double-redundancy form: its redundancy Σₜ p(t) min_A I(T = t; A) " *
        "averages specific informations over the outcomes of a single target, and ΦID " *
        "needs a redundancy among source sets on both sides (past and future)"
    else
        "$name has no ΦID double redundancy in LocalInformation.jl (add a method to " *
        "`LocalInformation.phiid_redundancies`)"
    end
    throw(ArgumentError(
        "$why. Use `IntegratedInformationDecomposition(IMMI())` (the default), e.g. " *
        "`JointProbabilities(IntegratedInformationDecomposition(IMMI()), " *
        "CodifyVariables(UniqueElements()))`. Got a $(describe(est))."))
end

# Joint entropies of subsets of the 2n inputs. `JointProbabilities` would build the dense
# joint table of all 2n inputs; with `UniqueElements` and plug-in probabilities, counting
# each subset (EntropyDecomposition with PlugIn) gives the same entropies without it.
phiid_subset_entropies(est, x...) = Associations.subset_entropies(est, x...)
function phiid_subset_entropies(
        est::JointProbabilities{<:PhiID,<:CodifyVariables{1,<:UniqueElements},<:RelativeAmount}, x...)
    e = EntropyDecomposition(est.definition, PlugIn(Shannon()), CodifyVariables(UniqueElements()))
    return Associations.subset_entropies(e, x...)
end

# ----------------------------------------------------------------------------------------
# Per-sample atoms.
# ----------------------------------------------------------------------------------------
"""
    LocalInformation.phiid_local_atoms(lattice::PhiIDLattice, R::AbstractMatrix, rows = nothing;
        backend = CPU(), precision = nothing, memory_limit = 2^31) → Matrix{T}

Per-sample ΦID atoms from per-sample double redundancies, with one matrix product
`R * lattice.kron` (i.e. ``\\Pi = Z^{-\\top} R Z^{-1}`` for every row) on `backend`, in
precision `T = LocalInformation.resolve_precision(backend, precision)`. Returns a host
`Matrix{T}`.

- `R`: `K × m²` local double redundancies, one row per distinct joint state (or per
    sample), in the `vec` layout: column `i + (j - 1) m` is the pair (past node `i` →
    future node `j`), nodes indexed as in `lattice.lattice.nodes`. Any unit; the atoms
    have the same.
- `rows`: `nothing` (the rows of `R` are the samples; the result is `K × m²`) or a vector
    of `N` row indices in `1:K`, the row of `R` of each sample (the result is `N × m²`,
    row `t` holding the atoms of row `rows[t]`).

By linearity, the mean of the local atoms is the inversion of the mean local redundancy.
An `ArgumentError` is thrown before any large allocation if the estimated memory (bytes)
exceeds `memory_limit`.
"""
function phiid_local_atoms(L::PhiIDLattice, R::AbstractMatrix, rows = nothing;
        backend = CPU(), precision = nothing, memory_limit::Real = 2^31)
    T = resolve_precision(backend, precision)
    m2 = nnodes(L)^2
    K = size(R, 1)
    size(R, 2) == m2 || throw(ArgumentError(
        "Local double redundancies must have $m2 columns (one per ΦID atom of " *
        "$(nvariables(L)) variables); got $(size(R, 2))."))
    if rows !== nothing
        rows isa AbstractVector{<:Integer} || throw(ArgumentError(
            "`rows` must be `nothing` or a vector of row indices; got $(typeof(rows))."))
        isempty(rows) || (1 ≤ minimum(rows) && maximum(rows) ≤ K) || throw(ArgumentError(
            "`rows` must index the $K rows of the local redundancies; got values in " *
            "$(extrema(rows))."))
    end
    N = rows === nothing ? K : length(rows)
    check_phiid_memory(backend, T, eltype(R), K, N, rows !== nothing, m2, memory_limit, nvariables(L))
    A = to_host(to_device(backend, R, T) * to_device(backend, L.kron, T))
    return rows === nothing ? A : A[rows, :]
end

# Estimated peak memory (bytes) of `phiid_local_atoms`.
function check_phiid_memory(backend, ::Type{T}, ::Type{S}, K, N, expand, m2, limit, n) where {T,S}
    s = sizeof(T)
    oncpu = backend isa KA.CPU
    need = (oncpu && T === S) ? 0 : s * K * m2  # converted / device copy of R
    need += s * m2^2 * (oncpu ? 1 : 2)          # Kronecker Möbius matrix (host and device)
    need += s * K * m2 * (oncpu ? 1 : 2)        # product (device and host)
    need += expand ? s * N * m2 : 0             # rows copied to the samples
    if need > limit
        throw(ArgumentError(
            "Local ΦID atoms ($m2 atoms of $n variables) for $N samples ($K distinct rows) " *
            "would need about $(round(need / 2^30; digits = 2)) GiB, more than " *
            "`memory_limit` = $(round(limit / 2^30; digits = 2)) GiB. Use fewer samples or " *
            "raise `memory_limit`."))
    end
    return need
end

"""
    LocalInformation.local_phiid_result(definition, lattice, R, rows = nothing; kwargs...) → LocalPhiIDResult

A [`LocalPhiIDResult`](@ref) from per-sample double redundancies `R` (`K × m²`) and the
row of each sample, computed by [`LocalInformation.phiid_local_atoms`](@ref) (the keywords
are passed on).
"""
function local_phiid_result(definition::PhiID, L::PhiIDLattice, R::AbstractMatrix, rows = nothing; kwargs...)
    return LocalPhiIDResult(definition, L, phiid_local_atoms(L, R, rows; kwargs...))
end

# No per-sample ΦID yet. Later phases add more specific methods (e.g. for
# `JointProbabilities{<:IntegratedInformationDecomposition{<:ISx}}`).
function local_association(est::PHIID_ESTIMATORS, x...; kwargs...)
    r = nameof(typeof(est.definition.redundancy))
    throw(ArgumentError(
        "Per-sample (local) ΦID atoms are not implemented for a $(describe(est)) with " *
        "redundancy function $r. Use `association(est, x...)` for the average atoms (a " *
        "`PhiIDResult`)."))
end
