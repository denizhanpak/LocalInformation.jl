# Local values of the dedicated k-nearest-neighbour MI/CMI estimators of Associations.jl
# (`KSG1`, `KSG2` for `MIShannon`; `FPVP` for `CMIShannon`), and the GPU k-NN kernels.
# "GPU" below means any KernelAbstractions backend other than `CPU()` (KernelAbstractions
# 0.10 has no separate `GPU` type).
#
# Associations' estimates are means of per-sample terms; we return those terms. With M
# marginals x₁, …, x_M, the joint space (x₁, …, x_M) and N samples, in nats:
#
#   KSG1: i_t = ψ(k) + (M - 1) ψ(N) - Σ_m ψ(n_m(t))
#   KSG2: i_t = ψ(k) - (M - 1) + (M - 1) ψ(N) - Σ_m ψ(θ_m(t))
#   FPVP: i_t = ψ(k) - ψ(n_xz(t)) - ψ(n_yz(t)) + ψ(n_z(t))
#
# with exactly Associations' conventions (see `KSG1.jl`, `KSG2.jl`, `FPVP.jl` there):
#
# - ε_t is the Chebyshev distance from joint point t to its k-th nearest joint neighbour
#   among the points s with |t - s| > w (Theiler window w; w = 0 excludes only t itself).
# - KSG1 and FPVP: n(t) = #{s : d(u_s, u_t) ≤ ε_t} in the subspace u (`metric_marginals`
#   for KSG1, Chebyshev for FPVP). The inequality is `≤` (`NearestNeighbors.inrangecount`),
#   the count includes t itself (so ψ(n) plays the role of the paper's ψ(n + 1)), and no
#   Theiler window is applied. Because of `≤`, the k-th joint neighbour is counted in the
#   marginal(s) where it attains ε_t, whereas Kraskov et al. (2004) count strictly inside.
# - KSG2: with j(t) the index of the k-th joint neighbour, ε_m(t) = d_m(x_m(t), x_m(j(t)))
#   + 10 eps (only the k-th neighbour, not the maximum over the k neighbours as in the
#   paper), θ_m(t) = #{s ≠ t : d_m(x_m(s), x_m(t)) ≤ ε_m(t)}, and the constant is M - 1
#   (Associations' choice) rather than the paper's (M - 1)/k.

using NearestNeighbors: inrangecount
using Associations: KSG1, KSG2, FPVP

include("gpu_knn.jl")

function check_knn_arguments(N, k, w)
    k ≥ 1 || throw(ArgumentError("`k` must be ≥ 1. Got k = $k."))
    w ≥ 0 || throw(ArgumentError("The Theiler window `w` must be ≥ 0. Got w = $w."))
    if N - (2w + 1) < k
        throw(ArgumentError(
            "Too few points ($N) for k = $k neighbours outside a Theiler window w = $w."
        ))
    end
    return nothing
end

# The type the neighbour searches and range counts compute in: `Float64` on the CPU (the
# k-d tree; results are converted to `T` at the end), `T` on a GPU.
compute_precision(::KA.CPU, ::Type{T}) where {T} = Float64
compute_precision(::KA.Backend, ::Type{T}) where {T} = T

"""
    LocalInformation.kth_neighbours(backend, P::AbstractMatrix, k, w, metric, T) → (dists, idxs)

Distance to and index of the `k`-th nearest neighbour of every column of `P` outside the
Theiler window `w`. The CPU method searches a `KDTree` point by point, as
`Neighborhood.bulksearch(tree, points, NeighborNumber(k), Theiler(w))` does in
Associations (so ties are resolved identically); the GPU method is
[`LocalInformation.gpu_knn`](@ref).
"""
function kth_neighbours(::KA.CPU, P::AbstractMatrix, k::Int, w::Int, metric, ::Type)
    tree = KDTree(P, metric)
    N = size(P, 2)
    ds = Vector{Float64}(undef, N)
    js = Vector{Int}(undef, N)
    for i in 1:N
        idx, d = knn(tree, view(P, :, i), k, true, j -> abs(i - j) ≤ w)
        ds[i], js[i] = d[k], idx[k]
    end
    return ds, js
end
kth_neighbours(backend::KA.Backend, P::AbstractMatrix, k::Int, w::Int, metric, ::Type{T}) where {T} =
    gpu_knn(backend, P, k, w, metric, T)

"""
    LocalInformation.range_counts(backend, P::AbstractMatrix, r::AbstractVector, metric, T) → Vector{Int}

For every column `i` of `P`, the number of columns `j` with `metric(P[:, i], P[:, j]) ≤ r[i]`,
including `i` itself and without Theiler window: `NearestNeighbors.inrangecount` on the CPU,
[`LocalInformation.gpu_range_counts`](@ref) on a GPU.
"""
function range_counts(::KA.CPU, P::AbstractMatrix, r::AbstractVector, metric, ::Type)
    tree = KDTree(P, metric)
    return [inrangecount(tree, view(P, :, i), r[i]) for i in axes(P, 2)]
end
range_counts(backend::KA.Backend, P::AbstractMatrix, r::AbstractVector, metric, ::Type{T}) where {T} =
    gpu_range_counts(backend, P, r, metric, T)

# D_m × N Float64 matrices of the marginals, as Associations builds them (`StateSpaceSet(xᵢ)`).
marginal_matrices(x...) = map(xᵢ -> point_matrix(StateSpaceSet(xᵢ)), x)

"""
    LocalInformation.ksg_counts(est, x...; backend = CPU(), precision = nothing) → Vector{Vector{Int}}

The per-sample neighbour counts behind the local values of `est`, one vector per term:

- `KSG1`: ``n_m(t)`` for every marginal `m` (`≤ ε_t`, the point itself included);
- `KSG2`: ``θ_m(t)`` for every marginal `m` (`≤ ε_m(t)`, the point itself excluded);
- `FPVP`: ``[n_{xz}, n_{yz}, n_z]`` (`≤ ε_t`, the point itself included).

On a GPU in `Float32`, a distance that ties with (or rounds across) a radius can shift a
count by one relative to the `Float64` CPU computation.
"""
function ksg_counts(est::Union{KSG1,KSG2}, x...; backend = CPU(), precision = nothing)
    T = resolve_precision(backend, precision)
    Associations.verify_number_of_inputs_vars(est.definition, length(x))
    N = check_input_lengths(x...)
    check_knn_arguments(N, est.k, est.w)
    Tc = compute_precision(backend, T)
    Ps = marginal_matrices(x...)
    joint = reduce(vcat, Ps)
    if est isa KSG1
        ε = knn_distances(backend, joint, est.k, est.w, Chebyshev(), Tc)
        return [range_counts(backend, P, ε, est.metric_marginals, Tc) for P in Ps]
    end
    _, js = kth_neighbours(backend, joint, est.k, est.w, Chebyshev(), Tc)
    θ = Vector{Vector{Int}}(undef, length(Ps))
    for (m, P) in enumerate(Ps)
        Pc = Matrix{Tc}(P)
        εm = [pad_radius(est.metric_marginals(view(Pc, :, i), view(Pc, :, js[i]))) for i in 1:N]
        θ[m] = range_counts(backend, P, εm, est.metric_marginals, Tc) .- 1
    end
    return θ
end

function ksg_counts(est::FPVP, x, y, z; backend = CPU(), precision = nothing)
    T = resolve_precision(backend, precision)
    N = check_input_lengths(x, y, z)
    check_knn_arguments(N, est.k, est.w)
    Tc = compute_precision(backend, T)
    X, Y, Z = marginal_matrices(x, y, z)
    ε = knn_distances(backend, vcat(X, Y, Z), est.k, est.w, est.metric_joint, Tc)
    return [range_counts(backend, P, ε, est.metric_marginals, Tc) for P in (vcat(X, Z), vcat(Y, Z), Z)]
end

# KSG2 radius ε_m(t) + δ. Associations adds δ = `1e1*eps()` (absolute) so that the k-th
# neighbour itself is counted despite rounding (`sqrt`, then `ε^2` in the k-d tree for
# `Euclidean`). In lower precision, 10 eps(T) would be a much wider margin than in `Float64`
# for small distances (and too narrow for large ones), so a few ulps of ε are added instead.
pad_radius(d::Float64) = d + 1e1 * eps()
pad_radius(d::AbstractFloat) = d + 8 * eps(d)

"""
    local_association(est::Union{KSG1,KSG2}, x, y; backend = CPU(), precision = nothing) → Vector
    local_association(est::FPVP, x, y, z; backend = CPU(), precision = nothing) → Vector

Local values of the dedicated k-NN estimators of Associations.jl, in nats converted to
`est.definition.base`, with `mean(local_association(est, x...)) ≈ association(est, x...)`:

- `KSG1`: ``ψ(k) + ψ(N) - ψ(n_x(t)) - ψ(n_y(t))``;
- `KSG2`: ``ψ(k) - 1 + ψ(N) - ψ(θ_x(t)) - ψ(θ_y(t))``;
- `FPVP`: ``ψ(k) - ψ(n_{xz}(t)) - ψ(n_{yz}(t)) + ψ(n_z(t))``,

with the counts of [`LocalInformation.ksg_counts`](@ref) (Associations' conventions:
Chebyshev joint `k`-th neighbour outside the Theiler window `w`; marginal counts `≤` the
radius, including the point itself for `KSG1`/`FPVP`, without Theiler window). On a GPU
backend the neighbour search and the counts use exact brute-force kernels.
"""
function local_association(est::KSG1{<:MIShannon}, x...; backend = CPU(), precision = nothing)
    T = resolve_precision(backend, precision)
    n = ksg_counts(est, x...; backend, precision = T)
    M, N = length(n), length(first(n))
    c = digamma(est.k) + (M - 1) * digamma(N)
    l = T[c - sum(m -> digamma(n[m][t]), 1:M) for t in 1:N]
    return nats_to_unit!(l, est.definition.base)
end

function local_association(est::KSG2{<:MIShannon}, x...; backend = CPU(), precision = nothing)
    T = resolve_precision(backend, precision)
    θ = ksg_counts(est, x...; backend, precision = T)
    M, N = length(θ), length(first(θ))
    c = digamma(est.k) - (M - 1) + (M - 1) * digamma(N)
    l = T[c - sum(m -> digamma(θ[m][t]), 1:M) for t in 1:N]
    return nats_to_unit!(l, est.definition.base)
end

function local_association(est::FPVP{<:CMIShannon}, x, y, z; backend = CPU(), precision = nothing)
    T = resolve_precision(backend, precision)
    nxz, nyz, nz = ksg_counts(est, x, y, z; backend, precision = T)
    ψk = digamma(est.k)
    l = T[ψk - digamma(nxz[t]) - digamma(nyz[t]) + digamma(nz[t]) for t in eachindex(nz)]
    return nats_to_unit!(l, est.definition.base)
end
